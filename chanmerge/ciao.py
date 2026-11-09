"""Calls to ciao tools. list arguments only (never shell=True), stdin closed, timeouts, a private PFILES folder.

every chandra_repro is followed by 'punlearn ardlib' for that observation, also when repro fails.
"""
import glob
import os
import shutil
import subprocess
from datetime import datetime, timezone

REPRO_MARKER = ".chanmerge_repro_done"
TIMEOUTS = {"download_chandra_obsid": 7200, "chandra_repro": 7200, "merge_obs": 7200}
DEFAULT_TIMEOUT = 1800


class ToolError(RuntimeError):
    def __init__(self, tool, code, output):
        super().__init__(f"{tool} failed (exit {code}): {output[-600:]}")
        self.tool, self.code, self.output = tool, code, output


def _subprocess_runner(args, cwd=None, env=None, timeout=None):
    r = subprocess.run(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=timeout, errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


class Ciao:
    """one per run. runner(args, cwd, env, timeout) -> (code, output) can be replaced in tests."""

    def __init__(self, pfiles_dir, log_path=None, runner=None):
        self.runner = runner or _subprocess_runner
        self.log_path = log_path
        os.makedirs(pfiles_dir, exist_ok=True)
        sys_part = os.environ.get("PFILES", "").split(";")[-1]
        if not sys_part and os.environ.get("ASCDS_INSTALL"):
            sys_part = os.path.join(os.environ["ASCDS_INSTALL"], "param")
        self.env = dict(os.environ, PFILES=f"{pfiles_dir};{sys_part}")
        self.calls = []

    def log(self, text):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text if text.endswith("\n") else text + "\n")

    def run(self, args, cwd=None, check=True, timeout=None):
        args = [str(a) for a in args]
        tool = args[0]
        self.calls.append(args)
        self.log(f"$ {' '.join(args)}" + (f"   (in {cwd})" if cwd else ""))
        code, out = self.runner(args, cwd=cwd, env=self.env, timeout=timeout or TIMEOUTS.get(tool, DEFAULT_TIMEOUT))
        self.log(out)
        if check and code != 0:
            raise ToolError(tool, code, out)
        return code, out

    # ---------- download and repro ----------
    def download(self, parent, obsid, grating):
        """download_chandra_obsid into parent/<obsid>[_g]; resumes; reruns once on 'file sizes do not match'"""
        oid = str(int(obsid))
        plain, final = os.path.join(parent, oid), os.path.join(parent, oid + "_g" if grating else oid)
        # the tool always writes <obsid>/; a grating folder is moved there and back so resuming works
        if grating and os.path.isdir(final) and not os.path.exists(plain):
            os.rename(final, plain)
        try:
            out = ""
            for _ in range(2):
                code, out = self.run(["download_chandra_obsid", oid], cwd=parent, check=False)
                if code == 0 and "do not match" not in out.lower():
                    break
            else:
                raise ToolError("download_chandra_obsid", code, out or "file sizes do not match after a second try")
            if code != 0:
                raise ToolError("download_chandra_obsid", code, out)
        finally:
            if grating and os.path.isdir(plain):
                if os.path.exists(final):
                    # never overwrite: merge what is there (fov1 may have been fetched before)
                    for root, _, files in os.walk(plain):
                        rel = os.path.relpath(root, plain)
                        os.makedirs(os.path.join(final, rel), exist_ok=True)
                        for f in files:
                            dst = os.path.join(final, rel, f)
                            if not os.path.exists(dst):
                                os.replace(os.path.join(root, f), dst)
                    shutil.rmtree(plain, ignore_errors=True)
                else:
                    os.rename(plain, final)
        if not glob.glob(os.path.join(glob.escape(final), "primary", "*evt2.fits*")):
            raise ToolError("download_chandra_obsid", 0, f"no evt2 file in {final}/primary after the download")
        return final

    def repro(self, obs_dir):
        """chandra_repro once per observation, done-marker after exit 0. punlearn ardlib right after, always."""
        repro_dir = os.path.join(obs_dir, "repro")
        marker = os.path.join(repro_dir, REPRO_MARKER)
        if os.path.exists(marker):
            return "reused"
        try:
            self.run(["chandra_repro", f"indir={obs_dir}", f"outdir={repro_dir}", "clobber=yes", "verbose=1"],
                     cwd=os.path.dirname(obs_dir))
        finally:
            # chandra_repro points ardlib at this observation's bad pixel file; reset before anything else runs
            self.run(["punlearn", "ardlib"], check=False)
        # exit 0 is not enough: the reprocessed event file must be there too
        if not glob.glob(os.path.join(glob.escape(repro_dir), "*_repro_evt2.fits")):
            raise ToolError("chandra_repro", 0, f"exit 0 but no *_repro_evt2.fits in {repro_dir}")
        with open(marker, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat() + "\n")
        return "done"

    # ---------- merging ----------
    def merge_obs(self, outroot, infiles, band_spec, asol=(), bpix=(), msk=(), dtf=()):
        """one merge_obs per (group, chip set, band); stacks written beside outroot"""
        d = os.path.dirname(outroot)
        os.makedirs(d, exist_ok=True)
        def stack(name, items):
            p = os.path.join(d, f".{os.path.basename(outroot)}_{name}.lis")
            with open(p, "w") as f:
                f.write("\n".join(items) + "\n")
            return "@" + p
        args = ["merge_obs", stack("infiles", infiles), outroot, f"bands={band_spec}", "clobber=yes", "cleanup=yes"]
        if asol:
            args.append(f"asolfiles={stack('asol', asol)}")
        if bpix:
            args.append(f"badpixfiles={stack('bpix', bpix)}")
        if msk:
            args.append(f"maskfiles={stack('msk', msk)}")
        if dtf:
            args.append(f"dtffiles={stack('dtf', dtf)}")
        self.run(args, cwd=d)

    def dmmerge(self, files, columns, out):
        filt = "[cols " + ",".join(columns) + "]"
        d = os.path.dirname(out)
        p = os.path.join(d, "." + os.path.basename(out) + "_in.lis")
        with open(p, "w") as f:
            f.write("\n".join(fl + filt for fl in files) + "\n")
        self.run(["dmmerge", "@" + p, out, "clobber=yes"], cwd=d)

    # ---------- streak and arms ----------
    def acisreadcorr(self, evt_in, evt_out, asol, x, y, bkg_region, ccd, dx=16, dy=100):
        d = os.path.dirname(evt_out)
        bkg = os.path.splitext(evt_out)[0] + "_bkg.pi"
        self.run(["dmextract", f"{evt_in}[ccd_id={ccd},sky={bkg_region}][bin pi=1:1024:1]", bkg, "opt=pha1", "clobber=yes"], cwd=d)
        self.run(["acisreadcorr", evt_in, evt_out, f"aspect={asol}", f"x={x}", f"y={y}", f"dx={dx}", f"dy={dy}",
                  f"bkg={bkg}", "bgroup=25", "random=1", "clobber=yes"], cwd=d)

    def tg_resolve_events(self, evt, mask, asol, out):
        """order-sort the events again with another region file; asol: list of asol1 files.
        used by: run.Runner.letg_imaging (ACIS + LETG imaging copy with a narrow arm box)"""
        d = os.path.dirname(out)
        if len(asol) > 1:
            p = os.path.join(d, "." + os.path.basename(out) + "_asol.lis")
            with open(p, "w") as f:
                f.write("\n".join(asol) + "\n")
            aca = "@" + p
        else:
            aca = asol[0]
        self.run(["tg_resolve_events", f"infile={evt}", f"outfile={out}", f"regionfile={mask}", f"acaofffile={aca}",
                  "osipfile=CALDB", "clobber=yes", "mode=h"], cwd=d)

    def fluximage(self, evt, outroot, band_spec, asol=(), bpix=(), msk=(), dtf=(), binsize=8):
        """one observation's image and exposure map; merge_obs makes the real images.
        used by: run.Runner.letg_imaging (exposure map to measure the leak beside each candidate LETG box)"""
        d = os.path.dirname(outroot)
        os.makedirs(d, exist_ok=True)
        args = ["fluximage", evt, outroot, f"bands={band_spec}", f"binsize={binsize}", "clobber=yes", "cleanup=yes"]
        for key, items in (("asolfile", asol), ("badpixfile", bpix), ("maskfile", msk), ("dtffile", dtf)):
            if items:
                args.append(f"{key}={','.join(items)}")
        self.run(args, cwd=d)

    def make_arm_mask(self, evt2, workdir):
        """last step of the mask source order: tgdetect2 + tg_create_mask with mode=h and a position table.
        returns the mask path; the caller checks the content, not only the exit code"""
        os.makedirs(workdir, exist_ok=True)
        src = os.path.join(workdir, "tgdetect2_src2.fits")
        mask = os.path.join(workdir, "tg_mask.fits")
        self.run(["tgdetect2", f"infile={evt2}", f"outfile={src}", "clobber=yes", "mode=h"], cwd=workdir)
        self.run(["tg_create_mask", f"infile={evt2}", f"outfile={mask}", f"input_pos_tab={src}", "clobber=yes", "mode=h"], cwd=workdir)
        return mask


def ciao_available():
    return all(shutil.which(t) for t in ("merge_obs", "chandra_repro", "download_chandra_obsid", "punlearn"))


# ---------- files of an observation ----------
def repro_files(obs_dir):
    # the folder part is escaped: a root path with * or ? would match sibling folders (test o)
    r = glob.escape(os.path.join(obs_dir, "repro"))
    evt = sorted(glob.glob(os.path.join(r, "*_repro_evt2.fits")))
    return dict(evt2=evt[0] if evt else None,
                asol=sorted(glob.glob(os.path.join(r, "pcadf*_asol1.fits"))),
                bpix=sorted(glob.glob(os.path.join(r, "*_repro_bpix1.fits"))),
                msk=sorted(glob.glob(os.path.join(r, "*_msk1.fits"))),
                dtf=sorted(glob.glob(os.path.join(r, "*_dtf1.fits"))))


def primary_evt2(obs_dir):
    p = sorted(glob.glob(os.path.join(glob.escape(obs_dir), "primary", "*evt2.fits*")))
    return p[0] if p else None


def read_header(evt):
    from astropy.io import fits
    h = fits.getheader(evt, "EVENTS")
    keys = ["DATE-OBS", "DATE-END", "EXPOSURE", "INSTRUME", "DETNAM", "GRATING", "READMODE", "DATAMODE", "NROWS",
            "RA_TARG", "DEC_TARG", "OBS_ID", "TSTART", "TSTOP"]
    return {k: h.get(k) for k in keys}


def header_mode_window(hdr):
    """(mode, window) from evt2 header keywords; hrc -> ('-', '-')"""
    if str(hdr.get("INSTRUME", "")).upper().startswith("HRC"):
        return "-", "-"
    rm = str(hdr.get("READMODE") or "").upper()
    dm = str(hdr.get("DATAMODE") or "").upper()
    mode = "CC" if (rm.startswith("CONTINUOUS") or dm.startswith("CC")) else "TE"
    nrows = hdr.get("NROWS")
    window = "sub" if (nrows is not None and int(nrows) < 1024) else "full"
    return mode, window


def parse_header_time(text):
    if not text:
        return None
    t = datetime.strptime(str(text)[:19], "%Y-%m-%dT%H:%M:%S")
    return t.replace(tzinfo=timezone.utc)
