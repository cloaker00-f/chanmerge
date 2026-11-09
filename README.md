# Chanmerge
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Topic](https://img.shields.io/badge/topic-astrophysics-red)

**Chanmerge** finds, downloads, reprocesses and merges **Chandra X-ray Observatory** observations into
exposure-corrected flux images, grouped in time, with tables that say exactly what was done to every image.
It was built for dust scattering halo (DSH) work, where merging observations far apart in time, grating arms,
readout streaks and chip-class steps can all mislead an analysis, so every one of these is a visible choice.

Chanmerge is part of the **Chandra Archival Fluxed Image Generation (CAFIG)** project, which turns the Chandra archive into flux images
for a machine-learning search for dust scattering halos.

Version 2.0.0. It breaks 1.x usage: prompts, layout, file names and the Python API changed.

---

## Requirements
* **CIAO 4.16+** initialised in the shell (`conda activate ciao-4.18` or similar); its Python (3.10+) runs chanmerge.
* astropy, astroquery, numpy (shipped with CIAO's Python or installed by pip).

```bash
pip install .
chanmerge
```

---

## The terminal flow
All questions come before any download.

1. RA, Dec (`hh:mm:ss` / `dd:mm:ss` or **decimal degrees**; a plain decimal is degrees) and a radius in arcmin.
2. Output folder name (blank = built from the coordinates). Characters that break CIAO tools are refused.
3. Optional start and end date (UTC, inclusive, by observation start).
4. Search: HEASARC `chanmaster` (archived and public only) plus the Chandra Ocat for mode, subarray and active chips.
   If Ocat cannot be reached you may go on; unknown modes are checked in the headers after the downloads.
5. Observations crossing a date edge: keep or not.
6. Category table (detector, grating, TE/CC, full frame/subarray) with one stage per row:
   `none` / `download` / `repro` / `flux` (each includes the ones before). CC has no flux stage.
7. Per-observation stage edits (`105 flux`, `103 104 none`, `done`).
8. Flux settings: band(s), chips per ACIS row (the screen shows which chips are on and which chip the entered
   coordinate falls on, read from each observation's small fov1 file), the grating arm treatment, the time window.
9. Which rows are merged together (`separate`, `1+3, 2`, `1+2, rest`), the readout-streak treatment per set, then the
   groups with edits: `move 102 g03`, `move 102 s`, `add 102 g03`, `new 101 102`, `drop 102`, `drop 102 g03`,
   `drop all g03`, `delete g03` (only an empty group), `g01 chips I3`, `done`. Only `move` and `drop` take an
   observation out of a group; `add` and `new` keep it where it was (shared data, reported in the tables).
10. Plan check: what CIAO cannot do is rejected (ACIS + HRC, HRC-I + HRC-S, CC); doubtful choices warn.
11. Confirmation with the real download size, the disk footprint and free disk.
12. Processing runs on its own and ends with a report.
    Only when Ocat could not be reached: if a header shows CC mode for an observation in a group you made by hand,
    that group is not merged (automatic groups just lose the observation, with a note); the other groups finish.
    The terminal then offers `1` edit and run again, `2` drop those observations and run again, `3` leave them.
    The second pass merges the problem groups plus every group your edit made or changed; groups the edit removed
    (or renamed, e.g. `s9052` that gained a member becomes `g02`) lose their image folders and table rows; the
    others stay as they were. The window shows the problem and lets you edit, tick `overwrite` and start again.

### Window (first version)
`chanmerge-gui` (or `python -m chanmerge.gui`) opens a small local window with five tabs (Target, Categories, Flux,
Groups, Run). It fills the same plan and runs the same checks and processing as the terminal; the row-set and group
edit fields take the same syntax. Not packaged yet.

### Time window
Observations inside each row set are grouped greedily by start time with the window you give (days).
`0` means no grouping, with a warning: for DSH work, merging different epochs mixes different halo states.

### Grating arms (asked per grating row, no default)
1. `merge_obs` filter only, no extra mask or fill (arm core and a dark corridor stay).
   `merge_obs` itself always keeps only `tg_m` 0 or Null events for ACIS grating data (CIAO's fluximage does this).
   For your own grating work use the untouched, unfiltered events (all orders): `repro_evt_file(_abs)` in every table,
   i.e. `observations/<obsid>_g/repro/acisf<obsid>_repro_evt2.fits`; `chandra_repro`'s spectra are beside it
   (`*_repro_pha2.fits`, responses in `repro/tg/`).
For a grating subarray row, options 2-4 show an extra warning: the strip runs along the arms, so a large share of it
is masked (more the fewer rows) and option 4 may leave much of it as no data.
A pixel on the edge of the zeroth-order circle counts as source when its centre is inside the circle (CIAO's rule).
2. Mask the arm boxes, keep the source (no data there, about 17% of a full frame).
3. Mask the arm boxes and the source.
4. Mask and fill the boxes from the same radius (rate from clean pixels of the same annulus; the innermost ~25-30"
   cannot be filled and stay no data). Filled pixels are marked in the tables and the mask file.
   Every 7.9" ring of a fill gets its expected total (rate x exposure, rounded at random so the mean is exact), spread
   over its pixels at random by rate x exposure: single pixels look like real counts, but a ring cannot come out much
   brighter or darker than expected by chance. Each ring has its own random stream, so a run is reproducible.

How wide the treated band is, per observation (measured, not a fixed factor):
* The band is CIAO's own arm box (`*_tgmask.fits`). Inside it `merge_obs`'s filter removes most events, background
  included, and the dark corridor ends exactly at the box edge (V404 Cyg HETG, PKS 2155-304 LETG).
* Arm light that leaks outside the box is measured on each observation's own events, per arm (HEG, MEG, or the LETG
  box), and when it is clearly there (Cen X-3: about 2x the background at the box edge, falling over about 60-75 sky px)
  the fitted leak is subtracted beside the box: counts are thinned at random, the pixels stay real data, the mask file
  marks them with 3. When no leak is seen or it cannot be checked (subarray strip, few counts) only the box is treated
  and the report says so.
* No arm visible at all (faint source): nothing is done for the arms, the report says so.
* ACIS + LETG: CIAO draws the LETG box ~2900 sky px wide on purpose (room for spectral background), and inside it the
  filter removes most events, so CIAO's own LETG image is darkened (50-100% of the events per chip). chanmerge
  order-sorts the LETG events again with a narrow box chosen per observation (100-400 sky px: the narrowest box with
  no arm light beside it; an excess that does not shrink when the box grows is subtracted instead) and uses that copy
  for the images only (`<target>/letg_imaging/..._letg_imaging_w<width>_evt.fits`, column `letg_imaging_evt_file`;
  never use it for spectra, the repro files are untouched).
* An observation that must not be mixed with others leaves its group and gets an image of its own (filter only, its
  `left_out_of` column names the group): (a) its box covers the whole image (e.g. a LETG subarray strip, or when the
  narrow re-sorting failed), with any arm option;
  (b) a full frame whose box covers more than 40% of the image, with options 2-4. Automatic groups are merged again
  without it; a manual group keeps it untreated and the terminal asks at the end whether to merge the group again
  without it.

### Readout streak (per row set: `acisreadcorr`, `fill`, `mask`, or nothing)
Detected only where a streak exists. `fill` and `mask` act on a +-2.5" strip along the streak; `acisreadcorr`
runs before `merge_obs` with each observation's own source position: the zeroth order (gratings), else the centroid
of the psf wings on the streak line, else the observation's target if it lies on the line (the search coordinate is
never used); a dark strip may remain near the source. acisreadcorr runs once per observation and writes one corrected
event file (all chips) to `<target>/acisreadcorr/<target>_<obsid>_acisreadcorr_evt.fits`, shared by every band, group
and chip set; the tables name it in `acisreadcorr_evt_file(_abs)`. Every chip with events is searched; the final
report lists, per band and observation, the chips searched and the chips with a streak.

Ring rates for every fill come from pixels outside the arm boxes (+30") and the streak strip. Fills come first, then
masks; where both touch a pixel, the mask wins.

---

## Folder layout
```
<root>/                                   the folder chanmerge runs from
├── chanmerge_index_final.csv             final images of every run here
├── chanmerge_index_members.csv           member images of every run here
├── observations/                         each observation once, reused by later runs
│   ├── 101/repro/                        a done-marker is written after chandra_repro succeeds
│   └── 301_g/repro/                      _g = grating
└── CenA/                                 output folder name
    ├── acisreadcorr/                     acisreadcorr's corrected event file of each observation (if used)
    └── merged_final_image_broad_w3d/     one folder per band and window
        ├── final_images.csv
        ├── member_images.csv             only members of groups with 2+ observations
        ├── g01_2006-03-14_2006-03-16/    2+ observations
        ├── g02_S3_<start>_<end>/         one folder per chip set when a group has several
        ├── s103_2006-04-02/              one observation
        └── all/                          window 0
```
File prefix `<target>_w<W>d_<group>`. merge_obs files keep their names; corrected versions get `_filled`,
`_masked` or `_filled_masked`. The original event files are never changed: every correction writes a new one
(`..._evt_filled.fits`, `..._merged_evt_filled.fits`; added events carry `FILLED=1`, use `[FILLED=0]` for spectra
and timing).

## Tables
`final_images.csv` (one row per image) and `member_images.csv` (one row per member) in every run folder; the two
indexes at the root collect every run. Paths are relative to the run folder (indexes: to the root) with `_abs`
twins in the indexes. Key columns:

* `flux_image`, `expmap_file`, `evt_file`: the version to use (after every correction);
  `*_original`: the raw merge_obs output; `repro_evt_file(_abs)`: the untouched file in `observations/`;
  `acisreadcorr_evt_file(_abs)`: the corrected full event file merge_obs read when acisreadcorr was applied (final
  tables: one entry per obsid, `;`-joined, empty where it was not applied).
* `streak_action`, `streak_obsids`, `arm_method`, `arm_masked_obsids`, `arm_filled_obsids`, `filled_pixel_fraction`,
  `mask_file` (0 untouched, 1 no data, 2 filled, 3 arm leak subtracted).
* `shares_photons_with`: other images holding the same photons (same obsid, overlapping chips and energy band).
* `left_out_of` (final tables): the group this single's observation left because it must not be mixed with others
  (see Grating arms); empty normally.

### Using the index from Python
```python
from chanmerge import load_index, rebuild_index

rows = load_index("chanmerge_index_final.csv")       # absolute paths recomputed from where the index is now
use = [r for r in rows if r["exists"] and not r["shares_photons_with"]]
```
`rebuild_index(root)` rewrites both indexes from all run folders; it runs after every run and can be run by hand
after moving or deleting run folders.

---

## Tested on

Chanmerge 2.0.0 was run on real Chandra observations (most of them downloaded and reprocessed by chanmerge itself; CIAO 4.18, macOS). This is a record of what was checked, not a guarantee for other data, other CIAO versions or other systems.

- **Unit tests:** 293 in `tests/`; they need neither CIAO nor network (`python -m pytest -q tests`).
- **Real-CIAO checks on local data:** `CHANMERGE_CIAO_TESTS=1 python -m pytest -q tests/test_ciao_local.py`.

| Case | Observations | What was checked |
|---|---|---|
| ACIS, no grating, readout streak | 950 (GX 13+1, ACIS-I), 9052 (IGR J18134-1636, ACIS-I), 6688 (ACIS-S), 17704 (V404 Cyg) | Chip sets, grouping, the streak search and its treatments (`acisreadcorr`, fill, mask). A streak was found only in 950; after `acisreadcorr` the excess of the streak strip over the same-radius expectation fell from 30x to 5.6x at 300-600 arcsec. The corrected merged event list equals `merge_obs`'s list minus the original events plus the corrected ones, exactly. |
| HETG, full frame | 27522 (Cen X-3), 17701, 16703, 16704 (V404 Cyg) | The four grating-arm options, the arm fill, and arm light that leaks outside the CIAO arm box (measured on Cen X-3 and subtracted: edge-zone ratio 1.4-1.6 before, 1.0-1.1 after). Filled V404 images: arm-box / outside-ring ratios 0.94-1.06. A real streak on 27522 together with the arm treatment. |
| HETG, subarray | 7490 (128 rows), 16702 (V404 Cyg, 512 rows) | The warning shown at the arm question. 16702: the arm boxes cover about a quarter of the exposed pixels within 240 arcsec of the source, and option 4 leaves 0.3% of the strip without data. 7490: 69% without data, as the warning says. |
| Full frame and subarray in one group | 16702 + 16703 + 16704 (V404 Cyg) | Arm options 2-4, a different option per row, unknown modes (Ocat unreachable). The rate of filled pixels over untouched pixels stays within about 1 sigma of 1 at 30-480 arcsec. |
| LETG | PKS 2155-304 (1790, 1796, 1797, 2324), Capella (55), NGC 4151 (3089), BD+30 3639 (8498), SN 1987A (9591), Mrk 421 (23307, 23309; 128 rows), XTE J0929-314 (3661), Spica (4509) | LETG arm band width chosen per observation (CIAO's default box can cover the whole image and darken it), a visibility check (no visible arm: nothing is masked), leak subtraction, observations whose band covers most of the image leave their group, two LETG or LETG + HETG in one group, and grating exposure maps (grating / non-grating effective area matches the CALDB factors). |
| HRC | 8225 + 8226 (Cyg X-2, HRC-I), 14323 + 14324 (Mrk 421, HRC-S with LETG) | Merging two observations: counts and exposure maps equal the sum of the members, exposure is the dead-time corrected live time, the source sits at the right position. |
| Continuous clocking (CC) | 13197 (GX 13+1, HETG), 1225 (Capella, LETG), 22110 + 22111 (Sco X-1, HETG), 6136 + 6137 (M87, ACIS-I) | CC found from the headers, no flux image for CC, and the question whether to delete data downloaded in this run (answered yes and no). |
| Archive and headers | GX 13+1 search (3 arcmin: 27 rows, 16 kept) | HEASARC and Ocat search, archive sizes, mode and window read from the headers (the header wins over Ocat), Ocat unreachable (simulated, and once with the connection to Ocat refused by a proxy). |
| Paths | 8225 (HRC-I) | Characters in the root path and the folder name, tried with `chandra_repro`, `merge_obs`, `tgdetect2` and `tg_create_mask`; the lists of refused and warned characters come from these runs. |
| Window | the V404 Cyg set, a GX 13+1 set | The Tk window was driven by script against real runs and produced the same files as the terminal flow. |

**Not tested:** a group with several readout streaks on real data, the ACIS S0 and S5 chips as the source chip, other CIAO versions and operating systems, and long manual sessions in the window and the viewer (a first version).

**How to read the results:** the arm and streak fills are statistical (Poisson draws at the rate measured in the surrounding clean pixels), not measured photons. Every filled pixel is flagged in the mask and in the corrected event files, so an analysis can leave them out.

## Authors
**Ahmet Sercan Kıyak** ([@cloaker00-f](https://github.com/cloaker00-f)) **& Berra Dülgerbaki** ([@berradbb](https://github.com/berradbb))
*Boğaziçi University, Department of Physics*

## License
This project is licensed under the **MIT License**.
