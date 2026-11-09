"""Sky geometry: tangent-plane projection, fov1 chip polygons, image pixel grids. numpy + astropy only."""
import numpy as np

SKY_ARCSEC = 0.492          # arcsec per acis sky pixel
EDGE_ARCSEC = 16.0          # dither size: closer than this to a chip edge -> both chips may hold the source


def radec_to_sky(ra, dec, crval, crpix, cdelt):
    """gnomonic projection with the fits wcs of a pair of x,y columns"""
    ra0, dec0 = np.radians(crval[0]), np.radians(crval[1])
    a, d = np.radians(ra), np.radians(dec)
    cosc = np.sin(dec0) * np.sin(d) + np.cos(dec0) * np.cos(d) * np.cos(a - ra0)
    xi = np.cos(d) * np.sin(a - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(d) - np.sin(dec0) * np.cos(d) * np.cos(a - ra0)) / cosc
    return crpix[0] + np.degrees(xi) / cdelt[0], crpix[1] + np.degrees(eta) / cdelt[1]


def sky_to_radec(x, y, crval, crpix, cdelt):
    xi = np.radians((np.asarray(x) - crpix[0]) * cdelt[0])
    eta = np.radians((np.asarray(y) - crpix[1]) * cdelt[1])
    ra0, dec0 = np.radians(crval[0]), np.radians(crval[1])
    den = np.cos(dec0) - eta * np.sin(dec0)
    ra = ra0 + np.arctan2(xi, den)
    dec = np.arctan2(np.sin(dec0) + eta * np.cos(dec0), np.hypot(xi, den))
    return np.degrees(ra) % 360, np.degrees(dec)


def column_wcs(header, xname="x", yname="y"):
    """(crval, crpix, cdelt) of the x,y columns of an event or region table header"""
    names = {}
    for k, v in header.items():
        if k.startswith("TTYPE") and isinstance(v, str):
            names[v.strip().lower()] = k[5:]
    ix, iy = names[xname], names[yname]
    return ((header[f"TCRVL{ix}"], header[f"TCRVL{iy}"]), (header[f"TCRPX{ix}"], header[f"TCRPX{iy}"]),
            (header[f"TCDLT{ix}"], header[f"TCDLT{iy}"]))


def inside_polygon(px, py, xs, ys):
    """ray casting, vectorized over points"""
    px = np.asarray(px, float); py = np.asarray(py, float)
    c = np.zeros(np.broadcast(px, py).shape, bool)
    n = len(xs); j = n - 1
    for i in range(n):
        dy = ys[j] - ys[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = xs[i] + (py - ys[i]) * (xs[j] - xs[i]) / (dy if dy != 0 else 1e-12)
        c ^= ((ys[i] > py) != (ys[j] > py)) & (px < xc)
        j = i
    return c


def edge_distance(px, py, xs, ys):
    best = np.inf
    for i in range(len(xs) - 1):
        x1, y1, x2, y2 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else min(1.0, max(0.0, ((px - x1) * dx + (py - y1) * dy) / L))
        best = min(best, float(np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))))
    return best


def read_fov(path):
    """fov1 file -> (wcs, [(ccd_id, xs, ys), ...]); polygons are in sky pixels, dither included"""
    from astropy.io import fits
    with fits.open(path) as h:
        tab = h[1]
        wcs = column_wcs(tab.header, "x", "y")
        polys = []
        for row in tab.data:
            xs = np.asarray(row["X"], float); ys = np.asarray(row["Y"], float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            ccd = int(row["CCD_ID"]) if "CCD_ID" in tab.columns.names else -1
            polys.append((ccd, xs[ok], ys[ok]))
    return wcs, polys


def fov_lookup(path, ra, dec, edge_arcsec=EDGE_ARCSEC):
    """chips on, chips under (ra, dec), and whether the point is near a chip edge.
    near an edge (< dither) the chips within edge_arcsec are all returned."""
    wcs, polys = read_fov(path)
    x, y = radec_to_sky(ra, dec, *wcs)
    on = sorted({c for c, _, _ in polys if c >= 0})
    under, near = [], False
    for c, xs, ys in polys:
        d = edge_distance(x, y, xs, ys) * SKY_ARCSEC
        ins = bool(inside_polygon(x, y, xs, ys))
        if ins or d < edge_arcsec:
            if d < edge_arcsec:
                near = True
            if ins or d < edge_arcsec:
                under.append((c, ins, d))
    # chips containing the point first, then neighbours within the dither zone
    under.sort(key=lambda t: (not t[1], t[2]))
    src = tuple(c for c, _, _ in under)
    return tuple(on), src, near


def image_sky_grid(header, shape):
    """sky x,y of every image pixel centre (physical wcs of merge_obs images)"""
    yy, xx = np.indices(shape)
    px = header["CRVAL1P"] + (xx + 1 - header["CRPIX1P"]) * header["CDELT1P"]
    py = header["CRVAL2P"] + (yy + 1 - header["CRPIX2P"]) * header["CDELT2P"]
    return px, py


def sky_to_pixel_index(x, y, header, shape):
    """event sky position -> image pixel index (same binning as merge_obs and dmcopy)"""
    bx = np.floor((np.asarray(x) - header["CRVAL1P"]) / header["CDELT1P"] + header["CRPIX1P"] - 0.5).astype(int)
    by = np.floor((np.asarray(y) - header["CRVAL2P"]) / header["CDELT2P"] + header["CRPIX2P"] - 0.5).astype(int)
    ok = (bx >= 0) & (by >= 0) & (bx < shape[1]) & (by < shape[0])
    return bx, by, ok


def transform_points(x, y, wcs_from, wcs_to):
    """sky pixels of one tangent plane -> another (e.g. observation frame -> merge_obs frame)"""
    ra, dec = sky_to_radec(x, y, *wcs_from)
    return radec_to_sky(ra, dec, *wcs_to)


def transform_angle(x, y, angle_deg, wcs_from, wcs_to, step=100.0):
    """rotate an angle through the frame change by mapping a point 'step' pixels away"""
    t = np.radians(angle_deg)
    x0, y0 = transform_points(x, y, wcs_from, wcs_to)
    x1, y1 = transform_points(x + step * np.cos(t), y + step * np.sin(t), wcs_from, wcs_to)
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
