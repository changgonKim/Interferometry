"""Make frame1.npy ... frame5.npy for psi_lab.py.

Two ways to use it (run from the same folder as psi_lab.py):

  python make_frames.py
      -> synthetic flat mirror shaped like our Zygo result
         (PV/RMS set by TARGET_PV_WAVES / TARGET_RMS_WAVES below)

  python make_frames.py measurement.datx
      -> frames built from a real Zygo Mx measurement (needs: pip install h5py)

  python make_frames.py surface.csv --unit nm
      -> frames built from a surface map exported as CSV/TXT (unit: nm, um, waves)

Frames: I_k = A + B cos(phi + delta_k), delta_k = -pi, -pi/2, 0, +pi/2, +pi
with phi = 4*pi*h/lambda (reflection), plus a little camera noise.
"""
import sys
import numpy as np

WAVELENGTH_NM = 632.8
TILT_FRINGES = 1.5      # fringes of tilt added so students can see 2*pi wrapping
NOISE = 0.002           # camera noise, fraction of peak intensity
rng = np.random.default_rng(0)

UNIT_TO_NM = {"meters": 1e9, "m": 1e9, "millimeters": 1e6, "mm": 1e6,
              "micrometers": 1e3, "microns": 1e3, "um": 1e3,
              "nanometers": 1.0, "nm": 1.0, "waves": WAVELENGTH_NM, "wave": WAVELENGTH_NM}


# ---- target for the synthetic mirror (values read off the Zygo Mx result) ----
TARGET_PV_WAVES = 0.063     # Zygo colour bar: -0.031 ... +0.032 waves
TARGET_RMS_WAVES = None     # put the Zygo RMS here (in waves), e.g. 0.011; None = keep shape's natural RMS


def _smooth_noise(shape, sigma_px):
    """Random texture with a given correlation length (FFT low-pass)."""
    ny, nx = shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    filt = np.exp(-2 * (np.pi * sigma_px) ** 2 * (fx**2 + fy**2))
    t = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * filt))
    return t / t.std()


def synthetic(n=480):
    """Flat mirror that looks like the Zygo result: bowl (centre low, edge high),
    a little astigmatism, edge roll-up, fine texture and a few dust pits."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r2 = x**2 + y**2
    th = np.arctan2(y, x)
    mask = r2 <= 0.95**2
    rn2 = r2 / 0.95**2                                  # normalised radius^2 inside the pupil

    smooth = (0.90 * (2 * rn2 - 1)                       # power: centre low, edge high
              + 0.18 * rn2 * np.cos(2 * (th - 2.4))     # astigmatism (left side higher)
              + 0.20 * (6 * rn2**2 - 6 * rn2 + 1))      # spherical: edge roll-up
    texture = 0.6 * _smooth_noise(r2.shape, 6) + 0.4 * _smooth_noise(r2.shape, 2)
    pits = np.zeros_like(r2)
    for _ in range(8):                                   # dust / pits = small low spots
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        pits -= np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.012**2))

    def shape(w):
        s = smooth / smooth[mask].std()
        return s + w * (0.5 * texture + 1.5 * pits)

    # choose texture weight w so PV/RMS matches the target ratio, then scale to the target PV
    def pv_rms(h):
        v = h[mask]
        return np.ptp(v), v.std()

    w = 0.18
    if TARGET_RMS_WAVES:
        target_ratio = TARGET_PV_WAVES / TARGET_RMS_WAVES
        ws = np.linspace(0, 3, 601)
        ratios = np.array([np.divide(*pv_rms(shape(wi))) for wi in ws])
        w = ws[np.argmin(np.abs(ratios - target_ratio))]
        if abs(ratios.min() - target_ratio) > 0 and target_ratio < ratios.min():
            print(f"Note: PV/RMS = {target_ratio:.2f} is lower than this shape allows "
                  f"({ratios.min():.2f}); RMS will be slightly low.")
    h = shape(w)
    pv, _ = pv_rms(h)
    h = h / pv * TARGET_PV_WAVES * WAVELENGTH_NM        # in nm

    # illumination: fairly uniform disc, gentle gradient, a few dark dust shadows
    intensity = 0.9 + 0.1 * x - 0.05 * r2
    for _ in range(4):
        cx, cy = rng.uniform(-0.6, 0.6, 2)
        intensity *= 1 - 0.35 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.03**2))
    return h, intensity, mask


def _decode(v):
    if isinstance(v, bytes):
        return v.decode(errors="ignore")
    if isinstance(v, np.ndarray) and v.size == 1:
        return _decode(v.item())
    return v


def from_datx(path, unit_override=None):
    import h5py
    surf = inten = None
    with h5py.File(path, "r") as f:
        found = []
        f.visititems(lambda name, obj: found.append(name) if isinstance(obj, h5py.Dataset) else None)
        for name in found:
            leaf = name.split("/")[-1].lower()
            if leaf == "surface" and surf is None:
                surf = name
            if leaf == "intensity" and inten is None:
                inten = name
        if surf is None:
            print("Datasets in file:\n  " + "\n  ".join(found))
            sys.exit("Could not find a 'Surface' dataset - send this list to update the script.")
        ds = f[surf]
        h = np.array(ds, dtype=float).squeeze()
        attrs = {k: _decode(v) for k, v in ds.attrs.items()}
        nodata = attrs.get("No Data")
        if nodata is not None:
            try:
                h[h == float(np.asarray(nodata).ravel()[0])] = np.nan
            except (TypeError, ValueError):
                pass
        h[np.abs(h) > 1e30] = np.nan
        unit = unit_override or str(attrs.get("Unit", "")).strip().lower()
        if unit not in UNIT_TO_NM:
            print(f"Surface attributes: {attrs}")
            sys.exit(f"Unknown unit '{unit}'. Re-run with --unit nm|um|waves|m")
        h *= UNIT_TO_NM[unit]
        print(f"Read '{surf}' ({h.shape}), unit '{unit}'")
        intensity = np.array(f[inten], dtype=float).squeeze() if inten else None
    return h, intensity


def main():
    args = sys.argv[1:]
    unit = None
    if "--unit" in args:
        i = args.index("--unit")
        unit = args[i + 1].lower()
        del args[i:i + 2]

    if not args:
        h, intensity, mask = synthetic()
        print("Synthetic flat mirror")
    else:
        path = args[0]
        if path.lower().endswith(".datx"):
            h, intensity = from_datx(path, unit)
        else:
            h = np.genfromtxt(path, delimiter="," if path.lower().endswith(".csv") else None)
            h *= UNIT_TO_NM[unit or "nm"]
            intensity = None
        mask = np.isfinite(h)

    h = np.where(mask, h, 0.0)
    if intensity is None or intensity.shape != h.shape or not np.isfinite(intensity[mask]).any():
        intensity = np.ones_like(h)
    intensity = np.where(mask & np.isfinite(intensity), intensity, 0.0)
    intensity = intensity / intensity[mask].max()

    # report the true surface with piston + tilt removed (should match psi_lab.py)
    yy, xx = np.nonzero(mask)
    G = np.c_[np.ones(len(xx)), xx, yy]
    resid = h[mask] - G @ np.linalg.lstsq(G, h[mask], rcond=None)[0]
    print(f"True surface (piston/tilt removed): PV {np.ptp(resid):.2f} nm ({np.ptp(resid)/WAVELENGTH_NM:.4f} waves), "
          f"RMS {resid.std():.2f} nm ({resid.std()/WAVELENGTH_NM:.4f} waves)")

    # add tilt fringes across the aperture, then convert to phase
    ny, nx = h.shape
    tilt_nm = TILT_FRINGES * (WAVELENGTH_NM / 2) * (np.arange(nx) / nx)[None, :]
    phi = 4 * np.pi * (h + tilt_nm) / WAVELENGTH_NM

    A = 0.5 * intensity
    B = 0.4 * intensity               # fringe visibility 0.8
    for k, d in enumerate([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], start=1):
        I = A + B * np.cos(phi + d) + NOISE * rng.standard_normal(h.shape)
        I = np.where(mask, np.clip(I, 0, None), 0.0)
        np.save(f"frame{k}.npy", (I * 4095).astype(np.uint16))   # 12-bit camera
    print("Saved frame1.npy ... frame5.npy")


if __name__ == "__main__":
    main()
