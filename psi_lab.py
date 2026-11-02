"""Precision Optical Interferometry Lab - Part 2
Direct phase calculation (Hariharan 5-step) and phase unwrapping.

Input : frame1.npy ... frame5.npy, intensity frames at
        delta = -pi, -pi/2, 0, +pi/2, +pi (same folder as this script)
Output: wrapped phase, unwrapped surface map (nm), PV and RMS
Needs : numpy, matplotlib  (scikit-image optional: python -m pip install scikit-image)
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.restoration import unwrap_phase as _sk_unwrap

    def unwrap(phi, mask):
        return _sk_unwrap(np.ma.masked_array(phi, ~mask)).filled(np.nan)

except ImportError:
    print("scikit-image not found: using simple numpy unwrapping (fine for nulled fringes)")

    def unwrap(phi, mask):
        """Unwrap the centre column, then each row outward from it."""
        out = np.full(phi.shape, np.nan)
        rows, cols = np.nonzero(mask)
        c0 = int(np.median(cols))
        col_rows = np.nonzero(mask[:, c0])[0]
        out[col_rows, c0] = np.unwrap(phi[col_rows, c0])
        for r in col_rows:
            idx = np.nonzero(mask[r])[0]
            line = np.unwrap(phi[r, idx])
            k = np.searchsorted(idx, c0)
            out[r, idx] = line - line[k] + out[r, c0]
        return out

WAVELENGTH_NM = 632.8          # HeNe
MOD_THRESHOLD = 0.1            # fraction of max modulation kept as valid data

# 1. Load the five phase-shifted frames
I1, I2, I3, I4, I5 = [np.load(f"frame{k}.npy").astype(float) for k in range(1, 6)]
frames = [I1, I2, I3, I4, I5]
deltas = ["-π", "-π/2", "0", "+π/2", "+π"]

# 1b. Plot the raw frames, plus the intensity of one pixel versus phase step
fig0, ax0 = plt.subplots(1, 6, figsize=(20, 3.6))
vmax = max(f.max() for f in frames)
for k, (f, d) in enumerate(zip(frames, deltas)):
    ax0[k].imshow(f, cmap="gray", vmin=0, vmax=vmax)
    ax0[k].set_title(f"frame{k + 1}.npy\nδ = {d}")
    ax0[k].set_xlabel("x (camera pixel)")
ax0[0].set_ylabel("y (camera pixel)")
py, px = I3.shape[0] // 2, I3.shape[1] // 2            # pixel to follow (image centre)
for a in ax0[:5]:
    a.plot(px, py, "r+", ms=12, mew=2)
step = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
vals = np.array([f[py, px] for f in frames])
d_fine = np.linspace(-np.pi, np.pi, 200)
n_, d_ = 2 * (vals[1] - vals[3]), 2 * vals[2] - vals[0] - vals[4]
phi_px = np.arctan2(n_, d_)
A_px = (vals[0] + vals[4] + 2 * vals[2]) / 4               # background A
B_px = np.hypot(n_, d_) / 4                                # fringe amplitude B
ax0[5].plot(d_fine, A_px + B_px * np.cos(phi_px + d_fine), "k-", lw=1, label="A + B cos(φ + δ)")
ax0[5].plot(step, vals, "ro", label="5 measured values")
ax0[5].set_xticks(step, deltas)
ax0[5].set_xlabel("phase step δ"); ax0[5].set_ylabel("intensity (counts)")
ax0[5].set_title(f"Red + pixel: φ = {phi_px:.2f} rad")
ax0[5].legend(fontsize=8, loc="best")
fig0.suptitle("Raw intensity frames (input to the phase calculation)")
fig0.tight_layout()

# 2. Hariharan 5-step algorithm: phi = atan2(2(I2 - I4), 2I3 - I1 - I5)
num = 2 * (I2 - I4)
den = 2 * I3 - I1 - I5
phi_wrapped = np.arctan2(num, den)          # -pi ... +pi

# 3. Modulation (fringe contrast) -> mask of usable pixels
modulation = np.hypot(num, den)
mask = modulation > MOD_THRESHOLD * modulation.max()

# 4. Unwrap: remove the 2*pi jumps
phi_unwrapped = unwrap(phi_wrapped, mask)
mask &= np.isfinite(phi_unwrapped)

# 5. Phase -> surface height (reflection doubles the path: h = lambda * phi / 4pi)
height = WAVELENGTH_NM / (4 * np.pi) * phi_unwrapped

# 6. Remove piston and tilt with a least-squares plane fit
yy, xx = np.nonzero(mask)
G = np.c_[np.ones(len(xx)), xx, yy]
coef, *_ = np.linalg.lstsq(G, height[mask], rcond=None)
height[mask] -= G @ coef

pv = np.nanmax(height) - np.nanmin(height)
rms = np.nanstd(height)
print(f"PV  = {pv:.2f} nm  = {pv / WAVELENGTH_NM:.4f} waves")
print(f"RMS = {rms:.2f} nm  = {rms / WAVELENGTH_NM:.4f} waves   (compare with the Zygo result)")

# 7. Plots: wrapped phase, unwrapped surface, one row before/after unwrapping
row = height.shape[0] // 2
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
im0 = ax[0].imshow(np.where(mask, phi_wrapped, np.nan), cmap="twilight")
ax[0].set_title("Wrapped phase (rad)"); fig.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(height, cmap="jet")
ax[1].set_title(f"Surface (nm), RMS {rms:.2f} nm"); fig.colorbar(im1, ax=ax[1])
ax[2].plot(np.where(mask, phi_wrapped, np.nan)[row], label="wrapped")
ax[2].plot(phi_unwrapped[row] - np.nanmean(phi_unwrapped[row]), label="unwrapped")
ax[2].set_title(f"Row {row}"); ax[2].set_xlabel("pixel"); ax[2].set_ylabel("phase (rad)")
ax[2].legend()
for a in ax[:2]:
    a.set_xlabel("x (camera pixel)"); a.set_ylabel("y (camera pixel)")
plt.tight_layout()
plt.show()
