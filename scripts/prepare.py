import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import torch
from configs import load_config


def load_meerkat_patch(
    fits_path: Path,
    patch_height: int | None,
    patch_width: int | None,
    x0: int | None,
    y0: int | None,
) -> tuple[np.ndarray, fits.Header]:
    with fits.open(fits_path) as hdul:
        element = hdul[0]
        full_data = element.data.astype(  # pyright: ignore[reportAttributeAccessIssue]
            np.float32
        )
        header = element.header  # pyright: ignore[reportAttributeAccessIssue]

    x0 = x0 if x0 is not None else 0
    y0 = y0 if y0 is not None else 0
    height = patch_height if patch_height is not None else (full_data.shape[0] - x0 + 1)
    width = patch_width if patch_width is not None else (full_data.shape[1] - y0 + 1)

    data = full_data[x0 : x0 + height, y0 : y0 + width]
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("All selected pixels are NaN or inf.")

    data = data - np.nanmin(data)
    data += 1e-5
    data = np.log(data)
    data = data - np.nanmin(data)
    data = data / np.nanmax(data)
    return data, header


# def get_ra_dec(
#     header: fits.Header, shape: tuple[int, int], x0: int, y0: int
# ) -> np.ndarray:
#     wcs = WCS(header, naxis=2)
#     ny, nx = shape
#     yy, xx = np.mgrid[0:ny, 0:nx]
#     ra, dec = wcs.pixel_to_world_values(xx + x0, yy + y0)

#     ra_rad = np.deg2rad(np.mod(ra, 360.0))
#     dec_rad = np.deg2rad(dec)

#     return np.stack([np.sin(ra_rad), np.cos(ra_rad), dec_rad], axis=-1).reshape(-1, 3)


def get_coord_grid(height: int, width: int) -> np.ndarray:
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = (np.linspace(-1, 1, num=height), np.linspace(-1, 1, num=width))
    mgrid = np.stack(np.meshgrid(*tensors), axis=-1)
    mgrid = mgrid.reshape(-1, 2)
    return mgrid


# def get_beam_angle(header: fits.Header) -> float:
#     bmin = np.deg2rad(header["BMIN"])
#     bmax = np.deg2rad(header["BMAJ"])
#     return np.sqrt(bmin * bmax)


# def choose_omega(header: fits.Header) -> float:
#     beam_angle = get_beam_angle(header)
#     return 1 / beam_angle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SIREN dataset from MeerKAT FITS"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    fits_path = Path(cfg.paths.fits)
    patch_height = cfg.prepare.patch_height
    patch_width = cfg.prepare.patch_width
    x0 = cfg.prepare.x0
    y0 = cfg.prepare.y0
    out_path = Path(cfg.prepare.out or cfg.paths.data)
    dtype_name = cfg.prepare.dtype

    data, header = load_meerkat_patch(fits_path, patch_height, patch_width, x0, y0)

    height, width = data.shape
    yy, xx = np.mgrid[0:height, 0:width]
    rows_flat = yy.reshape(-1)
    cols_flat = xx.reshape(-1)

    finite_mask = np.isfinite(data)
    mask_flat = finite_mask.reshape(-1)

    x0 = x0 if x0 is not None else 0
    y0 = y0 if y0 is not None else 0
    coords = get_coord_grid(height, width)

    pixels = data.reshape(-1, 1)
    coords = coords[mask_flat]
    pixels = pixels[mask_flat]
    rows_flat = rows_flat[mask_flat]
    cols_flat = cols_flat[mask_flat]

    dtype = np.float16 if dtype_name == "float16" else np.float32
    coords = coords.astype(dtype, copy=False)
    pixels = pixels.astype(dtype, copy=False)

    print("\n=== Sanity check: coordinate ↔ pixel alignment ===")
    idx = np.random.choice(len(coords), size=10, replace=False)

    for i in idx:
        r = rows_flat[i]
        c = cols_flat[i]
        coord = coords[i]
        target = pixels[i, 0]
        original = data[r, c]

        print(
            f"i={i:6d} | row={r:4d} col={c:4d} | "
            f"coord={coord} | "
            f"target={target:.6f} | image={original:.6f}"
        )

    print("\n=== Target stats ===")
    print(
        f"min={pixels.min():.6f}, "
        f"max={pixels.max():.6f}, "
        f"mean={pixels.mean():.6f}, "
        f"std={pixels.std():.6f}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, coords=coords, pixels=pixels, height=height, width=width
    )
    print(f"Saved dataset to {out_path}")

    if cfg.prepare.debug_plots:
        import matplotlib.pyplot as plt
        import scienceplots

        plt.style.use(["ieee", "no-latex"])

        plt.figure(figsize=(6, 6), dpi=150)
        plt.imshow(data, origin="lower")
        plt.colorbar()
        plt.title(
            f"Extracted Patch Image\n$(x_0, y_0) = ({x0, y0}), (h, w) = ({height, width})$"
        )
        plt.tight_layout()
        plt.savefig("debug_plot.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
