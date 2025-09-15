import argparse
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def _read_rgb(img_path: Path) -> np.ndarray:
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32)  # H x W x 3
    except FileNotFoundError:
        raise FileNotFoundError(f"Input not found: {img_path}")
    except UnidentifiedImageError:
        raise ValueError(f"Cannot decode image: {img_path}")
    except OSError as e:
        raise OSError(f"I/O error reading {img_path}: {e}")

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expect HxWx3, got {arr.shape}")

    return arr


def _save_gray(arr: np.ndarray, out_path: Path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def rgb_to_yuv(R: np.ndarray, G: np.ndarray, B: np.ndarray):
    """
    RGB -> YUV using the given coefficients (+128 offsets):
      Y = 0.299 R + 0.587 G + 0.114 B
      U = -0.169 R - 0.331 G + 0.5 B + 128
      V =  0.5 R - 0.419 G - 0.081 B + 128
    """
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.169 * R - 0.331 * G + 0.5 * B + 128.0
    V = 0.5 * R - 0.419 * G - 0.081 * B + 128.0
    return Y, U, V


def rgb_to_ycbcr(R: np.ndarray, G: np.ndarray, B: np.ndarray):
    """
    ITU-R BT.601 full-range (common slides/textbook form):
      Y  = 0.299000 R + 0.587000 G + 0.114000 B
      Cb = -0.168736 R - 0.331264 G + 0.5 B + 128
      Cr =  0.500000 R - 0.418688 G - 0.081312 B + 128
    """
    # Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128.0
    return Cb, Cr


def main():
    ap = argparse.ArgumentParser(description="Dump RGB→YUV/YCbCr channels to 8 grayscale images.")
    ap.add_argument("--input", "-i", type=Path, required=True, help="Input image path, e.g., lena.png")
    ap.add_argument("--outdir", "-o", type=Path, required=True, help="Output directory")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    rgb = _read_rgb(args.input)         # H x W x 3, float32, [0..255]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Save raw R/G/B single-channel grayscale
    _save_gray(R, args.outdir / "R.png")
    _save_gray(G, args.outdir / "G.png")
    _save_gray(B, args.outdir / "B.png")

    # RGB -> YUV 
    Y, U, V = rgb_to_yuv(R, G, B)
    _save_gray(Y, args.outdir / "Y.png")
    _save_gray(U, args.outdir / "U.png")
    _save_gray(V, args.outdir / "V.png")

    # RGB -> YCbCr 
    Cb, Cr = rgb_to_ycbcr(R, G, B)
    _save_gray(Cb, args.outdir / "Cb.png")
    _save_gray(Cr, args.outdir / "Cr.png")

    

    print("Done. Saved to:", args.outdir.resolve())


if __name__ == "__main__":
    main()
