import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import time


# ======================================
# 8x8 DCT / IDCT
# ======================================
def dct_2d(block_8x8: np.ndarray) -> np.ndarray:
    """2D DCT on an 8x8 block."""
    h, w = block_8x8.shape
    assert h == 8 and w == 8

    # DCT basis coefficient (C(0)=1/sqrt(2), C(k)=1 otherwise)
    c = np.ones(8, dtype=np.float64)
    c[0] = 1.0 / np.sqrt(2.0)

    x = np.arange(8).reshape(-1, 1)
    y = np.arange(8).reshape(1, -1)

    dct_block = np.zeros_like(block_8x8, dtype=np.float64)
    for u in range(8):
        cos_x = np.cos((2 * x + 1) * u * np.pi / 16.0)
        for v in range(8):
            cos_y = np.cos((2 * y + 1) * v * np.pi / 16.0)
            dct_block[u, v] = (2.0 / 8.0) * c[u] * c[v] * np.sum(
                block_8x8 * cos_x * cos_y
            )

    return dct_block


def idct_2d(coeff_8x8: np.ndarray) -> np.ndarray:
    """2D IDCT on an 8x8 block."""
    h, w = coeff_8x8.shape
    assert h == 8 and w == 8

    c = np.ones(8, dtype=np.float64)
    c[0] = 1.0 / np.sqrt(2.0)

    u = np.arange(8).reshape(-1, 1)
    v = np.arange(8).reshape(1, -1)

    out = np.zeros_like(coeff_8x8, dtype=np.float64)
    for x in range(8):
        cos_u = np.cos((2 * x + 1) * u * np.pi / 16.0)
        for y in range(8):
            cos_v = np.cos((2 * y + 1) * v * np.pi / 16.0)
            out[x, y] = (2.0 / 8.0) * np.sum(
                c[u] * c[v] * coeff_8x8 * cos_u * cos_v
            )

    return np.clip(out, 0, 255)


# ======================================
# Quantization
# ======================================
def quantize(block: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    return np.round(block / q_table).astype(np.int32)


def dequantize(q_block: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    return (q_block * q_table).astype(np.float64)


# ======================================
# Zigzag + RLE
# ======================================
# global zigzag order (flattened indices)
ZIGZAG_ORDER = np.array(
    [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ]
).flatten()


def zigzag(block: np.ndarray) -> np.ndarray:
    """Return 1D zigzag-scanned array from 8x8 block."""
    flat = np.zeros(64, dtype=np.int32)
    for k in range(64):
        idx = np.unravel_index(ZIGZAG_ORDER[k], (8, 8))
        flat[k] = block[idx]
    return flat


def inv_zigzag(flat: np.ndarray) -> np.ndarray:
    """Restore 8x8 block from 1D zigzag array."""
    block = np.zeros((8, 8), dtype=np.int32)
    for k in range(64):
        idx = np.unravel_index(ZIGZAG_ORDER[k], (8, 8))
        block[idx] = flat[k]
    return block


def rle_encode(block: np.ndarray):
    """RLE after zigzag (block: 8x8)."""
    seq = zigzag(block)
    rle = []
    zeros = 0

    for val in seq:
        if val == 0:
            zeros += 1
        else:
            if zeros > 0:
                rle.append((0, zeros))
                zeros = 0
            rle.append((val, 1))

    if zeros > 0:
        rle.append((0, zeros))

    return rle


def rle_decode(rle_list):
    """Inverse RLE to 8x8 block."""
    flat = np.zeros(64, dtype=np.int32)
    pos = 0
    for val, cnt in rle_list:
        for _ in range(cnt):
            if pos < 64:
                flat[pos] = val
                pos += 1
    return inv_zigzag(flat)


# ======================================
# PSNR
# ======================================
def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    diff = img_a.astype(np.float32) - img_b.astype(np.float32)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


# ======================================
# Whole image pipeline
# ======================================
def encode_decode_image(image: np.ndarray, q_table: np.ndarray):
    """Full pipeline: block DCT->quant->RLE, then inverse."""
    h, w = image.shape
    assert h % 8 == 0 and w % 8 == 0

    start = time.time()
    bitstream = []
    recon = np.zeros_like(image, dtype=np.uint8)

    # encode all blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blk = image[i : i + 8, j : j + 8].astype(np.float64)
            c_block = dct_2d(blk)
            q_block = quantize(c_block, q_table)
            bitstream.append(rle_encode(q_block))

    # decode all blocks
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            q_block_rec = rle_decode(bitstream[idx])
            c_block_rec = dequantize(q_block_rec, q_table)
            blk_rec = idct_2d(c_block_rec)
            recon[i : i + 8, j : j + 8] = blk_rec
            idx += 1

    elapsed = time.time() - start
    return recon, bitstream, elapsed, psnr(image, recon)


# ======================================
# Main
# ======================================
if __name__ == "__main__":
    print("==== DCT-based Image Compression Demo ====")

    # output dirs
    ENC_DIR = "encoded_image"
    DEC_DIR = "decoded_image"
    os.makedirs(ENC_DIR, exist_ok=True)
    os.makedirs(DEC_DIR, exist_ok=True)

    # load lena (grayscale)
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Cannot load 'lena.png'")

    # quant tables from homework
    Q1 = np.array(
        [
            [10, 7, 6, 10, 14, 24, 31, 37],
            [7, 7, 8, 11, 16, 35, 36, 33],
            [8, 8, 10, 14, 24, 34, 41, 34],
            [8, 10, 13, 17, 31, 52, 48, 37],
            [11, 13, 22, 34, 41, 65, 62, 46],
            [14, 21, 33, 38, 49, 62, 68, 55],
            [29, 38, 47, 52, 62, 73, 72, 61],
            [43, 55, 57, 59, 67, 60, 62, 59],
        ]
    )

    Q2 = np.array(
        [
            [10, 11, 14, 28, 59, 59, 59, 59],
            [11, 13, 16, 40, 59, 59, 59, 59],
            [14, 16, 34, 59, 59, 59, 59, 59],
            [28, 40, 59, 59, 59, 59, 59, 59],
            [59, 59, 59, 59, 59, 59, 59, 59],
            [59, 59, 59, 59, 59, 59, 59, 59],
            [59, 59, 59, 59, 59, 59, 59, 59],
            [59, 59, 59, 59, 59, 59, 59, 59],
        ]
    )

    print("> Quantization table 1...")
    rec1, bs1, t1, psnr1 = encode_decode_image(img, Q1)
    with open(os.path.join(ENC_DIR, "qtable1.pkl"), "wb") as f:
        pickle.dump(bs1, f)
    size1 = os.path.getsize(os.path.join(ENC_DIR, "qtable1.pkl"))

    print("> Quantization table 2...")
    rec2, bs2, t2, psnr2 = encode_decode_image(img, Q2)
    with open(os.path.join(ENC_DIR, "qtable2.pkl"), "wb") as f:
        pickle.dump(bs2, f)
    size2 = os.path.getsize(os.path.join(ENC_DIR, "qtable2.pkl"))

    print(f"[Q1] size={size1} bytes, time={t1:.3f}s, PSNR={psnr1:.2f} dB")
    print(f"[Q2] size={size2} bytes, time={t2:.3f}s, PSNR={psnr2:.2f} dB")

    # save images
    cv2.imwrite(os.path.join(DEC_DIR, "original.png"), img)
    cv2.imwrite(os.path.join(DEC_DIR, "recon_q1.png"), rec1)
    cv2.imwrite(os.path.join(DEC_DIR, "recon_q2.png"), rec2)

    # show comparison
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Q1 reconstructed")
    plt.axis("off")
    plt.imshow(rec1, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Q2 reconstructed")
    plt.axis("off")
    plt.imshow(rec2, cmap="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(DEC_DIR, "comparison.png"))
    plt.show()

    print("==== DONE ====")
