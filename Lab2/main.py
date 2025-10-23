import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from functools import lru_cache

# ---------------- I/O ----------------
def load_image(image_path):
    img = Image.open(image_path).convert('L')
    return np.asarray(img, dtype=np.float64)

def visualize_and_save_dct_coefficients(coefficients, output_path):
    V = np.log1p(np.abs(coefficients))
    plt.figure(figsize=(12, 10))
    plt.imshow(V, cmap='gray')
    plt.colorbar(label='Log magnitude')
    plt.title("DCT Coefficients (Log Domain)")
    plt.tight_layout()
    plt.savefig(output_path); plt.close()

def save_image(image, path):
    Image.fromarray(np.uint8(np.clip(image, 0, 255))).save(path)

# ------------- Orthonormal DCT basis -------------
@lru_cache(maxsize=None)
def _dct_basis(n: int) -> np.ndarray:
    n_idx = np.arange(n)[None, :]
    k_idx = np.arange(n)[:, None]
    C = np.sqrt(2.0 / n) * np.cos((2 * n_idx + 1) * k_idx * np.pi / (2 * n))
    C[0, :] *= 1 / np.sqrt(2.0)
    return C  # orthonormal: C @ C.T = I

# ---------------- 2D DCT / IDCT ----------------
def dct_2d(image):
    X = np.asarray(image, dtype=np.float64)
    M, N = X.shape
    Cx, Cy = _dct_basis(M), _dct_basis(N)
    return Cx @ X @ Cy.T

def idct_2d(coefficients):
    D = np.asarray(coefficients, dtype=np.float64)
    M, N = D.shape
    Cx, Cy = _dct_basis(M), _dct_basis(N)
    return Cx.T @ D @ Cy

# ---------------- 1D DCT / IDCT ----------------
def dct_1d(signal):
    x = np.asarray(signal, dtype=np.float64)
    C = _dct_basis(x.shape[0])
    return C @ x

def idct_1d(coefficients):
    a = np.asarray(coefficients, dtype=np.float64)
    C = _dct_basis(a.shape[0])
    return C.T @ a

# -------- 2D via two 1D (row pass + col pass) --------
def dct_2d_using_1d(image):
    X = np.asarray(image, dtype=np.float64)
    M, N = X.shape
    Cx, Cy = _dct_basis(M), _dct_basis(N)
    # row-wise 1D: Cx @ X，再 col-wise 1D: (·) @ Cy.T
    return (Cx @ X) @ Cy.T

def idct_2d_using_1d(coefficients):
    D = np.asarray(coefficients, dtype=np.float64)
    M, N = D.shape
    Cx, Cy = _dct_basis(M), _dct_basis(N)
    # row-wise inverse: Cx.T @ D，再 col-wise inverse: (·) @ Cy
    return (Cx.T @ D) @ Cy

# ---------------- Metrics / runner ----------------
def psnr(original, reconstructed):
    x = np.asarray(original, dtype=np.float64)
    y = np.asarray(reconstructed, dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def process_image(image, dct_function, idct_function, method_name):
    start_time = time.time()
    coeff = dct_function(image)
    recon = idct_function(coeff)
    processing_time = time.time() - start_time
    psnr_value = psnr(image, recon)
    os.makedirs('output', exist_ok=True)
    visualize_and_save_dct_coefficients(coeff, os.path.join('output', f"dct_coefficients_{method_name}.png"))
    save_image(recon, os.path.join('output', f"reconstructed_{method_name}.png"))
    return psnr_value, processing_time

def print_results(method_name, psnr_value, processing_time):
    print(f"Results for {method_name}:")
    print(f"  PSNR: {psnr_value:.2f} dB")
    print(f"  Processing time: {processing_time:.4f} seconds")
    print()

def main():
    if not os.path.exists('output'):
        os.makedirs('output')

    image = load_image('lena.png')

    psnr_2d, time_2d = process_image(image, dct_2d, idct_2d, "2d")
    print_results("2D-DCT", psnr_2d, time_2d)

    psnr_1d, time_1d = process_image(image, dct_2d_using_1d, idct_2d_using_1d, "1d")
    print_results("Two 1D-DCT", psnr_1d, time_1d)

if __name__ == "__main__":
    main()
