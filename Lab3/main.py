import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

block_size = 8

# Output root
output_path = os.path.join(".", "output")
# Subfolders
recon_dir    = os.path.join(output_path, "reconstructed")
residual_dir = os.path.join(output_path, "residual")
mv_dir       = os.path.join(output_path, "motion_vector")
# Make sure they exist
os.makedirs(recon_dir, exist_ok=True)
os.makedirs(residual_dir, exist_ok=True)
os.makedirs(mv_dir, exist_ok=True)

def _sad(a: np.ndarray, b: np.ndarray) -> int:
    """Sum of absolute differences using signed integers (prevents uint8 wrap-around)."""
    return int(np.sum(np.abs(a.astype(np.int16) - b.astype(np.int16))))

def full_search(reference_frame, current_frame, search_range):
    h, w = reference_frame.shape
    motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            best_match = (0, 0)  # (dy, dx)
            min_cost = float("inf")
            cur_blk = current_frame[i:i+block_size, j:j+block_size]

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ry, rx = i + dy, j + dx
                    if 0 <= ry <= h - block_size and 0 <= rx <= w - block_size:
                        ref_blk = reference_frame[ry:ry+block_size, rx:rx+block_size]
                        cost = _sad(cur_blk, ref_blk)
                        if cost < min_cost:
                            min_cost = cost
                            best_match = (dy, dx)

            motion_vectors[i // block_size, j // block_size] = best_match
    return motion_vectors

def three_step_search(reference_frame, current_frame, search_range):
    h, w = reference_frame.shape
    motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)
    step0 = max(1, search_range // 2)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            cur_blk = current_frame[i:i+block_size, j:j+block_size]
            best = (0, 0)  # (dy, dx)
            best_cost = _sad(cur_blk, reference_frame[i:i+block_size, j:j+block_size])

            step = step0
            # initial 9-point search
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    ry, rx = i + dy, j + dx
                    if 0 <= ry <= h - block_size and 0 <= rx <= w - block_size:
                        ref_blk = reference_frame[ry:ry+block_size, rx:rx+block_size]
                        cost = _sad(cur_blk, ref_blk)
                        if cost < best_cost:
                            best_cost = cost
                            best = (dy, dx)

            step //= 2
            while step >= 1:
                cand = best
                for dy in (-step, 0, step):
                    for dx in (-step, 0, step):
                        if dy == 0 and dx == 0:
                            continue
                        ry, rx = i + best[0] + dy, j + best[1] + dx
                        if 0 <= ry <= h - block_size and 0 <= rx <= w - block_size:
                            ref_blk = reference_frame[ry:ry+block_size, rx:rx+block_size]
                            cost = _sad(cur_blk, ref_blk)
                            if cost < best_cost:
                                best_cost = cost
                                cand = (best[0] + dy, best[1] + dx)
                best = cand
                step //= 2

            motion_vectors[i // block_size, j // block_size] = best
    return motion_vectors

def motion_compensation(reference_frame, motion_vectors):
    h, w = reference_frame.shape
    comp = np.zeros_like(reference_frame)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dy, dx = motion_vectors[i // block_size, j // block_size]
            ry, rx = i + dy, j + dx
            if 0 <= ry <= h - block_size and 0 <= rx <= w - block_size:
                comp[i:i+block_size, j:j+block_size] = reference_frame[ry:ry+block_size, rx:rx+block_size]
    return comp

def save_motion_field(motion_vectors: np.ndarray, block_size: int, out_path: str, overlay: np.ndarray | None = None):
    """Save a quiver plot of motion vectors (dy, dx)."""
    h_blocks, w_blocks, _ = motion_vectors.shape
    Yc, Xc = np.mgrid[0:h_blocks, 0:w_blocks]
    Yc = Yc * block_size + block_size / 2.0
    Xc = Xc * block_size + block_size / 2.0
    dy = motion_vectors[..., 0]
    dx = motion_vectors[..., 1]

    plt.figure(figsize=(8, 6))
    if overlay is not None:
        plt.imshow(overlay, cmap='gray', interpolation='nearest')
    plt.quiver(Xc, Yc, dx, -dy, angles='xy', scale_units='xy', scale=1)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title('Motion Vectors')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def me_mc(reference_frame, current_frame, search_range, algorithm):
    t0 = time.time()
    if algorithm == "full_search":
        mv = full_search(reference_frame, current_frame, search_range)
    else: # three_step_search
        mv = three_step_search(reference_frame, current_frame, search_range)
    recon = motion_compensation(reference_frame, mv)
    t1 = time.time()

    # residual visualization (absolute)
    diff = current_frame.astype(np.int16) - recon.astype(np.int16)
    residual = np.abs(diff).astype(np.uint8)

    val = psnr(current_frame, recon, data_range=255)
    runtime = t1 - t0

    # Save files into subfolders
    cv2.imwrite(os.path.join(recon_dir,    f"reconstructed_{algorithm}_{search_range}.png"), recon)
    cv2.imwrite(os.path.join(residual_dir, f"residual_{algorithm}_{search_range}.png"),       residual)
    save_motion_field(mv, block_size, os.path.join(mv_dir, f"motion_vectors_{algorithm}_{search_range}.png"))

    return val, runtime, mv

if __name__ == "__main__":
    if not os.path.exists("one_gray.png") or not os.path.exists("two_gray.png"):
        print("Error: one_gray.png or two_gray.png not found.")
        exit(1)

    reference_frame = cv2.imread("one_gray.png", cv2.IMREAD_GRAYSCALE)  # previous frame
    current_frame   = cv2.imread("two_gray.png", cv2.IMREAD_GRAYSCALE)  # current frame
    assert reference_frame is not None and current_frame is not None
    assert reference_frame.shape == current_frame.shape

    print("Start motion estimation and compensation comparison ...")
    print("=" * 60)

    for sr in [8, 16, 32]:
        p_full, t_full, mv_full = me_mc(reference_frame, current_frame, sr, algorithm="full_search")
        p_tss,  t_tss,  mv_tss  = me_mc(reference_frame, current_frame, sr, algorithm="three_step_search")

        print(f"Search Range: ±{sr}")
        print(f"  Full     : PSNR = {p_full:.3f} dB, Time = {t_full:.3f} s")
        print(f"  ThreeStep: PSNR = {p_tss:.3f} dB, Time = {t_tss:.3f} s")
        print("=" * 60)
