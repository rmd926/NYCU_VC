import subprocess
import re

def compute_psnr(orig_path: str, comp_path: str) -> float:
    """Compute average PSNR (orig is scaled to match comp). Returns 0.0 on failure."""
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", comp_path, "-i", orig_path,
        "-lavfi",
        # dist=compressed, ref=original
        "[0:v]setpts=PTS-STARTPTS[dist];"
        "[1:v]setpts=PTS-STARTPTS[ref];"
        # scale ref to match dist resolution
        "[ref][dist]scale2ref[ref2][dist2];"
        # compare at dist resolution
        "[dist2][ref2]psnr",
        "-f", "null", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0

    for line in proc.stderr.splitlines():
        if "PSNR" in line and "average:" in line:
            try:
                return float(line.split("average:")[1].split()[0])
            except Exception:
                return 0.0
    return 0.0


def compute_ssim(orig_path: str, comp_path: str) -> float:
    """Compute average SSIM (orig is scaled to match comp). Returns 0.0 on failure."""
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", comp_path, "-i", orig_path,
        "-lavfi",
        # dist=compressed, ref=original
        "[0:v]setpts=PTS-STARTPTS[dist];"
        "[1:v]setpts=PTS-STARTPTS[ref];"
        # scale ref to match dist resolution
        "[ref][dist]scale2ref[ref2][dist2];"
        # compare at dist resolution
        "[dist2][ref2]ssim",
        "-f", "null", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0

    m = re.search(r"All:([0-9.]+)", proc.stderr)
    return float(m.group(1)) if m else 0.0
