import os
import tempfile
from dataclasses import dataclass
from typing import Tuple

from video_utils import get_video_info, estimate_frames
from ffmpeg_encoders import EncodeOptions, run_encode

@dataclass
class SearchResult:
    best_crf: int
    best_bpp: float
    trials: list  # list of dicts

def _crf_range(codec: str) -> Tuple[int, int]:
    if codec in ("libx264", "libx265"):
        return 0, 51
    if codec in ("libaom-av1", "libsvtav1"):
        return 0, 63
    return 0, 51

def _bpp_from_file(file_path: str, width: int, height: int, frames: int) -> float:
    bits = os.path.getsize(file_path) * 8
    denom = max(1, width * height * frames)
    return bits / denom

def find_crf_for_target_bpp(
    input_path: str,
    opt_template: EncodeOptions,
    target_bpp: float,
    segment_seconds: float = 6.0,
    max_iters: int = 10,
    tolerance: float = 0.02,
) -> SearchResult:
    #Binary search CRF using short trial encodes (first N seconds).\"\"\"
    info = get_video_info(input_path)
    frames_est = estimate_frames(info, segment_seconds)

    lo, hi = _crf_range(opt_template.codec)
    best = None
    trials = []

    with tempfile.TemporaryDirectory(prefix="vc_trials_") as tmp_dir:
        for _ in range(max_iters):
            mid = (lo + hi) // 2

            out_path = os.path.join(tmp_dir, f"trial_{opt_template.codec}_crf{mid}.mp4")
            opt = EncodeOptions(**{**opt_template.__dict__, "crf": int(mid)})
            run_encode(input_path, out_path, opt, segment_seconds=segment_seconds)

            # Use output video dimensions for BPP calculation (important for scaling)
            out_info = get_video_info(out_path)
            out_frames = estimate_frames(out_info, segment_seconds)
            bpp = _bpp_from_file(out_path, out_info.width, out_info.height, out_frames)
            trials.append({"crf": mid, "bpp": bpp})

            if best is None or abs(bpp - target_bpp) < abs(best[1] - target_bpp):
                best = (mid, bpp)

            if target_bpp > 0 and abs(bpp - target_bpp) / target_bpp <= tolerance:
                break

            # Higher CRF => lower bpp (usually)
            if bpp > target_bpp:
                lo = mid + 1
            else:
                hi = mid - 1

            if lo > hi:
                break

    if best is None:
        best = (opt_template.crf, 0.0)

    return SearchResult(best_crf=int(best[0]), best_bpp=float(best[1]), trials=trials)
