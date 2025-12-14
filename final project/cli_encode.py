import argparse
import json
import time

from video_utils import get_video_info
from ffmpeg_encoders import EncodeOptions, has_encoder, run_encode
from bpp_search import find_crf_for_target_bpp
from metrics import compute_psnr

def main():
    ap = argparse.ArgumentParser(description="Video compression CLI (CRF or Target BPP)")
    ap.add_argument("--input", required=True, help="Input video path")
    ap.add_argument("--output", required=True, help="Output video path")
    ap.add_argument("--codec", default="libx265", choices=["libx264","libx265","libaom-av1","libsvtav1"])
    ap.add_argument("--mode", default="crf", choices=["crf","bpp"], help="crf=manual CRF, bpp=auto find CRF by target bpp")
    ap.add_argument("--crf", type=int, default=32)
    ap.add_argument("--target_bpp", type=float, default=0.12)
    ap.add_argument("--trial_seconds", type=float, default=6.0)
    ap.add_argument("--pix_fmt", default="yuv420p")
    ap.add_argument("--remove_audio", action="store_true")
    ap.add_argument("--keep_audio", dest="remove_audio", action="store_false")
    ap.set_defaults(remove_audio=True)
    ap.add_argument("--disable_bframes", action="store_true")
    ap.add_argument("--x26x_preset", default="fast")
    ap.add_argument("--aom_cpu_used", type=int, default=6)
    ap.add_argument("--svt_preset", type=int, default=8)
    ap.add_argument("--psnr", action="store_true", help="Compute PSNR (slower)")
    args = ap.parse_args()

    if not has_encoder(args.codec):
        raise SystemExit(f"ffmpeg does not support encoder: {args.codec}")

    info = get_video_info(args.input)

    opt_template = EncodeOptions(
        codec=args.codec,
        crf=int(args.crf),
        pix_fmt=args.pix_fmt,
        remove_audio=bool(args.remove_audio),
        x26x_preset=args.x26x_preset,
        aom_cpu_used=int(args.aom_cpu_used),
        svt_preset=int(args.svt_preset),
        disable_bframes=bool(args.disable_bframes),
    )

    chosen_crf = int(args.crf)
    search_detail = None

    if args.mode == "bpp":
        sr = find_crf_for_target_bpp(
            input_path=args.input,
            opt_template=opt_template,
            target_bpp=float(args.target_bpp),
            segment_seconds=float(args.trial_seconds),
        )
        chosen_crf = sr.best_crf
        search_detail = {"best_crf": sr.best_crf, "best_bpp_est": sr.best_bpp, "trials": sr.trials}

    opt_final = EncodeOptions(**{**opt_template.__dict__, "crf": chosen_crf})

    t0 = time.perf_counter()
    run_encode(args.input, args.output, opt_final, segment_seconds=None)
    t1 = time.perf_counter()

    psnr_val = compute_psnr(args.input, args.output) if args.psnr else None

    summary = {
        "input": {"path": args.input, "w": info.width, "h": info.height, "fps": info.fps, "nb_frames": info.nb_frames, "duration": info.duration},
        "encode": {"codec": args.codec, "mode": args.mode, "crf_used": chosen_crf, "time_sec": round(t1 - t0, 3), "output": args.output},
        "search": search_detail,
        "psnr": psnr_val,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
