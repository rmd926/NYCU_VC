import os
import time
import json
import uuid
from typing import Tuple

import gradio as gr

from video_utils import get_video_info
from ffmpeg_encoders import EncodeOptions, has_encoder, run_encode, run_encode_with_progress
from bpp_search import find_crf_for_target_bpp
from metrics import compute_psnr, compute_ssim

OUTPUT_DIR = os.path.abspath("outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X26X_PRESETS = ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]
PIX_FMTS = ["yuv420p", "yuv422p", "yuv444p"]
RESOLUTIONS = ["Original", "1080p", "720p", "480p", "360p"]

def _resolution_to_scale(res: str) -> str:
    """Convert resolution choice to ffmpeg scale string. Returns None for Original."""
    mapping = {
        "Original": None,
        "1080p": "-2:1080",
        "720p": "-2:720",
        "480p": "-2:480",
        "360p": "-2:360",
    }
    return mapping.get(res)

def _human_size(num_bytes: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    x = float(num_bytes)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"

def _compute_bpp_from_output(output_path: str, info) -> float:
    if info.nb_frames is not None:
        frames = info.nb_frames
    else:
        dur = info.duration or 0.0
        frames = max(1, int(round(info.fps * dur)))
    bits = os.path.getsize(output_path) * 8
    denom = max(1, info.width * info.height * frames)
    return bits / denom

def _validate_encoder(selected_codec: str) -> None:
    if not has_encoder(selected_codec):
        raise gr.Error(f"Encoder '{selected_codec}' is not supported by ffmpeg. Please change encoder or update ffmpeg.")

def get_bitrate_per_second(video_path: str) -> list:
    """使用 ffprobe 獲取每秒碼率"""
    import subprocess
    import json as jsonlib
    
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,size",
        "-of", "json",
        video_path
    ]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    
    try:
        data = jsonlib.loads(proc.stdout)
        packets = data.get("packets", [])
        
        # 按秒分組計算碼率
        bitrates = {}
        for pkt in packets:
            pts = float(pkt.get("pts_time", 0))
            size = int(pkt.get("size", 0))
            second = int(pts)
            if second not in bitrates:
                bitrates[second] = 0
            bitrates[second] += size * 8  # bytes to bits
        
        # 轉換為 kbps 列表
        if not bitrates:
            return []
        max_sec = max(bitrates.keys())
        result = []
        for i in range(max_sec + 1):
            kbps = bitrates.get(i, 0) / 1000  # bits to kbps
            result.append({"second": i, "kbps": kbps})
        return result
    except:
        return []

def create_bitrate_chart(bitrate_data: list):
    """創建碼率分佈圖"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    if not bitrate_data:
        return None
    
    seconds = [d["second"] for d in bitrate_data]
    kbps = [d["kbps"] for d in bitrate_data]
    avg_kbps = sum(kbps) / len(kbps) if kbps else 0
    
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2a2a2a')
    
    # 填充區域圖
    ax.fill_between(seconds, kbps, alpha=0.4, color='#00d4ff')
    ax.plot(seconds, kbps, color='#00d4ff', linewidth=1.5, label='Bitrate')
    
    # 平均碼率線
    ax.axhline(y=avg_kbps, color='#ff6b6b', linestyle='--', linewidth=2, 
               label=f'Average: {avg_kbps:.0f} kbps')
    
    ax.set_xlabel('Time (seconds)', color='white', fontsize=11)
    ax.set_ylabel('Bitrate (kbps)', color='white', fontsize=11)
    ax.set_title('Bitrate Distribution Over Time', color='white', fontsize=13, fontweight='bold')
    
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper right', facecolor='#3a3a3a', edgecolor='white', labelcolor='white')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig


def estimate_quality_curve(video_path: str, codec: str, opt_template, segment_seconds: float = 4.0, progress_callback=None):
    """對多個 CRF 值進行試編碼，獲取 CRF vs BPP 曲線"""
    import tempfile
    from video_utils import get_video_info, estimate_frames
    
    info = get_video_info(video_path)
    frames_est = estimate_frames(info, segment_seconds)
    
    # 根據 codec 決定 CRF 測試範圍
    if codec in ("libx264", "libx265", "h264_nvenc", "hevc_nvenc"):
        crf_values = [25, 30, 35, 40, 45, 50]
    elif codec in ("libaom-av1", "libsvtav1"):
        crf_values = [20, 28, 36, 44, 52, 60]
    # elif codec in ("h264_nvenc", "hevc_nvenc"):
    #     crf_values = [15, 20, 25, 30, 35, 40, 45]
    else:
        crf_values = [18, 24, 30, 36, 42]
    
    results = []
    
    with tempfile.TemporaryDirectory(prefix="vc_quality_") as tmp_dir:
        for i, crf_val in enumerate(crf_values):
            if progress_callback:
                progress_callback((i + 1) / len(crf_values), f"Testing CRF {crf_val}...")
            
            out_path = os.path.join(tmp_dir, f"test_crf{crf_val}.mp4")
            opt = EncodeOptions(**{**opt_template.__dict__, "crf": int(crf_val)})
            
            try:
                run_encode(video_path, out_path, opt, segment_seconds=segment_seconds)
                file_size = os.path.getsize(out_path)
                bpp = (file_size * 8) / (info.width * info.height * frames_est) if frames_est > 0 else 0
                # Estimate full video size
                if info.duration and info.duration > 0:
                    est_full_size = file_size * (info.duration / segment_seconds)
                else:
                    est_full_size = file_size
                
                results.append({
                    "crf": crf_val,
                    "bpp": bpp,
                    "file_size_mb": est_full_size / (1024 * 1024)
                })
            except Exception as e:
                print(f"Warning: Failed to encode with CRF {crf_val}: {e}")
    
    return results

def create_quality_chart(quality_data: list, codec: str):
    """創建品質預估對比圖"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    if not quality_data:
        return None
    
    crfs = [d["crf"] for d in quality_data]
    bpps = [d["bpp"] for d in quality_data]
    sizes = [d["file_size_mb"] for d in quality_data]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a1a')
    ax1.set_facecolor('#2a2a2a')
    
    # BPP 曲線 (左軸)
    color_bpp = '#00d4ff'
    ax1.set_xlabel('CRF / QP', color='white', fontsize=12)
    ax1.set_ylabel('BPP', color=color_bpp, fontsize=12)
    line1 = ax1.plot(crfs, bpps, 'o-', color=color_bpp, linewidth=2, markersize=8, label='BPP')
    ax1.tick_params(axis='y', labelcolor=color_bpp)
    ax1.tick_params(axis='x', colors='white')
    
    # 檔案大小曲線 (右軸)
    ax2 = ax1.twinx()
    color_size = '#6bcf63'
    ax2.set_ylabel('Est. Size (MB)', color=color_size, fontsize=12)
    line2 = ax2.plot(crfs, sizes, 's--', color=color_size, linewidth=2, markersize=8, label='Est. Size')
    ax2.tick_params(axis='y', labelcolor=color_size)
    
    # 圖例
    lines = line1 + line2
    labels = ['BPP', 'Est. File Size (MB)']
    ax1.legend(lines, labels, loc='upper right', facecolor='#3a3a3a', edgecolor='white', labelcolor='white')
    
    # 標題
    plt.title(f'Quality Estimation: CRF vs BPP / File Size ({codec})', color='white', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_search_chart(trials: list, target_bpp: float, best_crf: int):
    """創建 CRF 搜尋過程圖"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非互動式後端
    
    if not trials:
        return None
    
    # 提取數據
    trial_nums = list(range(1, len(trials) + 1))
    crfs = [t["crf"] for t in trials]
    bpps = [t["bpp"] for t in trials]
    
    # 創建圖表
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a1a')
    ax1.set_facecolor('#2a2a2a')
    
    # BPP 曲線 (左軸)
    color_bpp = '#00d4ff'
    ax1.set_xlabel('Trial', color='white', fontsize=12)
    ax1.set_ylabel('BPP', color=color_bpp, fontsize=12)
    line1 = ax1.plot(trial_nums, bpps, 'o-', color=color_bpp, linewidth=2, markersize=8, label='BPP')
    ax1.tick_params(axis='y', labelcolor=color_bpp)
    ax1.tick_params(axis='x', colors='white')
    
    # 目標 BPP 線
    target_line = ax1.axhline(y=target_bpp, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Target BPP ({target_bpp})')
    
    # CRF 曲線 (右軸)
    ax2 = ax1.twinx()
    color_crf = '#ffd93d'
    ax2.set_ylabel('CRF', color=color_crf, fontsize=12)
    line2 = ax2.plot(trial_nums, crfs, 's--', color=color_crf, linewidth=2, markersize=8, label='CRF')
    ax2.tick_params(axis='y', labelcolor=color_crf)
    
    # 標註最佳 CRF
    best_idx = next((i for i, t in enumerate(trials) if t["crf"] == best_crf), -1)
    if best_idx >= 0:
        ax1.annotate(f'Best CRF: {best_crf}', 
                     xy=(best_idx + 1, bpps[best_idx]), 
                     xytext=(best_idx + 1.5, bpps[best_idx] * 1.1),
                     fontsize=10, color='#6bcf63',
                     arrowprops=dict(arrowstyle='->', color='#6bcf63'))
    
    # 圖例
    lines = line1 + line2 + [target_line]
    labels = ['BPP', 'CRF', f'Target BPP ({target_bpp})']
    ax1.legend(lines, labels, loc='upper right', facecolor='#3a3a3a', edgecolor='white', labelcolor='white')
    
    # 標題
    plt.title('CRF Binary Search Process', color='white', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_comparison_video(original_path: str, compressed_path: str, output_path: str) -> str:
    """創建並排對比視頻（左側原始，右側壓縮後）"""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", original_path,
        "-i", compressed_path,
        "-filter_complex",
        "[0:v]scale=640:-2[v0];[1:v]scale=640:-2[v1];[v0][v1]hstack=inputs=2[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        output_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create comparison video: {proc.stderr}")
    return output_path

def compress(
    video_path: str,
    mode: str,
    codec: str,
    crf: int,
    target_bpp: float,
    x26x_preset: str,
    aom_cpu_used: int,
    svt_preset: int,
    pix_fmt: str,
    resolution: str,
    disable_bframes: bool,
    trial_seconds: float,
    compute_psnr_flag: bool,
    progress=gr.Progress(track_tqdm=False),
):
    # 先清空之前的輸出 (包含品質圖表)
    yield None, None, "", None, None, gr.update(visible=False, value=None)
    
    if not video_path:
        raise gr.Error("Please upload a video first.")
    _validate_encoder(codec)

    info = get_video_info(video_path)

    opt_template = EncodeOptions(
        codec=codec,
        crf=int(crf),
        pix_fmt=pix_fmt,
        x26x_preset=x26x_preset,
        aom_cpu_used=int(aom_cpu_used),
        svt_preset=int(svt_preset),
        disable_bframes=bool(disable_bframes),
        scale=_resolution_to_scale(resolution),
    )

    chosen_crf = int(crf)
    search_detail = None

    if mode == "Target BPP (auto find CRF/QP)":
        if target_bpp <= 0:
            raise gr.Error("target_bpp must be > 0.")
        progress(0, desc="Searching for optimal CRF...")
        sr = find_crf_for_target_bpp(
            input_path=video_path,
            opt_template=opt_template,
            target_bpp=float(target_bpp),
            segment_seconds=float(trial_seconds),
            max_iters=10,
            tolerance=0.02,
        )
        chosen_crf = sr.best_crf
        search_detail = {"best_crf": sr.best_crf, "best_bpp_est": sr.best_bpp, "trials": sr.trials}
        progress(0.3, desc=f"Found optimal CRF: {chosen_crf}")

    out_name = f"{uuid.uuid4().hex}_{codec}_crf{chosen_crf}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    opt_final = EncodeOptions(**{**opt_template.__dict__, "crf": int(chosen_crf)})
    
    # 取得視頻時長用於計算進度
    total_duration = info.duration or 0.0
    if total_duration <= 0 and info.nb_frames and info.fps > 0:
        total_duration = info.nb_frames / info.fps
    
    # Target BPP 模式下搜尋已佔 30%，壓縮從 30% 開始
    is_bpp_mode = mode == "Target BPP (auto find CRF)"
    if not is_bpp_mode:
        progress(0, desc="開始壓縮...")
    
    t0 = time.perf_counter()
    run_encode_with_progress(
        video_path, out_path, opt_final, 
        total_duration=total_duration,
        progress_callback=lambda p: progress(0.3 + p * 0.6 if is_bpp_mode else p * 0.9, desc="Compressing..."),
        segment_seconds=None
    )
    t1 = time.perf_counter()

    out_size = os.path.getsize(out_path)
    # Use output video info for BPP calculation (important when scaling)
    out_info = get_video_info(out_path)
    bpp = _compute_bpp_from_output(out_path, out_info)

    psnr = None
    ssim = None
    if compute_psnr_flag:
        psnr = compute_psnr(video_path, out_path)
        ssim = compute_ssim(video_path, out_path)

    summary = {
        "input": {
            "path": video_path,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "nb_frames": info.nb_frames,
            "duration": info.duration,
            "size": _human_size(os.path.getsize(video_path)),
        },
        "encode": {
            "codec": codec,
            "mode": mode,
            "crf_used": chosen_crf,
            "pix_fmt": pix_fmt,
            "resolution": resolution,
            "output_width": out_info.width,
            "output_height": out_info.height,
            "disable_bframes": disable_bframes if codec in ("libx264","libx265") else None,
            "x26x_preset": x26x_preset if codec in ("libx264","libx265") else None,
            "aom_cpu_used": aom_cpu_used if codec == "libaom-av1" else None,
            "svt_preset": svt_preset if codec == "libsvtav1" else None,
            "trial_seconds": trial_seconds if mode.startswith("Target BPP") else None,
            "time_sec": round(t1 - t0, 3),
            "output_path": out_path,
            "output_size": _human_size(out_size),
            "bpp": round(bpp, 6),
            "psnr": None if psnr is None else round(psnr, 3),
            "ssim": None if ssim is None else round(ssim, 4),
        },
        "search": search_detail,
    }

    # 創建 CRF 搜尋過程圖 (Target BPP 模式)
    search_chart = None
    if search_detail and search_detail.get("trials"):
        search_chart = create_search_chart(
            trials=search_detail["trials"],
            target_bpp=float(target_bpp),
            best_crf=search_detail["best_crf"]
        )

    # 創建碼率分佈圖 (兩個模式都顯示)
    progress(0.92, desc="Analyzing bitrate distribution...")
    bitrate_data = get_bitrate_per_second(out_path)
    bitrate_chart = create_bitrate_chart(bitrate_data)

    # 創建並排對比視頻
    progress(0.95, desc="Generating comparison video...")
    comparison_name = f"{uuid.uuid4().hex}_comparison.mp4"
    comparison_path = os.path.join(OUTPUT_DIR, comparison_name)
    create_comparison_video(video_path, out_path, comparison_path)
    progress(1.0, desc="Done!")

    yield comparison_path, out_path, json.dumps(summary, ensure_ascii=False, indent=2), search_chart, bitrate_chart, gr.update(visible=False)
def build_ui():
    # ✅ 不指定 theme → 回到 Gradio 預設白底
    with gr.Blocks(title="Video Compression UI") as demo:
        gr.Markdown(
            "# Video Compression\n"
            "H.264 / H.265 / AV1 / SVT-AV1 / NVENC"
        )

        with gr.Accordion("ℹ How to Use", open=False):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(
                            "### 使用步驟\n"
                            "1. 上傳影片（拖曳或選擇檔案）\n"
                            "2. 選擇模式（CRF/QP 或 Target BPP）\n"
                            "3. 選擇編碼器（軟體或 GPU 加速）\n"
                            "4. 調整參數（CRF/QP 等）\n"
                            "5. 點擊 Analyze Quality 進行分析\n"
                            "6. 點擊 Compress 開始壓縮影片\n"
                            "7. 查看輸出影片、結果summary.json"
                        )
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(
                            "### 模式說明\n"
                            "- **CRF/QP**：手動設定品質值\n"
                            "- **Target BPP**：自動搜尋最佳 CRF/QP\n"
                            "- CRF/QP 越小品質越高、檔案通常越大\n"
                            "- CRF/QP 越大檔案通常越小、品質越低\n"
                            "- 可依據分析品質結果後選擇最適切的CRF/QP/Target BPP"
                        )
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(
                            "### 編碼器建議\n"
                            "- **libx264**：H.264，相容性最好\n"
                            "- **libx265**：H.265，壓縮率較佳\n"
                            "- **libsvtav1**：AV1，較實用的速度/效率折衷\n"
                            "- **libaom-av1**：AV1，品質參考但通常更慢\n"
                            "- **h264_nvenc / hevc_nvenc**：GPU 加速"
                        )

        with gr.Row():
            video = gr.Video(label="Upload video")
            with gr.Column():
                mode = gr.Radio(
                    choices=["CRF/QP (manual)", "Target BPP (auto find CRF/QP)"],
                    value="CRF/QP (manual)",
                    label="Mode"
                )
                codec = gr.Dropdown(
                    choices=["libx264", "libx265", "libaom-av1", "libsvtav1", "h264_nvenc", "hevc_nvenc"],
                    value="libx265",
                    label="Video Encoder"
                )
                crf = gr.Slider(0, 51, value=32, step=1, label="CRF (0=best, larger=smaller/worse)", visible=True)
                target_bpp = gr.Number(value=0.12, label="Target BPP (e.g., 0.12)", visible=False)

                def update_crf_range(selected_codec):
                    if selected_codec in ("libx264", "libx265"):
                        return gr.Slider(minimum=0, maximum=51, value=23, step=1,
                                         label="CRF (0=best, larger=smaller/worse)")
                    elif selected_codec in ("libaom-av1", "libsvtav1"):
                        return gr.Slider(minimum=0, maximum=63, value=32, step=1,
                                         label="CRF (0=best, larger=smaller/worse)")
                    elif selected_codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
                        return gr.Slider(minimum=0, maximum=51, value=23, step=1,
                                         label="QP (0=best, larger=smaller/worse)")
                    elif selected_codec in ("h264_qsv", "hevc_qsv", "av1_qsv"):
                        return gr.Slider(minimum=1, maximum=51, value=23, step=1,
                                         label="Quality (1=best, larger=smaller/worse)")
                    else:
                        return gr.Slider(minimum=0, maximum=51, value=23, step=1, label="Quality")

                codec.change(fn=update_crf_range, inputs=[codec], outputs=[crf], show_progress=False)

                def update_mode_visibility(selected_mode):
                    if selected_mode == "CRF/QP (manual)":
                        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

                pix_fmt = gr.Dropdown(choices=PIX_FMTS, value="yuv420p", label="Pixel format")
                resolution = gr.Dropdown(choices=RESOLUTIONS, value="Original", label="Output Resolution")
                disable_bframes = gr.Checkbox(value=False, label="Disable B-frames (-bf 0) [x264/x265 only]")

        gr.Markdown("### Speed Controls (only some fields apply depending on codec)")
        with gr.Row():
            x26x_preset = gr.Dropdown(choices=X26X_PRESETS, value="fast", label="x264/x265 preset")
            aom_cpu_used = gr.Slider(0, 8, value=6, step=1, label="libaom-av1 cpu-used (0 slow .. 8 fast)")
            svt_preset = gr.Slider(0, 13, value=8, step=1, label="libsvtav1 preset (0 slow .. 13 fast)")

        with gr.Row():
            trial_seconds = gr.Slider(2, 12, value=6, step=1, label="Trial seconds (Target BPP mode)", visible=False)
            compute_psnr_flag = gr.Checkbox(value=False, label="Compute PSNR/SSIM (requires decoding; slower)")

        with gr.Row():
            btn = gr.Button("Compress")
            btn_analyze = gr.Button("Analyze Quality")

        # ✅ 新增：Analyze 的狀態列（按下就立刻顯示）
        status_analyze = gr.Markdown("", visible=False)

        quality_chart = gr.Plot(label="Quality Estimation: CRF vs BPP / File Size", visible=False)

        gr.Markdown("### Comparison (Left: Original / Right: Compressed)")
        comparison_video = gr.Video(label="Side-by-Side Comparison", interactive=False)

        search_chart = gr.Plot(label="CRF Search Process (Target BPP mode)", visible=False)
        bitrate_chart = gr.Plot(label="Bitrate Distribution Over Time")

        out_file = gr.File(label="Download compressed video")
        out_json = gr.Code(label="Summary (JSON)", language="json")

        btn.click(
            fn=compress,
            inputs=[video, mode, codec, crf, target_bpp, x26x_preset, aom_cpu_used, svt_preset,
                    pix_fmt, resolution, disable_bframes, trial_seconds, compute_psnr_flag],
            outputs=[comparison_video, out_file, out_json, search_chart, bitrate_chart, quality_chart]
        )

        # ===== Analyze Quality: 先顯示狀態，再跑分析，最後自動隱藏狀態 =====
        def show_analyzing():
            return gr.update(visible=True, value="⏳ **Analyzing...** Please wait.")

        def hide_analyzing():
            return gr.update(visible=False, value="")

        def analyze_quality(video_path, codec, x26x_preset, aom_cpu_used, svt_preset,
                            pix_fmt, resolution, disable_bframes,
                            progress=gr.Progress(track_tqdm=False)):
            if not video_path:
                raise gr.Error("Please upload a video first.")
            _validate_encoder(codec)

            opt_template = EncodeOptions(
                codec=codec,
                crf=28,  # Default value, will be overridden
                pix_fmt=pix_fmt,
                
                x26x_preset=x26x_preset,
                aom_cpu_used=int(aom_cpu_used),
                svt_preset=int(svt_preset),
                disable_bframes=bool(disable_bframes),
                scale=_resolution_to_scale(resolution),
            )

            progress(0, desc="Starting analysis...")
            quality_data = estimate_quality_curve(
                video_path=video_path,
                codec=codec,
                opt_template=opt_template,
                segment_seconds=4.0,
                progress_callback=lambda p, desc: progress(p, desc=desc),
            )

            chart = create_quality_chart(quality_data, codec)
            progress(1.0, desc="Analysis complete!")

            # 回傳：顯示圖表 + 隱藏狀態列
            return gr.update(visible=True, value=chart), hide_analyzing()

        btn_analyze.click(
            fn=show_analyzing,
            inputs=[],
            outputs=[status_analyze],
            queue=False,  # ✅ 立刻刷新 UI
        ).then(
            fn=analyze_quality,
            inputs=[video, codec, x26x_preset, aom_cpu_used, svt_preset, pix_fmt, resolution, disable_bframes],
            outputs=[quality_chart, status_analyze],
            queue=True,   # ✅ 長任務走 queue
        )

        # ===== mode 切換：清空輸出並更新顯示 =====
        def clear_and_update_mode(selected_mode):
            crf_update, bpp_update, trial_update = update_mode_visibility(selected_mode)
            show_search_chart = selected_mode == "Target BPP (auto find CRF/QP)"
            return (
                crf_update, bpp_update, trial_update,
                None, None, "",
                gr.update(visible=show_search_chart, value=None),
                None,
                gr.update(visible=False, value=None)
            )

        mode.change(
            fn=clear_and_update_mode,
            inputs=[mode],
            outputs=[crf, target_bpp, trial_seconds, comparison_video, out_file, out_json, search_chart, bitrate_chart, quality_chart],
            show_progress=False
        )

        gr.Markdown(
            "### Notes\n"
            "- Target BPP：對前幾秒試編，二分搜尋 CRF，使估計 BPP 接近目標，再輸出全片。\n"
            "- CRF 不可跨 codec 當作同等品質；建議用 BPP/bitrate + 指標比較。"
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue() 
    demo.launch(share = True)

