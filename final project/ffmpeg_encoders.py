import subprocess
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EncodeOptions:
    codec: str  # libx264, libx265, libaom-av1, libsvtav1
    crf: int
    pix_fmt: str = "yuv420p"
    remove_audio: bool = True
    x26x_preset: str = "fast"  # for x264/x265
    aom_cpu_used: int = 6      # 0..8
    svt_preset: int = 8        # 0..13
    disable_bframes: bool = False
    scale: Optional[str] = None  # e.g., "1280:720" or "-2:720"

def has_encoder(encoder_name: str) -> bool:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return (r.returncode == 0) and (encoder_name in r.stdout)

def build_ffmpeg_cmd(input_path: str, output_path: str, opt: EncodeOptions, segment_seconds: Optional[float]=None) -> List[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-i", input_path]

    if segment_seconds is not None and segment_seconds > 0:
        cmd += ["-t", str(segment_seconds)]

    if opt.remove_audio:
        cmd += ["-an"]

    # Add scale filter if specified
    if opt.scale:
        cmd += ["-vf", f"scale={opt.scale}"]

    cmd += ["-pix_fmt", opt.pix_fmt, "-c:v", opt.codec]
    
    # GPU 編碼器 (NVENC/QSV) 使用 -qp 或 -global_quality
    gpu_nvenc = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
    gpu_qsv = ("h264_qsv", "hevc_qsv", "av1_qsv")
    
    if opt.codec in gpu_nvenc:
        # NVENC: 使用 -cq (constant quality) 模式
        cmd += ["-rc", "constqp", "-qp", str(opt.crf)]
        cmd += ["-preset", "p4"]  # balanced preset
    elif opt.codec in gpu_qsv:
        # QSV: 使用 -global_quality
        cmd += ["-global_quality", str(opt.crf)]
        cmd += ["-preset", "medium"]
    elif opt.codec in ("libx264", "libx265"):
        cmd += ["-crf", str(opt.crf)]
        cmd += ["-preset", opt.x26x_preset]
        if opt.disable_bframes:
            cmd += ["-bf", "0"]
    elif opt.codec == "libaom-av1":
        cmd += ["-crf", str(opt.crf), "-b:v", "0", "-cpu-used", str(opt.aom_cpu_used)]
    elif opt.codec == "libsvtav1":
        cmd += ["-crf", str(opt.crf), "-preset", str(opt.svt_preset)]
    else:
        # 未知編碼器，嘗試使用 -crf
        cmd += ["-crf", str(opt.crf)]

    cmd += [output_path]
    return cmd

def run_encode(input_path: str, output_path: str, opt: EncodeOptions, segment_seconds: Optional[float]=None) -> None:
    cmd = build_ffmpeg_cmd(input_path, output_path, opt, segment_seconds=segment_seconds)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg encode failed")

def run_encode_with_progress(
    input_path: str, 
    output_path: str, 
    opt: EncodeOptions, 
    total_duration: float,
    progress_callback=None,
    segment_seconds: Optional[float]=None
) -> None:
    """
    執行 ffmpeg 編碼並透過 callback 回報進度
    
    Args:
        input_path: 輸入視頻路徑
        output_path: 輸出視頻路徑
        opt: 編碼選項
        total_duration: 視頻總時長（秒）
        progress_callback: 進度回調函數，接收 0.0-1.0 的進度值
        segment_seconds: 只編碼前 N 秒
    """
    import re
    import threading
    
    cmd = build_ffmpeg_cmd(input_path, output_path, opt, segment_seconds=segment_seconds)
    # 在命令開頭（ffmpeg 之後）插入 -progress pipe:1
    cmd.insert(1, "-progress")
    cmd.insert(2, "pipe:1")
    
    effective_duration = segment_seconds if segment_seconds else total_duration
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # 用線程讀取 stderr 避免緩衝區滿導致阻塞
    stderr_output = []
    def read_stderr():
        for line in proc.stderr:
            stderr_output.append(line)
    
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    
    time_pattern = re.compile(r"out_time_ms=(\d+)")
    
    try:
        for line in proc.stdout:
            match = time_pattern.search(line)
            if match and progress_callback:
                time_ms = int(match.group(1))
                time_sec = time_ms / 1_000_000
                progress = min(1.0, time_sec / effective_duration) if effective_duration > 0 else 0
                progress_callback(progress)
        
        proc.wait()
        stderr_thread.join(timeout=5)
        
        if proc.returncode != 0:
            stderr_text = "".join(stderr_output)
            raise RuntimeError(stderr_text.strip() or "ffmpeg encode failed")
    finally:
        if proc.poll() is None:
            proc.terminate()

