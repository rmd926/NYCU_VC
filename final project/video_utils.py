import json
import subprocess
from dataclasses import dataclass
from typing import Optional

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    nb_frames: Optional[int]
    duration: Optional[float]  # seconds

    @property
    def pixels_per_frame(self) -> int:
        return int(self.width * self.height)

def _run_ffprobe(args) -> str:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", *args],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    return proc.stdout.strip()

def get_video_info(path: str) -> VideoInfo:
    #Returns width/height/fps/nb_frames/duration for the first video stream.\"\"\"
    out = _run_ffprobe([
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json",
        path
    ])
    data = json.loads(out)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found")

    s = streams[0]
    w = int(s.get("width", 0))
    h = int(s.get("height", 0))

    afr = s.get("avg_frame_rate", "0/1")
    try:
        num, den = afr.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except Exception:
        fps = 0.0

    nb_frames = s.get("nb_frames", None)
    nb_frames = int(nb_frames) if nb_frames not in (None, "", "N/A") else None

    duration = s.get("duration", None)
    duration = float(duration) if duration not in (None, "", "N/A") else None

    if w <= 0 or h <= 0 or fps <= 0:
        out2 = _run_ffprobe([
            "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1",
            path
        ])
        try:
            fmt_dur = float(out2.strip())
        except Exception:
            fmt_dur = None
        if duration is None:
            duration = fmt_dur

    return VideoInfo(path=path, width=w, height=h, fps=fps, nb_frames=nb_frames, duration=duration)

def estimate_frames(info: VideoInfo, segment_seconds: float) -> int:
    #Estimate frames for a segment when nb_frames is unknown.\"\"\"
    return max(1, int(round(info.fps * segment_seconds)))
