# Video Compression Final Project
> Team menber: 蘇柏叡、許良亦、王致雅、陳晴川

> Intro: A video compression tool with multiple encoders, GPU acceleration, and resolution scaling.

## Details of this system
See [note.md](note.md).

## Features

- **Multiple Encoders**: H.264, H.265, AV1, SVT-AV1
- **GPU Acceleration**: NVIDIA NVENC (h264_nvenc, hevc_nvenc)
- **Two Modes**: CRF manual mode, Target BPP auto-search mode
- **Resolution Scaling**: Original, 1080p, 720p, 480p, 360p
- **Real-time Progress Bar**: Shows compression progress
- **Side-by-Side Comparison**: Compare original vs compressed quality
- **Chart Analysis**:
  - CRF Search Process (Target BPP mode)
  - Bitrate Distribution
  - Quality Estimation (Analyze Quality)

## Requirements

- Python 3.8+
- `ffmpeg` and `ffprobe` must be in PATH
- For GPU acceleration: NVIDIA GPU + ffmpeg with NVENC support

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open the URL shown in terminal (default: http://127.0.0.1:7860)

## Usage

1. **Upload Video** - Drag and drop or select a video file
2. **Select Mode** - CRF (manual) or Target BPP (auto-search)
3. **Select Encoder** - Software or GPU accelerated
4. **Select Resolution** - Scale to 720p, 480p, etc. to reduce file size
5. **Adjust Parameters** - Set CRF/QP values, etc.
6. **Analyze Quality** (optional) - Click to preview different CRF effects
7. **Compress** - Click Compress to start

## CLI Mode (Optional)
> Please check the argumentparser before you use CLI mode.

![alt text](image.png)
Manual CRF:
```bash
python cli_encode.py --input Big_Buck_Bunny_1080_10s_5MB.mp4 --output out.mp4 --codec libx264 --mode bpp --target_bpp 0.01 --psnr --ssim  
```

Target BPP:
```bash
python cli_encode.py --input test.mp4 --output out.mp4 --codec libsvtav1 --mode bpp --target_bpp 0.01 --psnr --ssim
```

## Dependencies

- gradio >= 4.0.0
- matplotlib >= 3.5.0

## Project Structure

```
project_test/
├── app.py              # Main app (Gradio UI)
├── ffmpeg_encoders.py  # FFmpeg encoding logic
├── bpp_search.py       # Target BPP binary search
├── video_utils.py      # Video info utilities
├── metrics.py          # PSNR/SSIM calculation
├── cli_encode.py       # CLI mode
├── README.md           # Project documentation
├── note.md             # Usage guide
└── requirements.txt    # Dependencies
```

## Notes

- Lower CRF = higher quality, larger file size
- Scaling down resolution is one of the most effective ways to reduce file size
- Target BPP mode performs multiple trial encodes on the first few seconds, uses binary search to find CRF that achieves the target BPP, then encodes the full video
- CRF scales differ between encoders and cannot be directly compared; use BPP/bitrate + quality metrics for comparison
