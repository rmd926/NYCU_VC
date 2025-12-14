# Video Compression UI 使用說明

## 模式（Mode）

| 模式 | 說明 |
|------|------|
| **CRF/QP (manual)** | 手動指定 CRF/QP 值，數值越小品質越高、檔案越大 |
| **Target BPP (auto find CRF/QP)** | 設定目標 BPP，系統會自動搜尋最佳 CRF/QP 值 |

---

## 編碼器（Video Encoder）

### 軟體編碼器

| Encoder | 說明 | CRF 範圍 |
|---------|------|----------|
| **libx264** | H.264 編碼器，相容性最好 | 0-51 |
| **libx265** | H.265/HEVC 編碼器，壓縮率更高 | 0-51 |
| **libaom-av1** | AV1 編碼器（較慢，品質佳） | 0-63 |
| **libsvtav1** | SVT-AV1 編碼器（較快的 AV1） | 0-63 |

### GPU 硬體編碼器 (NVIDIA NVENC)

| Encoder | 說明 | QP 範圍 | 需求 |
|---------|------|---------|------|
| **h264_nvenc** | NVIDIA H.264 硬體編碼 | 0-51 | NVIDIA GPU |
| **hevc_nvenc** | NVIDIA H.265/HEVC 硬體編碼 | 0-51 | NVIDIA GPU |

> **GPU 編碼特點**：速度快很多（可達 10x+），但壓縮效率略低於軟體編碼。建議 QP 值：18-28。

---

## CRF / QP (品質控制)

- **軟體編碼 (CRF)**：x264/x265 為 0-51，AV1 為 0-63
- **GPU 編碼 (QP)**：0-51
- **0** = 最高品質（檔案最大）
- **數值越大** = 品質越低、檔案越小


---

## Target BPP (Bits Per Pixel)

- 每個像素使用的位元數
- **較高的 BPP** = 較好的品質
- 系統會對前幾秒進行試編碼，二分搜尋最佳 CRF/QP

---

## Output Resolution（輸出解析度）

| 選項 | 說明 |
|------|------|
| **Original** | 保持原始解析度 |
| **1080p** | 縮放至 1080p（高度 1080，寬度自動） |
| **720p** | 縮放至 720p（高度 720，寬度自動） |
| **480p** | 縮放至 480p（高度 480，寬度自動） |
| **360p** | 縮放至 360p（高度 360，寬度自動） |

> **提示**：縮小解析度是減小檔案大小最有效的方法之一。

---

## Pixel Format（像素格式）

| 格式 | 說明 | 用途 |
|------|------|------|
| **yuv420p** | 4:2:0 取樣 | 最常用，適合大多數視頻、串流。檔案最小。 |
| **yuv422p** | 4:2:2 取樣 | 專業廣播、後期製作。色彩較好。 |
| **yuv444p** | 4:4:4 取樣 | 最高品質，專業用途。檔案最大。 |

---

## Speed Controls（速度控制）

### x264/x265 preset
| Preset | 速度 | 品質 |
|--------|------|------|
| ultrafast | 最快 | 較差 |
| superfast / veryfast / faster / fast | 快 | 中等 |
| **medium** | 中等 | 中等（預設） |
| slow / slower / veryslow | 慢 | 較好 |

### libaom-av1 cpu-used
- **0** = 最慢，品質最佳
- **8** = 最快，品質較差
- **建議值**：4-6

### libsvtav1 preset
- **0** = 最慢，品質最佳
- **13** = 最快，品質較差
- **建議值**：6-10

---

## 其他選項

| 選項 | 說明 |
|------|------|
| **Remove audio (-an)** | 移除音軌，減少檔案大小 |
| **Disable B-frames (-bf 0)** | 禁用 B-frame（僅 x264/x265） |
| **Trial seconds** | Target BPP 模式下，用於試編碼的秒數 |
| **Compute PSNR/SSIM** | 計算 PSNR/SSIM 品質指標（需額外解碼） |

---

## 圖表功能

### 1. Analyze Quality（品質預估圖）
- 點擊「Analyze Quality」按鈕
- 對不同 CRF/QP 值進行試編碼
- 顯示 CRF/QP vs BPP / 檔案大小曲線
- 幫助選擇合適的 CRF/QP 值

### 2. CRF Search Process（CRF/QP 搜尋圖）
- 僅在 Target BPP 模式顯示
- 顯示二分搜尋過程
- 藍線：BPP 值、黃線：CRF/QP 值
- 紅色虛線：目標 BPP

### 3. Bitrate Distribution（碼率分佈圖）
- 顯示壓縮後視頻的每秒碼率
- 藍色區域：碼率變化
- 紅色虛線：平均碼率

---

## 並排視頻對比

- 壓縮完成後自動生成並排對比視頻
- 左側：原始影片
- 右側：壓縮後影片
- 兩個視頻同步播放，方便對比畫質差異
