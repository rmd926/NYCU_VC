# Homework #3 – Motion Estimation & Compensation

> **Author:** 313553024 蘇柏叡
> **Date:** 2025/11/03

---

## Requirement

* **Motion Estimation (ME)**

  * Block size: **8×8**
  * Search range: **[±8]**
  * Algorithm: **Full Search Block Matching**
  * Precision: **Integer**
* **Motion Compensation (MC)**

  * Save the **reconstructed frame** (after MC) and the **residual**
* **Search Range Study**

  * Compare **PSNR** and **runtime** for **[±8], [±16], [±32]**
* **Algorithm Study**

  * Implement **Three-Step Search (TSS)** and compare with **Full Search** in **PSNR** & **runtime**
* **Submission**

  * `VC_HW3_[student_id].pdf` – Report PDF
  * `VC_HW3_[student_id].zip` – Source code (C/C++/Python/MATLAB) + **README**
* **Deadline:** 2023/10/28 1:19 PM
* **Upload:** E3

---

## Environment

* Python 3.8+
* numpy
* opencv-python
* matplotlib
* *(optional, if using `skimage.metrics` for PSNR)* scikit-image

**Install**

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py
```

---

## Output

* 影像輸出資料夾結構（Output directory layout）：

```
output/
├─ reconstructed/     # 重建後影像 (after MC)
├─ residual/          # 殘差圖 (|current - reconstructed|)
└─ motion_vector/     # 區塊運動向量圖 (quiver)
```

---

## Notes

* 全搜尋（Full Search）與三步搜尋（Three-Step Search）皆使用區塊為單位的整數位移；成本函數採 **SAD**（Sum of Absolute Differences）。
* 量測指標：**PSNR（dB）** 與 **執行時間（秒）**；可於 `main.py` 調整搜尋範圍。
* 若在無 GUI 的環境請使用 `opencv-python-headless` 或關閉彈窗顯示。
