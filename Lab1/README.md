# Homework #1 – Color Transform

**Author:** 313553024 蘇柏叡
**Date:** 2025/09/15

---

## Requirement

* Represent **lena.png** in **RGB, YUV, YCbCr**.

  1. **RGB → YUV**
     <img width="339" height="66" alt="RGB→YUV" src="https://github.com/user-attachments/assets/2a42c9e2-765e-48d4-8d63-6fc33b87cdcd" />
  2. **RGB → YCbCr**
     <img width="842" height="172" alt="RGB→YCbCr" src="https://github.com/user-attachments/assets/80f0f24a-47cb-43e7-8c85-953fb55a9cf8" />

* Language: C/C++/Python/MATLAB（We use python to complete this lab.）

* Output: **8-bit grayscale** images — `R, G, B, Y, U, V, Cb, Cr`

* **No** ready-made color transform functions

* Image I/O APIs allowed

* Deadline: **2025/09/29 13:19**

* Submit as a single ZIP

**Required files**

1. `VC_HW1_[student_id].pdf` — Report
2. `VC_HW1_[student_id].zip` — Source code + **README**

---

## Environment

* Python 3.8+
* numpy
* Pillow

```bash
pip install requriements.txt
```


---

## How to Run

```bash
python color_transform.py --input /path/to/lena.png --outdir ./result
```

* `--input, -i`：輸入影像路徑（任一 Pillow 可讀格式，程式內統一轉為 RGB）
* `--outdir, -o`：輸出資料夾（自動建立）

---

## Output

```
result/
├─ R.png   ├─ G.png   ├─ B.png
├─ Y.png   ├─ U.png   ├─ V.png
├─ Cb.png  └─ Cr.png
```

* 全部為 **8-bit 單通道** PNG（Pillow `L` 模式）

---

## Notes

* 僅使用 Pillow 進行 **讀寫與通道統一**（`convert("RGB")`），色彩轉換皆以手動係數計算。
請幫我用精美一點的排版 並且提供我markdown格式讓我複製
