````markdown
# Homework #1 – Color Transform

**Author:** 313553024 蘇柏叡  
**Date:** 2025/09/15

---

## 📌 Requirement

- Represent **lena.png** in **RGB, YUV, YCbCr**.
  1) **RGB → YUV**  
     <img width="339" height="66" alt="RGB→YUV" src="https://github.com/user-attachments/assets/2a42c9e2-765e-48d4-8d63-6fc33b87cdcd" />
  2) **RGB → YCbCr**  
     <img width="842" height="172" alt="RGB→YCbCr" src="https://github.com/user-attachments/assets/80f0f24a-47cb-43e7-8c85-953fb55a9cf8" />

- Language: C/C++/Python/MATLAB（本作業以 **Python** 實作）
- Output: **8-bit grayscale** images → `R, G, B, Y, U, V, Cb, Cr`
- **Do not** use any ready-made color transform functions  
- Image I/O APIs are allowed
- Deadline: **2025/09/29 13:19**
- Submit as a single ZIP

**Required files**
1. `VC_HW1_[student_id].pdf` — Report  
2. `VC_HW1_[student_id].zip` — Source code + **README**

---

## 🧰 Environment

- Python 3.8+
- numpy
- Pillow

Install (pip):
```bash
pip install -r requirements.txt
````

> 若無 `requirements.txt`，可直接：

```bash
pip install numpy pillow
```

---

## ▶️ How to Run

```bash
python color_transform.py --input /path/to/lena.png --outdir ./result
```

**Arguments**

| Flag           | Required | Description                         |
| -------------- | -------- | ----------------------------------- |
| `--input, -i`  | Yes      | 輸入影像路徑（任一 Pillow 可讀格式；程式內會統一轉成 RGB） |
| `--outdir, -o` | Yes      | 輸出資料夾（若不存在將自動建立）                    |

---

## 📦 Output

```
result/
├─ R.png   ├─ G.png   ├─ B.png
├─ Y.png   ├─ U.png   ├─ V.png
├─ Cb.png  └─ Cr.png
```

* 全部為 **8-bit 單通道** PNG（Pillow `L` 模式）

---

## 📐 Notes

* 僅使用 Pillow 進行 **讀寫與通道統一**（`convert("RGB")`）；色彩轉換完全以手動係數計算。
* 若課程要求 **YCbCr limited-range（BT.601 studio-range）**，請改用 16–235／16–240 版本公式並另存輸出。
* 若需避免下取整偏差，可在輸出前 `np.rint` 再轉 `uint8`。

---

