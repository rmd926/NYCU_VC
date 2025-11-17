# Homework #4 – Entropy Coding

> Author: 313553024 蘇柏叡  
> Date: 2025/11/17

---

Opened: Sunday, 9 November 2025, 12:00 AM
Due: Monday, 24 November 2025, 1:19 PM

---

## Requirement

* **Run Length Encoding and Run Length Decoding**

  * 8×8 block-based DCT coefficients of **“lena.png”**
  * Quantize the coefficients with the **two quantization tables**（如題目所給 Q1、Q2）。
  * Use a **raster scan** to visit all 8×8 blocks in the image.
  * Do the **run length encoding (RLE)** by using a **zigzag scan** to visit all pixels in one 8×8 block.
  * Do the **run length decoding** and **IDCT** to recover the image.
  * Compare the **encoded image sizes** with the two quantization tables（可同時比較 PSNR）。

* **Deadline:** 2024/11/24 1:19 PM

* **Upload to E3 with required files:**

  * `VC_HW4_[student_id].pdf` – Report PDF
  * `VC_HW4_[student_id].zip` – Zipped source code (C/C++/Python/MATLAB) and a **README** file

---

## Environment

* Python 3.8+
* numpy
* opencv-python
* matplotlib


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

*影像與編碼輸出資料夾結構（Output directory layout）：*

```text
decoded_image/
├─ original.png      # 原始灰階 Lena 影像
├─ recon_q1.png      # 使用 Quantization table 1 重建的影像
├─ recon_q2.png      # 使用 Quantization table 2 重建的影像
└─ comparison.png    # 原圖與兩種量化重建影像之對照圖

encoded_image/
├─ qtable1.pkl       # 使用 Quantization table 1 的 RLE bitstream
└─ qtable2.pkl       # 使用 Quantization table 2 的 RLE bitstream
```

---

## Notes

* DCT／IDCT 以 **8×8 區塊**為單位實作；量化採題目給定的兩組 8×8 quantization table。
* Zigzag 掃描後，RLE 以 `(0, count)` 表示連續零值，以 `(value, 1)` 記錄非零係數。
* 編碼檔案大小以 `qtable1.pkl`、`qtable2.pkl` 的檔案大小作近似比較，並可額外計算 PSNR 作為畫質指標。

Ref: [https://www.youtube.com/watch?v=Q2aEzeMDHMA](https://www.youtube.com/watch?v=Q2aEzeMDHMA)
Ref: [https://q-viper.github.io/2021/05/24/coding-run-length-encoding-in-python/](https://q-viper.github.io/2021/05/24/coding-run-length-encoding-in-python/)
