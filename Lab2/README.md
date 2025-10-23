# Homework #2 – 2D-DCT

> Author: 313553024 蘇柏叡  
> Date: 2025/10/24

## Requirement

- 2D-DCT
    - Implement 2D-DCT to transform “lena.png” to DCT coefficients (visualize in log domain).
        - Convert the input image to grayscale first.
        - Visualize the coefficients in the log domain. Feel free to scale and clip the coefficients for visualization.
    - Implement 2D-IDCT to reconstruct the image.
    - Evaluate the PSNR.
- Two 1D-DCT
    - Implement a fast algorithm by two 1D-DCT to transform “lena.png” to DCT coefficients.
- Compare the runtime between 2D-DCT and two 1D-DCT.
- Do **not** use any functions for DCT and IDCT, e.g., cv2.dct
    - Although, you can still use these functions to validate your output.
- Deadline: 2025/10/27 1:19 PM
- Upload to E3 with required files:
    - **VC_HW2_[student_id].pdf**: Report PDF
    - **VC_HW2_[student_id].zip**: Zipped source code (C/C++/Python/MATLAB) and a **README** file

---

## Environment

- Python 3.8+
- numpy
- Pillow
- matplotlib

Install via pip:
```bash
pip install -r requirements.txt

```

---

## How to Run

```bash
python main.py
```
## Output

- The DCT coefficients in the log domain and the reconstructed images will be saved in the `output/` folder
