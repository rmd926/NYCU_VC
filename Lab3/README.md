# Homework #3 – Motion Estimation & Compensation

> Author: 313553024 蘇柏叡  
> Date: 2025/11/03

## Requirement

Motion estimation (ME)
Block size: 8x8
Search range: [+-8]
Full search block matching algorithm
Integer precision
Motion compensation (MC)
Save the reconstructed frame (after MC) and the residual
Search range:
Compare the results (in PSNR and runtime) with different search ranges ([+-8], [+-16], [+-32]).
Three-step search
Compare the results (in PSNR and runtime) with the Full search algorithm.
Deadline: 2023/10/28 1:19 PM 
Upload to E3 with required files :
VC_HW3_[student_id].pdf: Report PDF
VC_HW3_[student_id].zip: Zipped source code (C/C++/Python/MATLAB) and a README file

---

## Environment

- Python 3.8+
- numpy
- opencv-python
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

Output Files:
output/reconstructed: 放置重建後輸出的影像
output/residual: 放置殘差圖
output/motion_vector: 放置motion_vector圖
