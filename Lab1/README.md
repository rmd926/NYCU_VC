# Homework #1 – Color Transform

> Author: 312553024 蘇柏叡 
> Date: 2025/09/15

## Requirement

- Please represent “lena.png” in terms of RGB, YUV, and YCbCr.
    1. RGB -> YUV:
    <img width="334" height="66" alt="image" src="https://github.com/user-attachments/assets/2785259a-10ea-4a91-9e5f-cfcc71e9fc01" />

    2. RGB -> YCbCr:
    <img width="842" height="172" alt="image" src="https://github.com/user-attachments/assets/80f0f24a-47cb-43e7-8c85-953fb55a9cf8" />

- In any programming language you are comfortable with (C/C++/Python/MATLAB).
- Output 8 grayscale images representing R, G, B, Y, U, V, Cb, and Cr, respectively.
- **Do not** use any ready-made functions to transform the color.
- You are allowed to use image reading/writing APIs.
- Deadline: 2024/09/30 13:19.
- Compressed as a single ZIP file.
- Required files :
    1. **VC_HW1_[student_id].pdf**: Report PDF
    2. **VC_HW1_[student_id].zip**: Source code (C/C++/Python/MATLAB) with a **README** file instructing the TAs on how to run your code.

## How to run?

1. Move into this folder
    
    ```bash
    cd ./VC_HW1_312553024/
    ```
    
2. Create this conda environment
    
    ```bash
    conda env create -f environment.yml
    ```
    
3. Activate this conda environment
    
    ```bash
    conda activate 1131-video-compression-HW1
    ```
    
4. Run this code
    
    ```bash
    python src/color_transform.py
    ```
    
5. The images representing R, G, B, Y, U, V, Cb, and Cr will be saved in the `image/` folder
