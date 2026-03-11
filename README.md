# Neural Rendering Benchmarking

- Support *skimage* PSNR and SSIM metrics.

## Usage

### For Image Benchmarking

- Run the scripts:  
    - Evaluate **single** image pair:  
        ```bash
        python benchmark.py -t image --test PATH-TO-TEST-IMAGE --gt PATH-TO-GT-IMAGE
        ```  
        This will output PSNR and SSIM in terminal. **No result.csv will be generated**.

    - Evaluate image **datasets**:   
        ```bash
        python benchmark.py -t image --test PATH-TO-TEST-FOLDER --gt PATH-TO-GT-FOLDER
        ```  
        - Optional arguments:  
            - `--compare_type`  
                - `1to1`: **Default option**. For same file names in both test and gt folder. 
                - `warps_vs_gt`: For specific naming convention: ggw_3.png (test) <-> colorNoScreenUI_3.png (gt). This will search gt image based on test image.

### For Video Benchmarking

- The code will extract the frames from both of the videos into `test` and `gt` folders, then compair the frames. 
    - Make sure to check the folders before benchmarking.
- Run the scripts:

    Evaluate video dataset:
    ```bash
    python benchmark.py -t video --test PATH-TO-TEST-VIDEO --gt PATH-TO-GT-VIDEO
    ```
- Optional arguments:
    - `--skip`
        - Default is `True`. 
        - Skip the even frames in the video. If you want to compare the whole video, then set it to `False`.

## Output

After benchmarking, it will output a file:
- result.csv
    - Scores of each frame.
    - Average.
    - Standard Variance.
    - Maximum.
    - Minimum.
    - Median.
