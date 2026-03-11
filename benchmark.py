import argparse
import os
import os.path as osp
import warnings

import numpy as np
import pandas as pd
import skimage.metrics
import cv2
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

############ Arguments ############
parser = argparse.ArgumentParser()
parser.add_argument("--type", "-t", type=str, choices=["image", "video"], required=True, help="Type of data to evaluate: image folders or videos.")
parser.add_argument("--test", type=str, required=True, help="Path to test image folder or video.")
parser.add_argument("--gt", type=str, required=True, help="Path to GT image folder or video.")
parser.add_argument("--skip", type=str, choices=["true", "false", "True", "False"], default="True", help="Whether to skip every other frame in video evaluation.")
parser.add_argument("--compare_type", type=str, choices=["1to1", "warps_vs_gt"], default="1to1", help="Type of comparison for image pairs.")
parser.add_argument("--crop", type=str, default=None, help="Crop size, format: 'height,width'")
args = parser.parse_args()

############ Helper: extract frames from video ############
def extract_frames_ffmpeg(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Use ffmpeg to extract frames
    cmd = f'ffmpeg -hide_banner -i "{video_path}" -pix_fmt rgb24 -start_number 0 "{osp.join(out_dir, "frame_%d.png")}"'
    print(f"Executing: {cmd}")
    os.system(cmd)
    # ffmpeg starts frame numbering from 1, let's count the files
    return len(os.listdir(out_dir))

def image_diff_visualization(img_gt, img_test, threshold=0.1, ssim_map=None):
    # 1. 計算 RGB difference vector (需轉 float 防止 uint8 overflow)
    # 確保只取前三個通道 (BGR) 進行比較，忽略 Alpha channel
    img_gt_f = img_gt[:,:,:3].astype(np.float32) if img_gt.ndim == 3 else img_gt.astype(np.float32)
    img_test_f = img_test[:,:,:3].astype(np.float32) if img_test.ndim == 3 else img_test.astype(np.float32)
    
    diff_vector = np.abs(img_gt_f - img_test_f)
    
    # 2. 根據每個 pixel 的 RGB 向量的 "長度" 來視覺化 (Euclidean norm)
    # axis=2 表示沿著 channel 方向計算 Norm
    diff_magnitude = np.linalg.norm(diff_vector, axis=2) # Shape: (H, W)
    
    # Rearrange 到 0~1，以該圖中 "最長長度向量" 為 1
    max_val = np.max(diff_magnitude)
    if max_val == 0:
        print("Images are identical.")
        return
    diff_norm = diff_magnitude / max_val

    cv2.imwrite("diff_magnitude.png", (diff_norm * 255).astype(np.uint8))  # 儲存誤差強度圖

    # 儲存一張誤差強度的熱力圖 (額外參考用)
    # diff_heatmap = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # cv2.imwrite("diff_magnitude_heatmap.png", diff_heatmap)

    # 3. 製作 Mask
    mask = diff_norm > threshold  # Boolean Mask
    cv2.imwrite(f"diff_mask_threshold_{threshold}.png", (mask.astype(np.uint8) * 255))  # 儲存 Mask 圖 (白色表示誤差大於閾值的區域)
    
    # 套到 test image 上：將誤差嚴重 (Mask為True) 的區域標記為紅色
    # 為了畫紅色，先確保底圖是 BGR 格式
    vis_overlay = img_test.copy()
    if vis_overlay.ndim == 2: # 如果是灰階，轉 BGR
        vis_overlay = cv2.cvtColor(vis_overlay, cv2.COLOR_GRAY2BGR)
    elif vis_overlay.shape[2] == 4: # 如果有 Alpha，轉 BGR 丟掉 Alpha
        vis_overlay = cv2.cvtColor(vis_overlay, cv2.COLOR_BGRA2BGR)
        
    # 將 Mask 區域塗成紅色 (0, 0, 255)
    vis_overlay[mask] = [0, 0, 255]
    
    cv2.imwrite(f"diff_overlay_threshold_{threshold}.png", vis_overlay)
    print(f"Visualization saved")

    # 4. SSIM Map 視覺化 (Structural Similarity Loss)
    # 相比 RGB diff，SSIM 更能反映「結構破損」或「模糊」的區域，而非單純的顏色誤差。
    try:
        # ssim_map 值域為 [-1, 1]，1 為完全相同。
        # 我們計算 (1 - ssim) 作為 "Loss Map"，數值越大代表結構差異越大。
        # 取各通道平均
        ssim_loss = 1.0 - np.mean(ssim_map, axis=2)
        
        # 視覺化：放大數值以利觀察 (x5)，並轉為熱力圖
        ssim_vis = np.clip(ssim_loss * 255 * 5, 0, 255).astype(np.uint8)
        ssim_heatmap = cv2.applyColorMap(ssim_vis, cv2.COLORMAP_JET)
        
        # cv2.imwrite("diff_ssim_heatmap.png", ssim_heatmap)
        cv2.imwrite("ssim_map_color.png", (ssim_map * 255).astype(np.uint8))  # 儲存原始 SSIM Loss Map (彩色)
        cv2.imwrite("ssim_map_gray.png", (ssim_loss * 255).astype(np.uint8))  # 儲存原始 SSIM Loss Map (灰階)
        # print("SSIM map visualization saved: diff_ssim_heatmap.png")
    
    except Exception as e:
        print(f"Could not generate SSIM map: {e}")

############ Prepare image pairs ############
if args.type == "image":
    if args.test.endswith((".png", ".jpg", ".jpeg")) and args.gt.endswith((".png", ".jpg", ".jpeg")): # For single image comparison 
        img_test_np = cv2.imread(args.test, flags=-1)
        img_gt_np = cv2.imread(args.gt, flags=-1)
        if args.crop is not None:
            H, W = img_test_np.shape[0:2]
            ch, cw = map(int, args.crop.split(','))
            y = (H - ch)
            x = (W - cw) // 2 - 100
            img_test_np = img_test_np[y:y+ch, x:x+cw, :]
            img_gt_np = img_gt_np[y:y+ch, x:x+cw, :]
            cv2.imwrite("cropped_gt.png", img_gt_np)
            cv2.imwrite("cropped_test.png", img_test_np)
        pnsr = skimage.metrics.peak_signal_noise_ratio(image_true=img_gt_np, image_test=img_test_np, data_range=255)
        ssim, ssim_map = skimage.metrics.structural_similarity(im1=img_gt_np, im2=img_test_np, data_range=255, channel_axis=2, full=True)
        print(f"PSNR: {pnsr}, SSIM: {ssim}")

        image_diff_visualization(img_gt_np, img_test_np, ssim_map=ssim_map)
        
        exit(0)
    else: 
        if args.compare_type == "1to1": # For GT and test images with same names
            test_files = sorted([osp.join(args.test, f) for f in os.listdir(args.test) if f.endswith((".png", ".jpg", ".jpeg"))])
            gt_files = sorted([osp.join(args.gt, f) for f in os.listdir(args.gt) if f.endswith((".png", ".jpg", ".jpeg"))])
            n_frames = min(len(test_files), len(gt_files))
            test_files = test_files[:n_frames]
            gt_files = gt_files[:n_frames]
        elif args.compare_type == "warps_vs_gt": # For specific naming convention: ggw_3.png <-> colorNoScreenUI_3.png
            test_files_list = sorted([f for f in os.listdir(args.test) if f.endswith((".png", ".jpg", ".jpeg"))])
            print(f"Found {len(test_files_list)} test files. Attempting to match with GT files based on naming convention...")
            test_files = []
            gt_files = []
            
            for f_test in test_files_list:
                try:
                    # Parse number from ggw_3.png -> extract '3'
                    name_without_ext = osp.splitext(f_test)[0]
                    parts = name_without_ext.split('_')
                    if len(parts) >= 2 and parts[1].isdigit():
                        number = parts[1]
                        f_gt_name = f"colorNoScreenUI_{number}.png"
                        f_gt_path = osp.join(args.gt, f_gt_name)
                        
                        if osp.exists(f_gt_path):
                            test_files.append(osp.join(args.test, f_test))
                            gt_files.append(f_gt_path)
                        else:
                            print(f"Warning: GT file {f_gt_name} not found for {f_test}")
                    else:
                        # Fallback or skip if naming convention doesn't match
                        pass
                except Exception as e:
                    print(f"Error processing {f_test}: {e}")

            if len(test_files) == 0:
                print("No valid image pairs found based on naming convention.")
                exit(1)
elif args.type == "video":
    test_dir = "test"
    gt_dir = "gt"
    # Extract test video frames if not already present
    if not osp.isdir(test_dir) or len(os.listdir(test_dir)) == 0:
        print(f"Extracting frames from {args.test} to {test_dir}")
        n_test = extract_frames_ffmpeg(args.test, test_dir)
    else:
        print("Test frames already extracted. Make sure they correspond to the test video.")
        n_test = len(os.listdir(test_dir))
    # Extract GT video frames if not already present
    if not osp.isdir(gt_dir) or len(os.listdir(gt_dir)) == 0:
        print(f"Extracting frames from {args.gt} to {gt_dir}")
        n_gt = extract_frames_ffmpeg(args.gt, gt_dir)
    else:
        print("GT frames already extracted. Make sure they correspond to the GT video.")
        n_gt = len(os.listdir(gt_dir))
    n_frames = min(n_test, n_gt)
    # ffmpeg outputs frame_0.png, frame_1.png, etc.
    test_files = [osp.join(test_dir, f"frame_{i}.png") for i in range(n_frames)]
    gt_files = [osp.join(gt_dir, f"frame_{i}.png") for i in range(n_frames)]

############ Evaluate ############
results = []
PSNRs = []
SSIMs = []

if args.type == "image":
    for idx, (f_test, f_gt) in enumerate(tqdm(zip(test_files, gt_files), total=len(test_files))):
        img_test_np = cv2.imread(f_test, flags=-1)
        img_gt_np = cv2.imread(f_gt, flags=-1)

        psnr = skimage.metrics.peak_signal_noise_ratio(image_true=img_gt_np, image_test=img_test_np, data_range=255)
        ssim = skimage.metrics.structural_similarity(im1=img_gt_np, im2=img_test_np, data_range=255, channel_axis=2)

        results.append([f_test, psnr, ssim])
        PSNRs.append(round(float(psnr), 5))
        SSIMs.append(round(float(ssim), 5))

elif args.type == "video":
    for idx, (f_test, f_gt) in enumerate(tqdm(zip(test_files, gt_files), total=len(test_files))):
        if (args.skip.lower() == "true") and (idx % 2 == 0): # Skip even video frames (0, 2, 4...)
            continue

        img_test_np = cv2.imread(f_test, flags=-1)
        img_gt_np = cv2.imread(f_gt, flags=-1)

        # # Logic for resized and cropped images
        # if strCategory == 'resized':
        #     
        #
        # elif strCategory == 'cropped':
        # Create directories for cropped frames if they don't exist
        #     os.makedirs("test_crop", exist_ok=True)
        #     os.makedirs("gt_crop", exist_ok=True)

        #     img_test_np = img_test_np[240:, 400:-400, :]
        #     # Assuming ground truth also needs cropping
        #     img_gt_np = img_gt_np[240:, 400:-400, :]

        #     # Save the cropped frames
        #     test_crop_path = osp.join("test_crop", osp.basename(f_test))
        #     gt_crop_path = osp.join("gt_crop", osp.basename(f_gt))
        #     cv2.imwrite(test_crop_path, img_test_np)
        #     cv2.imwrite(gt_crop_path, img_gt_np)

        psnr = skimage.metrics.peak_signal_noise_ratio(image_true=img_gt_np, image_test=img_test_np, data_range=255)
        ssim = skimage.metrics.structural_similarity(im1=img_gt_np, im2=img_test_np, data_range=255, channel_axis=2)

        # Get frame index from filename for accurate reporting
        frame_index = int(osp.basename(f_gt).split('.')[0].split('_')[-1])
        results.append([frame_index, psnr, ssim])
        PSNRs.append(round(float(psnr), 5))
        SSIMs.append(round(float(ssim), 5))

############ Save CSV & Print Average ############
csv_result_file = "result.csv"
df = pd.DataFrame(results, columns=["idx", "psnr", "ssim"])

# Add average row
avg_psnr = np.mean(PSNRs)
avg_ssim = np.mean(SSIMs)
statistical_row = pd.DataFrame({
    "idx": ["avg", "std", "max", "min", "median"],
    "psnr": [
        round(float(avg_psnr), 5),
        round(float(np.std(PSNRs, ddof=0)), 5),
        round(float(np.max(PSNRs)), 5),
        round(float(np.min(PSNRs)), 5),
        round(float(np.median(PSNRs)), 5),
    ],
    "ssim": [
        round(float(avg_ssim), 5),
        round(float(np.std(SSIMs, ddof=0)), 5),
        round(float(np.max(SSIMs)), 5),
        round(float(np.min(SSIMs)), 5),
        round(float(np.median(SSIMs)), 5),
    ],
})

# Round per-frame psnr/ssim to 5 decimal places before appending the summary rows
df[["psnr", "ssim"]] = df[["psnr", "ssim"]].round(5)
df = pd.concat([df, statistical_row], ignore_index=True)
df.to_csv(csv_result_file, index=False)

print(f"\nResults saved to {csv_result_file}")
print(f"Computed average psnr: {avg_psnr:.5f}")
print(f"Computed average ssim: {avg_ssim:.5f}")