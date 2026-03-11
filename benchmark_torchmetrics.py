import argparse
import os
import os.path as osp
import warnings

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############ Arguments ############
parser = argparse.ArgumentParser()
parser.add_argument("--type", "-t", type=str, choices=["image", "video"], required=True)
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--gt", type=str, required=True)
parser.add_argument("--skip_even", type=str, choices=["true", "false", "True", "False"], default="false")
args = parser.parse_args()

############ Helper: extract frames from video ############
def extract_frames_ffmpeg(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = f'ffmpeg -i "{video_path}" -pix_fmt rgb24 -start_number 0 "{osp.join(out_dir, "frame_%d.png")}"'
    print(f"Executing: {cmd}")
    os.system(cmd)
    return len(os.listdir(out_dir))

############ Prepare image pairs ############
if args.type == "video":
    test_dir = "test_frames"
    gt_dir = "gt_frames"
    if not osp.isdir(test_dir) or len(os.listdir(test_dir)) == 0:
        print(f"Extracting frames from {args.test} to {test_dir}")
        n_test = extract_frames_ffmpeg(args.test, test_dir)
    else:
        print("Test frames already extracted.")
        n_test = len(os.listdir(test_dir))
    if not osp.isdir(gt_dir) or len(os.listdir(gt_dir)) == 0:
        print(f"Extracting frames from {args.gt} to {gt_dir}")
        n_gt = extract_frames_ffmpeg(args.gt, gt_dir)
    else:
        print("GT frames already extracted.")
        n_gt = len(os.listdir(gt_dir))
    n_frames = min(n_test, n_gt)
    test_files = [osp.join(test_dir, f"frame_{i}.png") for i in range(n_frames)]
    gt_files = [osp.join(gt_dir, f"frame_{i}.png") for i in range(n_frames)]
else:
    test_files = sorted([osp.join(args.test, f) for f in os.listdir(args.test) if f.endswith((".png", ".jpg", ".jpeg"))])
    gt_files = sorted([osp.join(args.gt, f) for f in os.listdir(args.gt) if f.endswith((".png", ".jpg", ".jpeg"))])
    n_frames = min(len(test_files), len(gt_files))
    test_files = test_files[:n_frames]
    gt_files = gt_files[:n_frames]

############ Initialize Metrics ############
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

############ Evaluate ############
results = []

# for strCategory in ['resized', 'cropped']: # Uncomment to use
for idx, (f_test, f_gt) in enumerate(tqdm(zip(test_files, gt_files), total=len(test_files))):
    if (args.skip_even.lower() == "true") and (idx % 2 != 0):
        continue

    # Load images as tensors
    img_test = TF.to_tensor(Image.open(f_test)).unsqueeze(0).to(device)
    img_gt = TF.to_tensor(Image.open(f_gt)).unsqueeze(0).to(device)

    # # Logic for resized and cropped
    # if strCategory == 'resized':
    #     img_test = TF.resize(img_test, (1080, 2048), interpolation=TF.InterpolationMode.BICUBIC)
    #     img_gt = TF.resize(img_gt, (1080, 2048), interpolation=TF.InterpolationMode.BICUBIC)
    #
    # elif strCategory == 'cropped':
    #     img_test = TF.crop(img_test, 540, 1024, 1080, 2048)
    #     img_gt = TF.crop(img_gt, 540, 1024, 1080, 2048)

    # Calculate metrics
    psnr = psnr_metric(img_test, img_gt).item()
    ssim = ssim_metric(img_test, img_gt).item()
    # LPIPS expects input in range [-1, 1]
    lpips = lpips_metric(img_test * 2 - 1, img_gt * 2 - 1).item()

    frame_index = int(osp.basename(f_gt).split('.')[0].split('_')[-1])
    results.append([frame_index, psnr, ssim, lpips])

############ Save CSV & Print Average ############
csv_result_file = "result_torchmetrics.csv"
df = pd.DataFrame(results, columns=["idx", "psnr", "ssim", "lpips"])

# Calculate averages
avg_psnr = df["psnr"].mean()
avg_ssim = df["ssim"].mean()
avg_lpips = df["lpips"].mean()

# Add average row
avg_row = pd.DataFrame([["avg", avg_psnr, avg_ssim, avg_lpips]], columns=df.columns)
df = pd.concat([df, avg_row], ignore_index=True)
df.to_csv(csv_result_file, index=False)

print(f"\nResults saved to {csv_result_file}")
print(f"Computed average psnr: {avg_psnr}")
print(f"Computed average ssim: {avg_ssim}")
print(f"Computed average lpips: {avg_lpips}")

# ========== Find worst 5 frames for each metric ==========
def make_title(name):
    name = f" {name} "
    return name.center(30, "=")

metrics_to_check = ["psnr", "ssim", "lpips"]
output_lines = []
for metric_name in metrics_to_check:
    # Exclude the 'avg' row from sorting
    df_numeric = df[df['idx'] != 'avg'].copy()
    df_numeric['idx'] = pd.to_numeric(df_numeric['idx'])
    
    ascending = True
    if metric_name == "lpips":
        ascending = False # Higher LPIPS is worse

    worst = df_numeric.sort_values(by=metric_name, ascending=ascending).head(5)
    output_lines.append(make_title(metric_name))
    for i, row in enumerate(worst.itertuples(), 1):
        output_lines.append(f"{i}. frame_{int(row.idx)}")
    output_lines.append("")

output_str = "\n".join(output_lines)
print(f"\n{output_str}")

# 輸出到 txt 檔
with open("worst_frames_torchmetrics.txt", "w") as f:
    f.write(output_str)
