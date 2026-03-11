import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def calculate_psnr_with_mask(original_image, processed_image, mask, data_range=255):
    """
    Calculates PSNR for masked regions of images.

    Args:
        original_image (np.ndarray): The original, ground truth image.
        processed_image (np.ndarray): The processed or noisy image.
        mask (np.ndarray): A binary mask where 1 indicates regions to include
                           in the PSNR calculation and 0 indicates regions to exclude.
        data_range (int or float): The maximum possible pixel value of the images.
                                   Defaults to 255 for 8-bit images.

    Returns:
        float: The PSNR value for the masked region.
    """
    # Apply the mask to both images
    masked_original = original_image * mask
    masked_processed = processed_image * mask

    # Calculate PSNR using skimage's function
    psnr_value = peak_signal_noise_ratio(masked_original, masked_processed, data_range=data_range)
    return psnr_value

# Example usage:
# Assuming original_img, processed_img, and mask_img are NumPy arrays
# representing your images and mask.
# original_img = ...
# processed_img = ...
# mask_img = ... # Ensure mask_img is a binary mask (0s and 1s)

# psnr_result = calculate_psnr_with_mask(original_img, processed_img, mask_img)
# print(f"PSNR with mask: {psnr_result}")