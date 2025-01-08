import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import argparse


# Define the augmentation pipeline


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    A.RandomCrop(height=24, width=24, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.ChannelShuffle(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToGray(p=0.1),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=1, brightness_coefficient=0.7, p=0.2),
    ToTensorV2()
])
# Function to apply augmentations and save images
def augment_and_save(image_path, output_dir):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.uint8)  # Ensure the image is of type uint8
    augmented = transform(image=image)['image']
    output_file = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_file, augmented.permute(1, 2, 0).numpy())

def main(dataset_path,output_path):
    # Iterate through the dataset and apply augmentations
    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                augment_and_save(file_path, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Augment images in a dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("output_path", type=str, help="Path to the output directory.")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    main(dataset_path,output_path)