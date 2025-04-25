import os
import imageio.v2 as imageio
import numpy as np

# Define source and target paths
busi_raw_root = "Dataset_BUSI_with_GT"
busi_target_root = "data/BUSI_Mixed"

# Create folders
images_dir = os.path.join(busi_target_root, "images")
labels_dir = os.path.join(busi_target_root, "labels")
lists_dir = os.path.join(busi_target_root, "lists")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(lists_dir, exist_ok=True)

image_list = []
counter = 0

# Only use benign and malignant
for category in ["benign", "malignant"]:
    src_folder = os.path.join(busi_raw_root, category)

    for file in os.listdir(src_folder):
        if file.endswith(".png") and "_mask" not in file:
            base_name = os.path.splitext(file)[0]
            img_path = os.path.join(src_folder, file)
            mask_path = os.path.join(src_folder, base_name + "_mask.png")

            save_name = f"{category}_{counter:04d}.png"
            img_dst = os.path.join(images_dir, save_name)
            mask_dst = os.path.join(labels_dir, save_name)

            # Read image
            image = imageio.imread(img_path)

            # Read and process mask
            if os.path.exists(mask_path):
                mask = imageio.imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]  # Take one channel
                mask = (mask > 0).astype(np.uint8) * 255
            else:
                mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

            # Save both
            imageio.imwrite(img_dst, image.astype(np.uint8))
            imageio.imwrite(mask_dst, mask)

            image_list.append(save_name)
            counter += 1

# Save the image list
with open(os.path.join(lists_dir, "images.txt"), "w") as f:
    f.write("\n".join(sorted(image_list)))

print("BUSI Mixed dataset created successfully!")
