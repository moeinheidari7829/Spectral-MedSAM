import os
import nibabel as nib
import numpy as np
import imageio.v2 as imageio

# Define source paths
root_dir = "US_data/US_volunteer_dataset/ground_truth_data"
data_src = os.path.join(root_dir, "US")
label_src = os.path.join(root_dir, "US_thyroid_label")

# Define target path
target_root = "data/ThySegMultiLabel"

# Ensure target root exists
os.makedirs(target_root, exist_ok=True)

# Get list of data files
data_files = [f for f in os.listdir(data_src) if f.endswith(".nii")]

# Process each data file
for data_file in data_files:
    base_name = data_file.replace("_US.nii", "")  # Remove '_US' to match label name

    # Create subfolder for this sweep
    sweep_folder = os.path.join(target_root, base_name)
    os.makedirs(os.path.join(sweep_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(sweep_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(sweep_folder, "lists"), exist_ok=True)

    # Load .nii files
    data_nifti = nib.load(os.path.join(data_src, data_file)).get_fdata()
    label_file = base_name + ".nii"
    label_path = os.path.join(label_src, label_file)

    if not os.path.exists(label_path):
        print(f"Label file not found for {base_name}, skipping...")
        continue

    label_nifti = nib.load(label_path).get_fdata()

    # Find all unique labels across the volume (excluding background 0)
    unique_labels = np.unique(label_nifti)
    label_ids = unique_labels[unique_labels != 0].astype(int)
    num_labels = len(label_ids)

    # Create consistent grayscale mapping
    fixed_label_map = {
        label_id: int((i + 1) * (255 // (num_labels + 1)))  # Avoid 0
        for i, label_id in enumerate(label_ids)
    }

    print(f"{base_name} - Label mapping:", fixed_label_map)

    # Save each slice as PNG
    num_slices = data_nifti.shape[2]
    image_list = []

    for i in range(num_slices):
        img_filename = f"{base_name}_{i:04d}.png"

        # Normalize image and save
        img_slice = data_nifti[:, :, i]
        img_norm = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice)) * 255
        imageio.imwrite(os.path.join(sweep_folder, "images", img_filename), img_norm.astype(np.uint8))

        # Map labels to grayscale using fixed mapping
        label_slice = label_nifti[:, :, i]
        label_mapped = np.zeros_like(label_slice, dtype=np.uint8)

        for label_val, gray_val in fixed_label_map.items():
            label_mapped[label_slice == label_val] = gray_val

        imageio.imwrite(os.path.join(sweep_folder, "labels", img_filename), label_mapped)

        # Add to image list
        image_list.append(img_filename)

    # Write images.txt file
    with open(os.path.join(sweep_folder, "lists", "images.txt"), "w") as f:
        f.write("\n".join(image_list))

print("Data restructuring complete!")
