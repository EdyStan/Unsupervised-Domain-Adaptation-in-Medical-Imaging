import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import shutil
import gdown
import zipfile
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff


def download_and_extract(url, dest_folder, zip_name):
    # Check if the folder already exists and contains files
    extracted_folder = os.path.join(dest_folder, os.path.splitext(zip_name)[0])  # Remove .zip extension

    # Check if the extracted folder exists and is not empty
    if os.path.exists(extracted_folder) and os.listdir(extracted_folder):
        print(f"Folder '{extracted_folder}' already exists and is not empty. Skipping download.")
        return
    
    os.makedirs(dest_folder, exist_ok=True)

    # Download the file using gdown
    zip_file_path = os.path.join(dest_folder, zip_name)
    gdown.download(url, zip_file_path, quiet=True)
    
    # Check if the file was downloaded and unzip it
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print(f"Unzipped {zip_name} successfully to {dest_folder}")
        
        # Remove the zip file after extraction
        try:
            os.remove(zip_file_path)
            print(f"Removed {zip_name} successfully.")
        except FileNotFoundError:
            print(f"File {zip_file_path} not found for removal.")
    else:
        print(f"File {zip_file_path} not found. Download might have failed.")


def safe_rename(old_path, new_path):
    old_path, new_path = Path(old_path), Path(new_path)
    if old_path.exists():
        # Copy the entire folder
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
        print(f"Copied '{old_path}' to '{new_path}'")

        # Delete the old folder after copying
        shutil.rmtree(old_path)
        print(f"Deleted old folder: {old_path}")
    else:
        print(f"Error: The folder '{old_path}' does not exist.")


def resize_images_by_folder(input_folder, output_folder, max_images=5000, target_size=(256, 256), extension_filter='.png'):
    """
    Resize a subset of images in a folder to the specified dimensions.
    LEAVE USE_MAX_IMAGES TRUE

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to save the resized images.
        no_resized_images (int): Number of images to resize.
        target_size (tuple): Target dimensions as (width, height).
        extension_filter (str): Only resize images with this file extension (e.g., '.png').
        step (int): Step size for selecting images from the sorted list of paths.

    Returns:
        None
    """
    # if os.path.exists(output_folder):
    #     print(f"Skipping resize: {output_folder} already exists.")
    #     return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image paths with the specified extension and sort them
    image_paths = [
        os.path.join(input_folder, filename)
        for filename in sorted(os.listdir(input_folder))
        if filename.lower().endswith(extension_filter)
    ]

    len_paths = len(image_paths)
    if len_paths == 0:
        raise FileNotFoundError("No files were found. Check if the exception filter is correct.")

    # Select images by step to get the desired number of resized images
    if len_paths < max_images:
        print(f'max_images > all images. The number of resized images is {len_paths} instead of {max_images}')
    max_step = len_paths // min(len_paths, max_images)
    selected_paths = image_paths[::max_step]

    # Resize and save the selected images
    for input_path in selected_paths:
        # Read the image
        image = cv2.imread(input_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize the image
        resized_image = cv2.resize(image, target_size)

        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, os.path.basename(input_path))
        cv2.imwrite(output_path, resized_image)

    print("Resizing complete.")


def train_test_split_img_and_mask(all_img_dir, 
                                  all_mask_dir,
                                  train_img_dir, 
                                  test_img_dir,
                                  train_mask_dir,
                                  test_mask_dir,
                                  extension_train_img, 
                                  extension_train_mask,
                                  extension_test_img,
                                  extension_test_mask,
                                  data_used_ratio=0.5, 
                                  split_ratio=0.2):
    """
    Splits images and masks into train and test directories based on the specified ratios.

    Parameters:
    all_img_dir (str): Path to the directory containing all images.
    all_mask_dir (str): Path to the directory containing all masks.
    train_img_dir (str): Output directory for training images.
    test_img_dir (str): Output directory for test images.
    train_mask_dir (str): Output directory for training masks.
    test_mask_dir (str): Output directory for test masks.
    extension_train_img (str): File extension for training images (e.g., ".jpg").
    extension_train_mask (str): File extension for training masks (e.g., "_segmentation.png").
    extension_test_img (str): File extension for test images (e.g., ".jpg").
    extension_test_mask (str): File extension for test masks (e.g., "_segmentation.png").
    data_used_ratio (float): Ratio of data to use (e.g., 0.5 means use 50% of the dataset).
    split_ratio (float): Ratio of test data (e.g., 0.2 means 20% test and 80% train).
    """
    # Ensure output directories exist
    for dir_path in [train_img_dir, test_img_dir, train_mask_dir, test_mask_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Get all image files in the directory
    all_files = [f for f in os.listdir(all_img_dir) if f.endswith(extension_train_img)]
    
    # Shuffle and select a subset of data based on data_used_ratio
    random.shuffle(all_files)
    subset_size = int(len(all_files) * data_used_ratio)
    selected_files = all_files[:subset_size]

    # Split into train and test based on split_ratio
    test_size = int(len(selected_files) * split_ratio)
    test_files = selected_files[:test_size]
    train_files = selected_files[test_size:]

    # Function to copy files
    def copy_files(file_list, img_source_dir, mask_source_dir, img_dest_dir, mask_dest_dir, img_ext, mask_ext):
        for file_name in file_list:
            img_source = os.path.join(img_source_dir, file_name)
            mask_source = os.path.join(mask_source_dir, file_name.replace(img_ext, mask_ext))

            img_dest = os.path.join(img_dest_dir, file_name)
            mask_dest = os.path.join(mask_dest_dir, file_name.replace(img_ext, mask_ext))

            shutil.copy(img_source, img_dest)
            if os.path.exists(mask_source):
                shutil.copy(mask_source, mask_dest)
            else:
                print(f"Warning: No corresponding mask found for {file_name}")

    # Copy train files
    print("Copying training data...")
    copy_files(train_files, all_img_dir, all_mask_dir, train_img_dir, train_mask_dir, extension_train_img, extension_train_mask)

    # Copy test files
    print("Copying test data...")
    copy_files(test_files, all_img_dir, all_mask_dir, test_img_dir, test_mask_dir, extension_test_img, extension_test_mask)

    print("Data split completed.")


def separate_files_respecting_txt(input_dir, output_dir, txt_file):
    """
    Extracts files from a source folder to a destination folder, 
    based on file names listed in a text file.

    Parameters:
    source_folder (str): The path to the source folder containing files.
    destination_folder (str): The path to the destination folder where files should be copied.
    txt_file (str): The path to the text file listing the files to extract.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the text file to get the list of target files
    with open(txt_file, 'r') as file:
        target_files = file.read().splitlines()

    # Loop through each file name and copy it if it exists in the source folder
    for file_name in target_files:
        source_file_path = os.path.join(input_dir, file_name)
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, output_dir)


def mask_to_voc_labels(mask_dir, label_dir, mask_extension=".png", class_id=1):
    """
    Converts masks to Pascal VOC labels.

    Parameters:
        mask_dir (str): Directory containing the mask images.
        label_dir (str): Directory to save VOC label files.
        class_id (int): Class ID for all objects in the mask.
    """
    if os.path.exists(label_dir):
        print(f"Skipping label generation: {label_dir} already exists.")
        return

    os.makedirs(label_dir, exist_ok=True)

    for mask_name in os.listdir(mask_dir):

        # Load mask
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
          print("is none: ", mask_path)
          continue

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = mask.shape[:2]
        voc_labels = []

        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Normalize coordinates
            no_decimals = 5
            x_max = round((x + w) / width, no_decimals)
            y_max = round((y + h) / height, no_decimals)
            x_min = round( x / width, no_decimals)
            y_min = round( y / height, no_decimals)

            # Append VOC label
            voc_labels.append(f"{class_id} {x_min} {y_min} {x_max} {y_max}")

        # Save labels to a file
        label_path = os.path.join(label_dir, mask_name.replace(mask_extension, '.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(voc_labels))  # ex: BraTS20_Training_001_0.txt


def voc_to_yolo_labels(voc_label_dir, yolo_label_dir, img_size=(256, 256), voc_extension=".txt"):
    """
    Converts Pascal VOC labels to YOLO format.

    Parameters:
        voc_label_dir (str): Directory containing VOC label files.
        yolo_label_dir (str): Directory to save YOLO label files.
        img_width (int): Width of the images.
        img_height (int): Height of the images.
    """
    os.makedirs(yolo_label_dir, exist_ok=True)

    for voc_file in os.listdir(voc_label_dir):
        if not voc_file.endswith(voc_extension):
            continue

        voc_path = os.path.join(voc_label_dir, voc_file)
        yolo_path = os.path.join(yolo_label_dir, voc_file.replace(voc_extension, '.txt'))

        yolo_labels = []

        with open(voc_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # Skip malformed lines

            class_id = int(parts[0])  # First value is the class ID
            xmin, ymin, xmax, ymax = map(float, parts[1:])  # Extract bounding box

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            # Format with precision
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save YOLO labels
        with open(yolo_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

    print(f"VOC to YOLO conversion completed. Labels saved in: {yolo_label_dir}")
    

def voc_to_coco_xml(voc_folder, output_folder, img_size=(256, 256)):
    image_width, image_height = img_size
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for txt_file in os.listdir(voc_folder):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(voc_folder, txt_file)
            xml_file = os.path.join(output_folder, txt_file.replace(".txt", ".xml"))
            
            with open(txt_path, "r") as f:
                lines = f.readlines()
            
            # Create XML structure
            annotation = ET.Element("annotation")
            ET.SubElement(annotation, "folder").text = os.path.basename(output_folder)
            ET.SubElement(annotation, "filename").text = txt_file.replace(".txt", ".jpg")
            ET.SubElement(annotation, "path").text = os.path.join(output_folder, txt_file.replace(".txt", ".jpg"))
            
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(image_width)
            ET.SubElement(size, "height").text = str(image_height)
            ET.SubElement(size, "depth").text = "3"
            
            if not lines:
                # If the file is empty, write an XML with no objects
                tree = ET.ElementTree(annotation)
                tree.write(xml_file)
                continue
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id, xmin, ymin, xmax, ymax = map(float, parts[:5])
                
                # Convert VOC format to absolute coordinates
                xmin = int(xmin * image_width)
                ymin = int(ymin * image_height)
                xmax = int(xmax * image_width)
                ymax = int(ymax * image_height)
                
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = str(int(class_id))
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymax").text = str(ymax)
            
            tree = ET.ElementTree(annotation)
            tree.write(xml_file)


def get_segmentation_mask(model, processor, image, input_boxes, multimask_output=False, device="cpu"):
    """
    Perform preprocessing, inference, and post-processing to generate segmentation masks.

    Args:
        model: The MedSAM model.
        processor: The MedSAM processor.
        image: The input image (PIL Image).
        input_boxes: Bounding boxes as a list [x0, y0, x1, y1].
        device: Device for inference ("cpu" or "cuda").
        threshold: Threshold for binary mask generation.

    Returns:
        A list of processed segmentation masks.
    """
    # Preprocess input
    inputs = processor(image, input_boxes=[[input_boxes]], return_tensors="pt").to(device)

    # Model inference
    outputs = model(**inputs, multimask_output=multimask_output)

    # Post-process predictions
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.sigmoid().cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
        mask_threshold=0.5,
        binarize=True
    )

    # Apply threshold for binary masks
    return [mask.cpu().numpy().squeeze() for mask in masks[0][0]]


def compute_dice_scores_with_multiple_bboxes(
    img_dir,
    mask_dir,
    input_coords_dir,
    output_dir,
    model_type=None,
    model=None,
    processor=None,
    device="cpu",
    input_img_sufix=".png",
    input_mask_sufix=".png",
    input_coords_sufix=".txt",
    shift_values_px=[0, 5, 10, 15, 20, 25, 30, 50],
    shift_values_percent=[0.0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
    colors_px = [
        (255, 0, 0),     # Bright Red
        (255, 165, 0),   # Orange
        (255, 255, 0),   # Yellow
        (255, 69, 0),    # Red-Orange
        (255, 105, 180), # Hot Pink
        (220, 20, 60),   # Crimson
    ],
    colors_percent = [
        (0, 0, 255),     # Bright Blue
        (0, 255, 255),   # Cyan
        (0, 255, 0),     # Bright Green
        (135, 206, 250), # Light Sky Blue
        (64, 224, 208),  # Turquoise
        (70, 130, 180),  # Steel Blue
    ],
    skip_empty_coords=False,
    apply_offset=False,
    multimask_output=False,
    print_results=False,
    debug=False
):
    if model_type not in ("sam", "medsam"):
        raise ValueError("Please specify the model type (\"sam\"/\"medsam\")")
    if model_type == "medsam" and processor is None:
        raise ValueError("Please include the processor for the MedSAM model")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    all_metrics = {"px": {shift: {1:[], 2:[], 3:[]} for shift in shift_values_px}, "percent": {shift: {1:[], 2:[], 3:[]} for shift in shift_values_percent}}  # Store metrics for each shift
    evaluated_files_number = 0
    empty_input_files_number = 0
    processing_times = []

    def get_mask_contour_points(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_points = np.vstack(contours).squeeze(axis=1) if contours else np.array([])
        return contour_points  # Get coordinates of all True values in the mask

    for input_coords_file_name in os.listdir(input_coords_dir):
        start_time = time.time()
        image_file_path = os.path.join(img_dir, input_coords_file_name.replace(input_coords_sufix, input_img_sufix))
        gt_mask_file_path = os.path.join(mask_dir, input_coords_file_name.replace(input_coords_sufix, input_mask_sufix))
        input_coords_file_path = os.path.join(input_coords_dir, input_coords_file_name)

        if os.path.getsize(input_coords_file_path) == 0:
            empty_input_files_number += 1
            if skip_empty_coords:
                if debug:
                    print(f"Coordinates file is empty for {image_file_path}. Skipping.")
                continue
            input_coords = np.array([])
        else:
            input_coords = np.genfromtxt(input_coords_file_path, delimiter=None, usecols=(1, 2, 3, 4))

        image = cv2.imread(image_file_path)
        if image is None and debug:
            print("Image not found:", image_file_path)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if input_coords.ndim == 1:
            input_coords = input_coords[np.newaxis, :]

        scale_factor = [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]
        bounding_boxes = (input_coords * scale_factor).astype(int)

        gt_mask = cv2.imread(gt_mask_file_path, cv2.IMREAD_GRAYSCALE).astype(bool)
        if gt_mask is None and debug:
            print("GT mask not found for:", image_file_path)
            continue
        gt_mask_points = get_mask_contour_points(gt_mask)
        gt_mask_flat = gt_mask.flatten()

        if model_type == "sam":
            model.set_image(image)

        evaluated_files_number += 1

        if print_results:
            shifted_bboxes_by_shift = {"px": [], "percent": []}

            os.makedirs(os.path.join(output_dir, input_coords_file_name.replace(input_coords_sufix, "")), exist_ok=True)
            output_image = image.copy()
            for coords in bounding_boxes:
                x_min, y_min, x_max, y_max = coords
                cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            output_image_file_path = os.path.join(output_dir, input_coords_file_name.replace(input_coords_sufix, ""), f"label{input_img_sufix}")
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns (side-by-side)

            # Left: Image with bounding box
            ax[0].imshow(output_image)
            ax[0].set_title("Ground Truth Bounding Box")
            ax[0].axis("off")

            # Right: Image with predicted mask overlayed
            ax[1].imshow(image)
            ax[1].imshow(gt_mask, alpha=0.5, cmap="jet")  # Overlay mask with transparency
            ax[1].set_title("Ground Truth Mask Overlay")
            ax[1].axis("off")

            # Save the combined image
            plt.savefig(output_image_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        # Combined iteration over both pixel and percentage shifts
        for shift_type, shifts in [("px", shift_values_px), ("percent", shift_values_percent)]:
            for shift_idx, shift in enumerate(shifts):
                prev_masks = np.zeros_like(gt_mask, dtype=bool)
                if multimask_output:
                    prev_masks = [prev_masks, prev_masks, prev_masks]

                if print_results:
                    shifted_bbox_coords = []

                for bbox_idx, bounding_box in enumerate(bounding_boxes):
                    bbox_width = bounding_box[2] - bounding_box[0]
                    bbox_height = bounding_box[3] - bounding_box[1]

                    if shift_type == "px":
                        shifted_bbox = bounding_box.copy()
                        shifted_bbox[:2] -= shift
                        shifted_bbox[2:] += shift
                    else:  # shift_type == "percent"
                        shift_x = int(bbox_width * shift)
                        shift_y = int(bbox_height * shift)
                        shifted_bbox = bounding_box.copy()
                        shifted_bbox[:2] -= [shift_x, shift_y]
                        shifted_bbox[2:] += [shift_x, shift_y]

                    # If apply_offset=True, apply random offset to position ground truth randomly inside the shifted bounding box
                    if apply_offset:
                        max_offset_x = int(0.5 * (min(image.shape[1], shifted_bbox[2] - shifted_bbox[0]) - bbox_width))
                        max_offset_y = int(0.5 * (min(image.shape[0], shifted_bbox[3] - shifted_bbox[1]) - bbox_height))

                        offset_xmin = random.randint(-max_offset_x, max_offset_x)
                        offset_ymin = random.randint(-max_offset_y, max_offset_y)
                        offset_xmax = random.randint(-max_offset_x, max_offset_x)
                        offset_ymax = random.randint(-max_offset_y, max_offset_y)

                        shifted_bbox += [offset_xmin, offset_ymin, offset_xmax, offset_ymax]
                    
                    shifted_bbox[2] = max(shifted_bbox[0], shifted_bbox[2])
                    shifted_bbox[3] = max(shifted_bbox[1], shifted_bbox[3])

                    shifted_bbox = np.clip(shifted_bbox, [0, 0, 0, 0], [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

                    if print_results:
                        shifted_bbox_coords.append(shifted_bbox)

                    if model_type == "sam":
                        masks, _, _ = model.predict(box=shifted_bbox, multimask_output=multimask_output)
                    elif model_type == "medsam":
                        masks = get_segmentation_mask(model, processor, image, [shifted_bbox], multimask_output=multimask_output, device=device)

                    masks = [mask.astype(bool) for mask in masks]
                    prev_masks = np.logical_or(prev_masks, masks)

                    # Execute the evaluation only if it's the last bbox
                    if bbox_idx < len(bounding_boxes) - 1:
                        continue

                    # Generalize for the case in which we have 3 predicted masks. Default, we have 1.
                    for mask_idx, mask in enumerate(prev_masks):
                        mask_points = get_mask_contour_points(mask) 
                        mask_flat = mask.flatten()                
                        if mask_points.size == 0 or gt_mask_points.size == 0:
                            dice, precision, recall, iou = 0, 0, 0, 0
                            hausdorff = image.shape[0]
                            if debug:
                                print("Empty points detected. Skipping Hausdorff calculation.")
                                print("Mask points shape:", mask_points.shape)
                                print("GT mask points shape:", gt_mask_points.shape)
                        else:
                            dice = f1_score(gt_mask_flat, mask_flat)
                            precision = precision_score(gt_mask_flat, mask_flat)
                            recall = recall_score(gt_mask_flat, mask_flat)
                            intersection = np.logical_and(gt_mask_flat, mask_flat).sum()
                            union = np.logical_or(gt_mask_flat, mask_flat).sum()
                            iou = intersection / union if union != 0 else 0.0
                            hausdorff = directed_hausdorff(gt_mask_points, mask_points)[0]

                        all_metrics[shift_type][shift][mask_idx+1].append({
                            "Dice": round(dice, 3),
                            "Precision": round(precision, 3),
                            "Recall": round(recall, 3),
                            "IoU": round(iou, 3),
                            "Hausdorff": round(hausdorff)
                        })

                        if debug:
                            print(f"Shift ({shift_type}) {shift}, Mask {mask_idx + 1}, BBox {bbox_idx}: ", {
                                "Dice": round(dice, 3),
                                "Precision": round(precision, 3),
                                "Recall": round(recall, 3),
                                "IoU": round(iou, 3),
                                "Hausdorff": round(hausdorff)
                            })

                        if print_results:
                            output_image = image.copy()
                            # colors = colors_px if shift_type == "px" else colors_percent
                            for coords in shifted_bbox_coords:
                                x_min, y_min, x_max, y_max = coords
                                # color = colors[shift_idx % len(colors)]
                                cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                            multiplier = 1 if shift_type == "px" else 100
                            output_image_file_path = os.path.join(output_dir, input_coords_file_name.replace(input_coords_sufix, ""), f"mask_{mask_idx+1}_and_{shift_type}_{int(shift*multiplier)}{input_img_sufix}")
                            fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns (side-by-side)

                            # Left: Image with bounding box
                            ax[0].imshow(output_image)
                            ax[0].set_title(f"Image with Bounding Box. Shift = {int(shift*multiplier)} {shift_type}.")
                            ax[0].axis("off")

                            # Right: Image with predicted mask overlayed
                            ax[1].imshow(image)
                            ax[1].imshow(mask, alpha=0.5, cmap="jet")  # Overlay mask with transparency
                            ax[1].set_title(f"Predicted Mask Overlay. Dice Score = {round(dice, 3)}")
                            ax[1].axis("off")

                            # Save the combined image
                            plt.savefig(output_image_file_path, bbox_inches='tight', pad_inches=0)
                            plt.close()


        processing_time = time.time() - start_time 
        processing_times.append(processing_time)

    statistics = {}
    for shift_type, shifts in all_metrics.items():
        statistics[shift_type] = {}
        for shift, masks in shifts.items():
            mask_stats = {}
            for mask_index, mask_metrics in masks.items():
                mask_stats[f"Mask {mask_index}"] = {
                    "Dice Mean": np.mean([m["Dice"] for m in mask_metrics]),
                    "Precision Mean": np.mean([m["Precision"] for m in mask_metrics]),
                    "Recall Mean": np.mean([m["Recall"] for m in mask_metrics]),
                    "IoU Mean": np.mean([m["IoU"] for m in mask_metrics]),
                    "Hausdorff Mean": np.mean([m["Hausdorff"] for m in mask_metrics])
                }
            statistics[shift_type][shift] = mask_stats

    statistics["Mean Processing Time per Image"] = np.round(np.mean(processing_times), 3)
    statistics["Empty Input Files"] = empty_input_files_number
    statistics["Evaluated Files"] = evaluated_files_number
    statistics["Skipped Empty Files"] = skip_empty_coords

    return statistics


def get_results(statistics, destination_folder, multimask_output=False):
    """
    Extract and save relevant data from statistics to a results file and separate images.

    Parameters:
        statistics (dict): Dictionary containing Dice score statistics for each shift and mask.
        destination_folder (str): Path to the folder where results and images will be saved.

    Returns:
        None
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Define the output file path
    output_file = os.path.join(destination_folder, "results.txt")

    # Separate non-shift metadata entries
    stats = statistics.copy()
    metadata_keys = {"Empty Input Files", "Evaluated Files", "Skipped Empty Files", "Mean Processing Time per Image"}
    metadata = {key: stats.pop(key) for key in list(stats.keys()) if key in metadata_keys}

    with open(output_file, 'w') as file:
        # Iterate through 'px' and 'percent' keys separately
        for key in ['px', 'percent']:
            if key not in stats:
                file.write(f"Warning: Missing '{key}' data in statistics.\n")
                continue

            shift_data = stats[key]

            # Flatten nested statistics for DataFrame creation
            try:
                data = {
                    (shift, mask): metrics for shift, masks in shift_data.items() for mask, metrics in masks.items()
                }
                df = pd.DataFrame.from_dict(data, orient="index")
                df.index.names = ["Shift", "Mask"]
            except Exception as e:
                file.write(f"Error converting '{key}' statistics to DataFrame: {e}\n")
                continue

            if multimask_output:
                # Write the results table
                file.write(f"\nResults Table for '{key}':\n")
                file.write(df.round(2).to_string() + "\n")

            # Create a separate table with the mean statistics for all masks per shift
            try:
                df_mean_stats = df.groupby("Shift").mean(numeric_only=True)
                file.write(f"\nMean Statistics for All Masks ({key}):\n")
                file.write(df_mean_stats.round(2).to_string() + "\n")
            except Exception as e:
                file.write(f"Error calculating mean statistics for '{key}': {e}\n")
                continue

            # Determine the best shift and mask based on the highest mean Dice score
            if 'Dice Mean' in df.columns:
                best_result = df[df["Dice Mean"] == df["Dice Mean"].max()]
                if not best_result.empty:
                    best_shift, best_mask = best_result.index[0]
                    best_score = best_result["Dice Mean"].values[0]
                    file.write(f"\nConclusion ({key}): The best shift is {best_shift} for {best_mask} with the highest mean Dice score of {best_score:.2f}.\n")
                else:
                    file.write(f"\nNo valid results found in the DataFrame for '{key}'.\n")
            else:
                file.write(f"Error: The 'Dice Mean' column is missing in the '{key}' statistics.\n")
                continue

            # Plot the trend of the mean Dice score by shift for each mask
            try:
                # Save plot for Dice Score Trend
                if multimask_output:
                    df_unstacked = df["Dice Mean"].unstack(level="Mask")
                    title = f"Dice Score Trend by Shift and Mask ({key.capitalize()})"
                    plot_path_trend = os.path.join(destination_folder, f'{key}_dice_score_trend.png')
                    df_unstacked.plot(kind='line', title=title, ylabel='Mean Dice Score', xlabel='Shift')
                    plt.grid(True)
                    plt.savefig(plot_path_trend)
                    plt.close()

                # Save plot for Mean Dice Score
                plot_path_mean = os.path.join(destination_folder, f'{key}_mean_dice_score.png')
                df_mean_stats["Dice Mean"].plot(kind='line', title=f"Mean Dice Score by Shift ({key.capitalize()})", ylabel='Mean Dice Score', xlabel='Shift', color='red')
                plt.grid(True)
                plt.savefig(plot_path_mean)
                plt.close()

            except KeyError as e:
                file.write(f"Error plotting data for '{key}': {e}\n")

        # Write metadata if available
        if metadata:
            file.write("\nMetadata:\n")
            for key, value in metadata.items():
                file.write(f"{key}: {value}\n")