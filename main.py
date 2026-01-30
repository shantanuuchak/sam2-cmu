import torch
import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

import sam2
from sam2.build_sam import build_sam2_video_predictor

# Getting current directory
cwd = Path.cwd()

def get_path(p):
    return str(cwd / p)

# Path configuration
SOURCE_IMAGE_PATH = get_path("source/can_chowder_000001.jpg")
SOURCE_MASK_PATH  = get_path("source/can_chowder_000001_1_gt.png")
TARGET_IMAGE_PATH = get_path("source/can_chowder_000004.jpg")

CHECKPOINT_PATH = get_path("sam2.1_hiera_tiny.pt")
MODEL_CONFIG = "sam2.1/sam2.1_hiera_t.yaml"
SAM2_BASE_PATH = os.path.dirname(sam2.__file__)
OUTPUT_DIR = get_path("output")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

def show_mask(mask, ax, obj_id=None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def process_img_png_mask(imgpath, maskpath, visualize=False):
    mask = np.array(Image.open(maskpath).convert("L"))
    ys, xs = np.where(mask > 0)
    
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"No object found in mask: {maskpath}")
        
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    if visualize:
        img = Image.open(imgpath)
        fig, ax = plt.subplots()
        ax.set_title("Source Image: Ground Truth Box")
        ax.imshow(img)
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.show()
    return [xmin, xmax, ymin, ymax]

def track_item_boxes(imgpath1, imgpath2, box_list, visualize=True, output_dir="output"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup temp directory and copy files
    temp_dir = tempfile.mkdtemp()
    shutil.copy(imgpath1, os.path.join(temp_dir, "00000.jpg"))
    shutil.copy(imgpath2, os.path.join(temp_dir, "00001.jpg"))

    # 2. Initialize Predictor
    config_dir = os.path.join(SAM2_BASE_PATH, "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        predictor_vid = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT_PATH, device=device)

    inference_state = predictor_vid.init_state(video_path=temp_dir)
    predictor_vid.reset_state(inference_state)

    # 3. Add original boxes to Frame 0
    for (coords, obj_id) in box_list:
        [xmin, xmax, ymin, ymax] = coords
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        predictor_vid.add_new_points_or_box(inference_state, frame_idx=0, obj_id=obj_id, box=box)

    # 4. Propagate Tracking
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor_vid.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # 5. Visualization and Saving Visual Plot
    if visualize:
        plt.figure(figsize=(12, 8))
        plt.title("Target Image: Predicted Mask & Bounding Box")
        target_img = Image.open(os.path.join(temp_dir, "00001.jpg"))
        plt.imshow(target_img)
        
        # Frame 1 is the target frame
        for out_obj_id, out_mask in video_segments[1].items():
            # Show transparent mask
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            
            # Find coordinates for the box from the mask pixels
            mask_coords = np.where(out_mask[0])
            if mask_coords[0].size > 0:
                y_min, y_max = np.min(mask_coords[0]), np.max(mask_coords[0])
                x_min, x_max = np.min(mask_coords[1]), np.max(mask_coords[1])
                
                # Draw the Lime Box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                         linewidth=3, edgecolor='lime', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(x_min, y_min - 10, f"OBJ {out_obj_id}", color='lime', weight='bold')
        
        # Save the plot with the box and mask
        visual_path = os.path.join(output_dir, "tracked_visual_result.png")
        plt.axis('off')
        plt.savefig(visual_path, bbox_inches='tight', pad_inches=0)
        print(f"Visual result (with box) saved to: {visual_path}")
        plt.show()

    shutil.rmtree(temp_dir)
    return video_segments


# MARK: TEST
import glob

def process_all_products(products_root):
    # 1. Get all product subdirectories
    product_dirs = [d for d in Path(products_root).iterdir() if d.is_dir()]
    
    overall_results = {}

    for p_dir in product_dirs:
        product_name = p_dir.name
        print(f"\nProcessing product: {product_name}")
        
        # 2. Get all images and SORT them to ensure 000001 is first
        all_images = sorted(list(p_dir.glob("*.jpg")))
        if not all_images:
            continue
            
        source_img = str(all_images[0])
        # Derive mask name from source image name (pattern: image_name_1_gt.png)
        source_mask = source_img.replace(".jpg", "_1_gt.png")
        
        if not os.path.exists(source_mask):
            print(f"Skipping {product_name}: Source mask not found.")
            continue

        # 3. Initialize the model with the FIRST frame only
        # We send these to SAM2 to create the 'embedding'
        coords_seed = process_img_png_mask(source_img, source_mask, visualize=False)
        
        # 4. Prepare the rest of the images for tracking
        target_images = all_images[1:] 
        
        # Note: In a real video predictor, we'd feed the whole folder to init_state.
        # Since these are discrete files, we can copy them to a temp folder 
        # in the correct sequence (00000, 00001, 00002...)
        
        # 5. Loop through targets and evaluate
        for t_img_path in target_images:
            t_img_str = str(t_img_path)
            t_mask_gt_path = t_img_str.replace(".jpg", "_1_gt.png")
            
            # Prediction from SAM2 (zero-shot)
            # ... model logic here ...
            
            # Manual Box from GT for evaluation
            if os.path.exists(t_mask_gt_path):
                gt_coords = process_img_png_mask(t_img_str, t_mask_gt_path, visualize=False)
                # Now compare gt_coords vs predicted_coords
                # Calculate IoU

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_pipeline():
    create_if_not_exists(OUTPUT_DIR)
    
    # 1. Extract box from source
    print("Extracting bounding box from source mask...")
    coords = process_img_png_mask(SOURCE_IMAGE_PATH, SOURCE_MASK_PATH, visualize=False)
    
    # 2. Run tracking and save the visual plot
    print("Running SAM 2.1 tracking...")
    results = track_item_boxes(SOURCE_IMAGE_PATH, TARGET_IMAGE_PATH, [(coords, 1)], 
                               visualize=True, output_dir=OUTPUT_DIR)
    
    # 3. Save raw binary mask for later analysis
    predicted_mask = results[1][1]
    mask_img = (predicted_mask[0] * 255).astype(np.uint8)
    binary_save_path = os.path.join(OUTPUT_DIR, "predicted_mask_binary.png")
    Image.fromarray(mask_img).save(binary_save_path)
    
    print(f"Raw binary mask saved to: {binary_save_path}")
    print("Done.")

if __name__ == "__main__":
    run_pipeline()