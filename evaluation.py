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

# --- CONFIGURATION ---
PRODUCTS_ROOT = Path("products")
CHECKPOINT_PATH = Path("sam2.1_hiera_tiny.pt")
MODEL_CONFIG = "sam2.1/sam2.1_hiera_t.yaml"
SAM2_BASE_PATH = os.path.dirname(sam2.__file__)
OUTPUT_DIR = Path("output")
REPORT_FILE = OUTPUT_DIR / "report.md"

def calculate_iou(box1, box2):
    """Calculates IoU of two boxes: [xmin, xmax, ymin, ymax]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[2], box2[2])
    x_right = min(box1[1], box2[1])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
    box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def get_coords_from_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None
    return [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]

def evaluate_products():
    if not OUTPUT_DIR.exists(): os.makedirs(OUTPUT_DIR)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dir = os.path.join(SAM2_BASE_PATH, "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT_PATH), device=device)

    results_data = []

    # Iterate through product folders
    product_folders = sorted([f for f in PRODUCTS_ROOT.iterdir() if f.is_dir()])
    
    for p_folder in product_folders:
        print(f"--> Analyzing Product: {p_folder.name}")
        all_jps = sorted(list(p_folder.glob("*.jpg")))
        if not all_jps: continue

        # 1. Setup Temp Video Sequence
        temp_dir = tempfile.mkdtemp()
        for i, img_path in enumerate(all_jps):
            shutil.copy(img_path, os.path.join(temp_dir, f"{i:05d}.jpg"))

        # 2. Initialize with first frame
        source_img = all_jps[0]
        source_mask = str(source_img).replace(".jpg", "_1_gt.png")
        seed_coords = get_coords_from_mask(source_mask)
        
        inference_state = predictor.init_state(video_path=temp_dir)
        predictor.reset_state(inference_state)
        
        # Add box prompt (Frame 0)
        predictor.add_new_points_or_box(
            inference_state, frame_idx=0, obj_id=1, 
            box=np.array([seed_coords[0], seed_coords[2], seed_coords[1], seed_coords[3]], dtype=np.float32)
        )

        # 3. Propagate through "video"
        predictions = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            coords = np.where(mask[0])
            if coords[0].size > 0:
                predictions[out_frame_idx] = [int(coords[1].min()), int(coords[1].max()), int(coords[0].min()), int(coords[0].max())]
            else:
                predictions[out_frame_idx] = [0, 0, 0, 0]

        # 4. Compare with GT for all Target Frames (starting from index 1)
        for i in range(1, len(all_jps)):
            target_img_path = all_jps[i]
            gt_mask_path = str(target_img_path).replace(".jpg", "_1_gt.png")
            
            manual_gt = get_coords_from_mask(gt_mask_path)
            pred_coords = predictions.get(i, [0,0,0,0])
            
            iou_score = calculate_iou(pred_coords, manual_gt) if manual_gt else 0.0
            
            results_data.append({
                "product": p_folder.name,
                "image": target_img_path.name,
                "pred": pred_coords,
                "gt": manual_gt,
                "iou": round(iou_score, 4)
            })

        shutil.rmtree(temp_dir)

    # 5. Generate report.md and Print Table
    write_report(results_data)

def write_report(data):
    header = "| Product | Image | Predicted Coords (xmin, xmax, ymin, ymax) | Manual GT Coords | IoU Score |\n"
    separator = "| :--- | :--- | :--- | :--- | :--- |\n"
    
    with open(REPORT_FILE, "w") as f:
        f.write("# SAM 2.1 Zero-Shot Tracking Evaluation Report\n\n")
        f.write(header + separator)
        
        print("\n" + header + separator, end="")
        for entry in data:
            row = f"| {entry['product']} | {entry['image']} | {entry['pred']} | {entry['gt']} | **{entry['iou']}** |\n"
            f.write(row)
            print(row, end="")
            
    print(f"\nReport generated: {REPORT_FILE}")

if __name__ == "__main__":
    evaluate_products()