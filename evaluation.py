import torch
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

import sam2
from sam2.build_sam import build_sam2_video_predictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Basic setup
PRODUCTS_ROOT = Path("products")
CHECKPOINT_PATH = Path("sam2.1_hiera_tiny.pt")
MODEL_CONFIG = "sam2.1/sam2.1_hiera_t.yaml"
SAM2_BASE_PATH = os.path.dirname(sam2.__file__)
OUTPUT_DIR = Path("output")
REPORT_FILE_MD = OUTPUT_DIR / "report.md"
REPORT_FILE_TXT = OUTPUT_DIR / "report.txt"

def calculate_iou(box1, box2):
    """Calculates IoU of two boxes: [xmin, xmax, ymin, ymax]"""
    if not box1 or not box2: return 0.0
    x_left = max(box1[0], box2[0])
    y_top = max(box1[2], box2[2])
    x_right = min(box1[1], box2[1])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
    box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
    
    return intersection_area / float(box1_area + box2_area - intersection_area)

def get_coords_from_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]

def write_text_report(data, mAP_scores):
    header = f"{'Product':<15} | {'Image':<25} | {'IoU Score':<10}\n"
    separator = "-" * 55 + "\n"
    
    with open(REPORT_FILE_TXT, "w") as f:
        f.write("=== SAM 2.1 Full Evaluation Report ===\n\n")
        f.write("--- mAP Summary ---\n")
        for prod, score in mAP_scores.items():
            f.write(f"{prod}: {score:.4f}\n")
        f.write("\n--- Per-Frame Details ---\n")
        f.write(header)
        f.write(separator)
        
        for entry in data:
            f.write(f"{entry['product']:<15} | {entry['image']:<25} | {entry['iou']:<10.4f}\n")
    
    print(f"\n[SUCCESS] Detailed report saved to: {REPORT_FILE_TXT}")

def evaluate_products():
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
    
    # Clear old reports
    if REPORT_FILE_MD.exists(): os.remove(REPORT_FILE_MD)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dir = os.path.join(SAM2_BASE_PATH, "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT_PATH), device=device)

    product_folders = sorted([f for f in PRODUCTS_ROOT.iterdir() if f.is_dir()])
    
    all_results_data = []
    final_mAP_scores = {}

    for p_folder in product_folders:
        print(f"\n--> Analyzing Product: {p_folder.name}")
        all_jps = sorted(list(p_folder.glob("*.jpg")))
        if not all_jps:
            continue

        temp_dir = tempfile.mkdtemp()
        for i, img_path in enumerate(all_jps):
            shutil.copy(img_path, os.path.join(temp_dir, f"{i:05d}.jpg"))

        source_mask = str(all_jps[0]).replace(".jpg", "_1_gt.png")
        seed_coords = get_coords_from_mask(source_mask)
        if seed_coords is None:
            print("No seed mask, skipping")
            shutil.rmtree(temp_dir)
            continue
        
        inference_state = predictor.init_state(video_path=temp_dir)
        predictor.reset_state(inference_state)
        predictor.add_new_points_or_box(
            inference_state, frame_idx=0, obj_id=1, 
            box=np.array([seed_coords[0], seed_coords[2], seed_coords[1], seed_coords[3]], dtype=np.float32)
        )

        predictions = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            coords = np.where(mask[0])
            if coords[0].size > 0:
                predictions[out_frame_idx] = [int(coords[1].min()), int(coords[1].max()), int(coords[0].min()), int(coords[0].max())]
            else:
                predictions[out_frame_idx] = [0, 0, 0, 0]

        # Now collect for coco eval, only targets (from 1)
        gt_dict = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": p_folder.name}]
        }
        dt_list = []
        img_id = 1
        ann_id = 1

        for i in range(1, len(all_jps)):
            target_img_path = all_jps[i]
            gt_mask_path = str(target_img_path).replace(".jpg", "_1_gt.png")
            if not os.path.exists(gt_mask_path):
                continue
            
            img = Image.open(target_img_path)
            width, height = img.size

            gt_dict["images"].append({
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": str(target_img_path)
            })

            gt_coords = get_coords_from_mask(gt_mask_path)
            if gt_coords:
                gt_bbox = [gt_coords[0], gt_coords[2], gt_coords[1] - gt_coords[0], gt_coords[3] - gt_coords[2]]
                gt_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": gt_bbox,
                    "area": gt_bbox[2] * gt_bbox[3],
                    "iscrowd": 0
                })
                ann_id += 1

            pred_coords = predictions.get(i, [0, 0, 0, 0])
            if pred_coords != [0, 0, 0, 0]:
                pred_bbox = [pred_coords[0], pred_coords[2], pred_coords[1] - pred_coords[0], pred_coords[3] - pred_coords[2]]
                dt_list.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": pred_bbox,
                    "score": 1.0
                })

            # Calculate IoU for tabular report
            iou_val = calculate_iou(pred_coords, gt_coords)
            all_results_data.append({
                "product": p_folder.name,
                "image": target_img_path.name,
                "iou": iou_val
            })

            img_id += 1

        shutil.rmtree(temp_dir)

        # Eval mAP
        if gt_dict["annotations"]:
            coco_gt = COCO()
            coco_gt.dataset = gt_dict
            coco_gt.createIndex()

            if dt_list:
                coco_dt = coco_gt.loadRes(dt_list)
            else:
                # DUMMY
                dummy_dt = []
                coco_dt = coco_gt.loadRes(dummy_dt) if dummy_dt else COCO()
                if not dummy_dt: 
                    coco_dt.dataset = {"images": gt_dict["images"], "annotations": [], "categories": gt_dict["categories"]}
                    coco_dt.createIndex()

            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            map_score = coco_eval.stats[0]
            final_mAP_scores[p_folder.name] = map_score
            print(f"RES -> mAP for {p_folder.name}: {map_score:.4f}")

            # Update markdown report
            with open(REPORT_FILE_MD, "a") as f:
                f.write(f"## {p_folder.name}\n")
                f.write(f"mAP: {map_score:.4f}\n\n")

    # Final Text Report
    write_text_report(all_results_data, final_mAP_scores)

if __name__ == "__main__":
    evaluate_products()
