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
REPORT_FILE = OUTPUT_DIR / "report.md"

def get_coords_from_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [xs.min(), xs.max(), ys.min(), ys.max()]

def evaluate_products():
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dir = os.path.join(SAM2_BASE_PATH, "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT_PATH), device=device)

    product_folders = sorted([f for f in PRODUCTS_ROOT.iterdir() if f.is_dir()])
    
    for p_folder in product_folders:
        print(f"Processing {p_folder.name}")
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
                predictions[out_frame_idx] = [coords[1].min(), coords[1].max(), coords[0].min(), coords[0].max()]
            else:
                predictions[out_frame_idx] = None

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

            pred_coords = predictions.get(i)
            if pred_coords:
                pred_bbox = [pred_coords[0], pred_coords[2], pred_coords[1] - pred_coords[0], pred_coords[3] - pred_coords[2]]
                dt_list.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": pred_bbox,
                    "score": 1.0
                })

            img_id += 1

        shutil.rmtree(temp_dir)

        # Eval if stuff
        if gt_dict["annotations"]:
            coco_gt = COCO()
            coco_gt.dataset = gt_dict
            coco_gt.createIndex()

            if dt_list:
                coco_dt = coco_gt.loadRes(dt_list)
            else:
                coco_dt = COCO()
                coco_dt.dataset = {"images": gt_dict["images"], "annotations": [], "categories": gt_dict["categories"]}
                coco_dt.createIndex()

            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            map_score = coco_eval.stats[0]
            print(f"mAP for {p_folder.name}: {map_score}")

            # Quick report
            with open(REPORT_FILE, "a") as f:
                f.write(f"## {p_folder.name}\n")
                f.write(f"mAP: {map_score}\n\n")

if __name__ == "__main__":
    evaluate_products()