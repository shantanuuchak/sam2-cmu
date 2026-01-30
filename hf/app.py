import torch
import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import gradio as gr

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

import sam2
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
CHECKPOINT_PATH = "sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "sam2.1/sam2.1_hiera_t.yaml"
SAM2_BASE_PATH = os.path.dirname(sam2.__file__)

# Helper to ensure checkpoint exists
def ensure_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Downloading checkpoint {CHECKPOINT_PATH}...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        urllib.request.urlretrieve(url, CHECKPOINT_PATH)
        print("Download complete.")

def get_coords_from_mask(mask_img):
    if mask_img is None:
        return None
        
    # Gradio might provide an RGBA image or a binary one. 
    # Converting to Greyscale (L) and using > 0 threshold is the most robust.
    pil_mask = Image.fromarray(mask_img.astype(np.uint8)).convert("L")
    mask = np.array(pil_mask)
    
    ys, xs = np.where(mask > 0) 
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]

def show_mask(mask, ax, obj_id=0):
    cmap = plt.get_cmap("tab10")
    color = np.array([*cmap(obj_id)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def predict_tracking(source_img, source_mask, target_img):
    if source_img is None or source_mask is None or target_img is None:
        return None, "Please upload all three images."

    ensure_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Get box from mask
    coords = get_coords_from_mask(source_mask)
    if coords is None:
        return None, "No object detected in the source mask. Ensure the mask is bright on a dark background."
    
    [xmin, xmax, ymin, ymax] = coords
    
    # 2. Setup Predictor
    config_dir = os.path.join(SAM2_BASE_PATH, "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        predictor = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT_PATH, device=device)

    # 3. Create temp directory for SAM2
    temp_dir = tempfile.mkdtemp()
    Image.fromarray(source_img).save(os.path.join(temp_dir, "00000.jpg"))
    Image.fromarray(target_img).save(os.path.join(temp_dir, "00001.jpg"))

    try:
        inference_state = predictor.init_state(video_path=temp_dir)
        predictor.reset_state(inference_state)

        # Add box prompt
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        predictor.add_new_points_or_box(inference_state, frame_idx=0, obj_id=1, box=box)

        # Propagate
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 4. Visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(target_img)
        
        target_mask = video_segments[1][1]
        show_mask(target_mask, plt.gca(), obj_id=1)
        
        mask_coords = np.where(target_mask[0])
        if mask_coords[0].size > 0:
            y_min, y_max = np.min(mask_coords[0]), np.max(mask_coords[0])
            x_min, x_max = np.min(mask_coords[1]), np.max(mask_coords[1])
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     linewidth=3, edgecolor='lime', facecolor='none')
            plt.gca().add_patch(rect)
            status_msg = f"Success! Object tracked to coordinates: [{x_min}, {y_min}, {x_max}, {y_max}]"
        else:
            status_msg = "Object lost in target frame. No pixels above threshold."
        
        plt.axis('off')
        
        out_path = os.path.join(temp_dir, "result.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        result_img = Image.open(out_path)
        return result_img, status_msg

    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        # shutil.rmtree(temp_dir) # Handle carefully or let system clean up
        pass

# --- GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 SAM 2.1 Zero-Shot Object Tracker")
    gr.Markdown("Upload a source image, its bounding mask, and a target image to predict the new location.")
    
    with gr.Row():
        with gr.Column():
            src_input = gr.Image(label="1. Source Image")
            mask_input = gr.Image(label="2. Source Mask (PNG/Binary)")
            tgt_input = gr.Image(label="3. Target Image")
            btn = gr.Button("🔍 Track Object", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Predicted Result")
            output_text = gr.Textbox(label="Result Info")

    btn.click(fn=predict_tracking, inputs=[src_input, mask_input, tgt_input], outputs=[output_img, output_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
