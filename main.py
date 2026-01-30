import os
import torch
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2_video_predictor
import sam2

# =============================================================================
# 1. INPUT CONFIGURATION
# =============================================================================
# Paths to your local images and mask
SOURCE_IMAGE_PATH = "/workspaces/sam2-cmu/source/can_chowder_000001.jpg"
SOURCE_MASK_PATH = "/workspaces/sam2-cmu/source/can_chowder_000001_1_gt.png"
TARGET_IMAGE_PATH = "/workspaces/sam2-cmu/source/can_chowder_000003.jpg"

# Path to your SAM2 installation and weights - AUTO-DETECTED
SAM2_BASE_PATH = os.path.dirname(sam2.__file__)
CHECKPOINT_PATH = "/workspaces/sam2-cmu/sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "sam2.1/sam2.1_hiera_t.yaml"

# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================
def get_bbox_from_mask(mask_path):
    """Extracts [xmin, ymin, xmax, ymax] from a mask image."""
    mask = np.array(Image.open(mask_path).convert("L"))
    pos = np.where(mask > 0)
    if pos[0].size == 0:
        return None
    return np.array([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])], dtype=np.float32)

# =============================================================================
# 3. CORE TRANSCODING PIPELINE
# =============================================================================
def run_transcoding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory: {temp_dir}")
    
    try:
        # A. Setup temporary 'video' environment for SAM2
        shutil.copy(SOURCE_IMAGE_PATH, os.path.join(temp_dir, "00000.jpg"))
        shutil.copy(TARGET_IMAGE_PATH, os.path.join(temp_dir, "00001.jpg"))
        print("Copied source and target images to temp directory")

        # B. Initialize SAM2 Video Predictor
        config_dir = os.path.abspath(os.path.join(SAM2_BASE_PATH, "configs"))
        print(f"SAM2 base path: {SAM2_BASE_PATH}")
        print(f"Config directory: {config_dir}")
        print(f"Config directory exists: {os.path.exists(config_dir)}")
        
        GlobalHydra.instance().clear()
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            predictor = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT_PATH, device=device)
        
        print("SAM2 video predictor initialized")
        inference_state = predictor.init_state(video_path=temp_dir)
        print("Inference state initialized")

        # C. Define object using Bounding Box from source mask
        bbox = get_bbox_from_mask(SOURCE_MASK_PATH)
        if bbox is None:
            raise ValueError("No object found in the source mask.")
        
        print(f"Bounding box from mask: {bbox}")

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=bbox,
        )
        print("Added bounding box to frame 0")

        # D. Propagate to target image (Transcoding)
        print("Starting propagation...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            print(f"Processed frame {out_frame_idx}")

        # E. Visualization of results
        if 1 not in video_segments or 1 not in video_segments[1]:
            raise ValueError("Failed to generate mask for target frame")
            
        final_mask = video_segments[1][1][0]
        print(f"Final mask shape: {final_mask.shape}")
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(SOURCE_IMAGE_PATH))
        plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                          edgecolor='red', facecolor='none', lw=2))
        plt.title("Source (Frame 0) + Input BBox")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(Image.open(TARGET_IMAGE_PATH))
        plt.imshow(final_mask, alpha=0.5, cmap='jet')
        plt.title("Transcoded Result (Frame 1)")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = "/workspaces/sam2-cmu/transcoding_result.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
        
        plt.show()
        
        print("\n✓ Transcoding completed successfully!")

    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("Pipeline execution finished. Temporary files cleaned.")

# =============================================================================
# 4. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_transcoding()