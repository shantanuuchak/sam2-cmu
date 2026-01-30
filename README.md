# Overview

This is a repository for the CMU SAM2 project. We first take the sample image + corresponding mask and then try to generate masks for the other images in the dataset. We use the SAM2 model to generate the masks. We use the pycocotools library to evaluate the masks. 

# Requirements

- Requires `uv` as the package manager. Install it using `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Requires `python3.12`.

# Usage

- `uv sync` to install dependencies.
- `uv run main.py` to run the script.
- `uv run evaluation.py` to evaluate the masks.

# Dataset

The dataset is available at `products/`. It consists of images and corresponding masks.

> Note: We only use the first image's mask to create embedding to hone zero-shot capabilities of SAM2. Rest of the masks are used for evaluation purposes only.

# Web Dashboard

The `hf` directory contains the code for the Gradio dashboard used for deployment. You can access the working web app directly here: [SAM2-CMU on Hugging Face](https://huggingface.co/spaces/shantanuuchak/sam2-cmu)

> [!IMPORTANT]
> The `hf` directory is for deployment purposes only and should not be taken into consideration for the main project logic or evaluation.