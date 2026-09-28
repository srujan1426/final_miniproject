# GraphContrastSum: Video Summarization using Graph Contrastive Learning

An AI-driven video summarization pipeline that combines ConvNeXt feature extraction, Graph Attention Networks (GAT) with contrastive learning, and BLIP multimodal captioning.

## Overview

This project implements an end-to-end framework to generate concise video summaries:
- **Feature Extraction:** Pretrained ConvNeXt-Tiny extracts visual representations from extracted video frames.
- **Graph Modeling & Contrastive Learning:** Constructs a k-NN graph of frame embeddings and applies a Graph Attention Network (`GraphContrastSum`) trained via contrastive loss to capture temporal and semantic relationships.
- **Multimodal Video Summarization:** Utilizes BLIP (`Salesforce/blip-image-captioning-base`) to generate frame captions and compute cross-modal relevance.
- **Summary Generation & Metrics:** Generates the final compiled summary video alongside evaluation metrics:
  - F-Score
  - Graph Diversity Index (GDI)
  - Compression Ratio (CR)

## Requirements

To run the notebook in Google Colab or locally:

```bash
pip install torch torchvision torchaudio torch-geometric opencv-python moviepy tqdm einops transformers sentencepiece matplotlib numpy scikit-learn
```

## Usage

1. Open [`final_miniproject.ipynb`](final_miniproject.ipynb) in Jupyter Notebook or Google Colab.
2. Upload the target video file when prompted.
3. Run all cells sequentially to extract frames, train/evaluate the graph contrastive model, generate captions, and synthesize the summarized video output.
