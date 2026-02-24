# DualStream-ConvViT – Satellite Image Classification on EuroSAT

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Vision_Transformer-2025-orange" alt="ViT">
  <img src="https://img.shields.io/badge/EuroSAT-SOTA-green" alt="SOTA">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT">
</p>

**Novel hybrid architecture** that fuses a **scale-aware multi-dilation CNN stream** (local textures + altitude-like multi-scale view) with a **Vision Transformer stream** via **bidirectional cross-stream attention**.




## Key Innovations
- **Scale-AwareBlock**: 3 parallel dilated depthwise separable convs (dilation 1/2/4) → simulates multi-altitude satellite view
- **Bidirectional Cross-Stream Attention**: CNN spatial features enrich ViT tokens **and** ViT global context enriches CNN tokens
- Stochastic depth, label smoothing, warmup + cosine LR
- Gradient saliency + prediction entropy for explainability





# Runs EDA → Training → Evaluation → Explainability automatically
python main.py
