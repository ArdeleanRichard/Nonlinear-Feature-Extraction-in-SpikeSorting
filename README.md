# A Study of Non-Linear Manifold Feature Extraction in Spike Sorting

**Benchmarking Manifold Learning for Next-Generation Neural Data Analysis**

[![DOI](https://img.shields.io/badge/DOI-10.1007/s12021--025--09744--3-blue)](https://doi.org/10.1007/s12021-025-09744-3)

This repository contains the code and configurations used in the study:

**"A Study of Non-Linear Manifold Feature Extraction in Spike Sorting"**  
by Eugen-Richard Ardelean and Raluca Portase, published in *Neuroinformatics*, 2025.

---

## Overview

This project investigates non-linear manifold learning methods for **spike sorting**, a key step in processing extracellular neuronal recordings. We compare modern approaches such as **PHATE, UMAP, TriMap, t-SNE, Diffusion Maps, Isomap, LLE** against classical methods (**PCA, ICA, MDS, Autoencoders**) to determine which feature extraction techniques produce the most separable and robust embeddings of spike waveforms.

Experiments were conducted on:

- **95 synthetic datasets** (with ground truth cluster labels).  
- **2 real datasets** (spe-1 recordings, Marques-Smith et al.) with dual extracellular/intracellular ground truth validation.

Results show that **UMAP, PHATE, and TriMap** consistently outperform other methods, offering strong separability, robustness to noise, and scalability for high-density probes.

---

## Datasets

### Synthetic Spike Waveforms (Pedreira et al., 2012)
- **Description**: 95 single-channel synthetic datasets derived from real monkey recordings.  
- **Characteristics**: 2–20 clusters per dataset, ~9,300 spikes on average, including multi-unit clusters.  
- **Usage**: Benchmarking feature extraction across diverse conditions.  
- **Access**: Publicly available.  

### Real Datasets (spe-1, Marques-Smith et al., 2018/2020)
- **Description**: Patch-clamp + 384-channel CMOS extracellular recordings in rat cortex.  
- **Ground Truth**: Dual intracellular/extracellular data for 21 neurons.  
- **Datasets Used**: c28 and c37.  

---

## Feature Extraction Methods

- **Linear**: PCA, MDS, ICA  
- **Non-linear**: KPCA, SOM, Autoencoders  
- **Manifold learning**: LLE, MLLE, HLLE, LTSA, Isomap, Spectral Embedding, Diffusion Maps, PHATE, UMAP, TriMap, t-SNE  

Clustering was performed using **K-Means** for evaluation.

---

## Results Summary

Performance was evaluated using clustering metrics: Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI), Purity, Silhouette Score (SS), Calinski-Harabasz Score (CHS), and Davies-Bouldin Score (DBS).

**Key findings:**
- **UMAP, PHATE, TriMap**: best overall performance across metrics and datasets.  
- **t-SNE, Autoencoders**: good separability but slower runtime.  
- **Diffusion Maps, LLE, MLLE**: strong only for datasets with few clusters, degrade with complexity.  
- **PCA, ICA, MDS, SOM**: consistently underperformed.  

---

## Citation

If you use this work, please cite:

```bibtex
@article{Ardelean2025Manifold,
  title     = {A Study of Non-Linear Manifold Feature Extraction in Spike Sorting},
  author    = {Ardelean, Eugen-Richard and Portase, Raluca},
  journal   = {Neuroinformatics},
  year      = {2025},
  volume    = {23},
  pages     = {48},
  doi       = {10.1007/s12021-025-09744-3}
}
```

---

## 📬 Contact

For questions, please contact:
📧 [ardeleaneugenrichard@gmail.com](mailto:ardeleaneugenrichard@gmail.com)

---

