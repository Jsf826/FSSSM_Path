## FSSSM-Path: Trusted Similarity Diffusion for Federated Semi-Supervised Pathology Segmentation

This repository implements **FSSSM-Path**, a trusted similarity diffusion framework for **federated semi‑supervised pathology image segmentation**.
The code corresponds to the method described in our paper, which targets extremely high‑resolution whole‑slide images and heterogeneous client distributions.

### 1. Overview

- **Task**: Federated semi‑supervised pathology image segmentation under non‑IID, limited‑annotation, privacy‑preserving settings.
- **Key idea**: Instead of naive global aggregation, FSSSM-Path performs **trusted similarity diffusion**:
  - density‑aware dynamic encoder that adapts to local structural complexity,
  - dual‑domain client reliability estimation (model + data distribution),
  - correlation graph‑driven pseudo‑label diffusion for structure‑aware supervision.
- **Architecture entry**: `FSSSM_Path.py`

### 2. Datasets

We evaluate FSSSM-Path on three pathology segmentation datasets:

- **LUAD-HistoSeg**: Lung adenocarcinoma histopathology segmentation.
- **BCSS**: Breast Cancer Semantic Segmentation dataset.
- **HNCCS**: A newly constructed **Head and Neck Cancer Cell Segmentation (HNCCS)** dataset.

Data should be organized per dataset as:

```text
<DATA_ROOT>/<DATASET_NAME>/
  images/
    xxx.png
  masks/
    xxx.png      # same filename as the corresponding image
```

In this repo, `options.py` uses a single root path:

```text
--path_pathology_seg
  images/
  masks/
```

For multi‑dataset experiments, you can either:

- run separate experiments with different `--path_pathology_seg`, or
- extend `PathologySegDataset` to include a dataset ID / mixing strategy.

### 3. Method Components (Code Mapping)

- **Density-aware Dynamic Encoder (DDE)**
  - File: `Model/dde_unet.py`
  - Function: `build_dde_unet`
  - Role: computes grayscale entropy, assigns each sample to a granularity level, and builds structure‑adaptive features for later similarity modeling and pseudo‑label refinement.

- **Federated Semi-Supervised Training**
  - File: `FSSSM_Path.py`
  - Class `Client`: local semi‑supervised training (`local_train`), including:
    - supervised loss: cross‑entropy + Dice on labeled patches,
    - pseudo‑supervised loss: correlation graph‑driven pseudo‑label diffusion on unlabeled patches,
    - cross‑client consistency: consistency on shared anchor features across trusted neighbors.
  - Function `compute_dual_neighbors`: dual‑domain distance \(D_{ij}\) (2‑Wasserstein + parameter distance) and cosine similarity \(S_{ij}\), used to select trusted neighbors \(\mathcal{N}_i^*\).
  - Function `decentralized_aggregation`: centerless aggregation over trusted neighbors with weights derived from \(D_{ij}\).

- **Correlation Graph-driven Pseudo-label Diffusion**
  - Function: `pseudo_label_diffusion` in `FSSSM_Path.py`
  - Builds pixel‑wise affinity graph from feature maps, performs local diffusion, then region‑level refinement around high‑confidence anchors to generate structure‑consistent pseudo‑labels.

- **Dataset Loader**
  - File: `Dataset/pathology_seg.py`
  - Class: `PathologySegDataset`
  - Handles PNG/TIFF pathology images and masks, resizing to a unified patch size and returning `(image, mask)` tensors.

### 4. Installation

1. Create environment:

```bash
conda create --name fsssm_path python=3.8
conda activate fsssm_path
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 5. Configuration

Main hyper‑parameters are defined in `options.py`:

- **Federation & semi-supervision**
  - `--num_clients`: number of federated clients
  - `--seg_labeled_ratio`: labeled ratio per client (default `0.3`, i.e., 30% labeled / 70% unlabeled)
  - `--seg_num_rounds`: number of communication rounds

- **Segmentation**
  - `--seg_num_classes`: number of segmentation classes (including background)
  - `--patch_size`: input patch size
  - `--seg_batch_size`: batch size (default `32`)

- **Reliability & diffusion**
  - `--dual_alpha`, `--dual_C1`, `--tau_trust`, `--tau_sim`
  - `--graph_topk`, `--tau_conf`, `--tau_region`

- **Paths**
  - `--path_pathology_seg`: root of pathology dataset (`images/` and `masks/`)

### 6. Running FSSSM-Path

Example command (single dataset, single run):

```bash
python FSSSM_Path.py \
  --gpu_id 0 \
  --num_clients 5 \
  --seg_num_classes 2 \
  --seg_labeled_ratio 0.3 \
  --seg_batch_size 32 \
  --seg_num_rounds 50 \
  --path_pathology_seg /path/to/LUAD-HistoSeg
```

To reproduce results with multiple seeds (e.g., 5 runs), vary `--seed`:

```bash
for s in 1 2 3 4 5; do
  python FSSSM_Path.py --seed $s ...
done
```

### 7. Notes

- This implementation focuses on the **federated semi‑supervised segmentation framework**; data preprocessing (patch extraction from WSIs, stain normalization, etc.) should be performed beforehand.
- The current code assumes that all three datasets (LUAD‑HistoSeg, BCSS, HNCCS) can be loaded into the `PathologySegDataset` interface via appropriate `path_pathology_seg`.

