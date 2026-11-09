# PV-Ground + Entropy-Gradient ROI Sampling

3D visual grounding on ScanRefer: given a 50k-point indoor cloud and a natural
language description, predict the 3D bounding box of the referred object.

On top of [PV-Ground](https://github.com/AaNnWwTt/PV-Ground) this repository adds
a **training-free ROI re-sampling branch**: besides the 1024 FPS seeds, a small
region of interest is selected with

$$
R_i=\mathrm{rank}(f_i)\cdot\mathrm{rank}(s_i)\cdot\widehat{\mathrm{cons}}_i,
\qquad s_i=\Big\|\frac{\partial H_{sel}}{\partial F_i}\Big\|_2
$$

($s_i$: entropy-gradient score of seed $i$; $f_i$: box-proposal field built from
the model's own predicted boxes; $\mathrm{cons}_i$: kNN spatial consensus), and
256 extra points are FPS-sampled from one robust ball around the
weighted-median centre of the reliable seeds.  The seeds become 1024 → 1280
tokens; **no module is trained, no parameter is updated, no extra loss is used,
and no ground truth is involved at inference**.

---

## 1. Installation

```bash
conda create -n 3dgrounding python=3.12 -y
conda activate 3dgrounding

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy plyfile open3d matplotlib tqdm easydict pyyaml \
            transformers==4.44.2 tokenizers ipdb

# CUDA operators
source setup_cuda_paths.sh
cd pointnet2   && python setup.py install && cd ..
cd segmentator && python setup.py install && cd ..
```

## 2. Data and pretrained model

* **Data**: follow the official
  [PV-Ground README](https://github.com/AaNnWwTt/PV-Ground/blob/main/README.md)
  to prepare ScanNet + ScanRefer and put everything under `./data/`.
* **Pretrained weights**: download `PV-Ground_ScanRefer_single.pth` from
  [HuggingFace](https://huggingface.co/AaNnWwTt/PV-Ground) and place it at
  `./data/PV-Ground_ScanRefer_single.pth`.

## 3. Inference

**Baseline (1024 seeds)**

```bash
bash tools/test_scanrefer_single.sh
```

**Entropy-gradient ROI sampling (best configuration)**

```bash
EXTRA_ARGS="--roi --roi_selector field --roi_points 256 --roi_radius_max 2.0" \
bash tools/test_scanrefer_single.sh
```

## 4. Results

ScanRefer `val`, 9508 expressions, released checkpoint, metric = `mask@kiou`
(overall).  Baseline is averaged over 3 runs (σ ≈ 0.13 pp).

| method | Acc@0.25 | Acc@0.50 |
|---|---|---|
| baseline (1024 seeds) | 0.6187 | 0.5448 |
| **+ entropy-gradient ROI (256 extra points)** | **0.6231** | **0.5583** |

## 5. Visualisation

Six panels per sample (entropy-gradient score, box-proposal field, Top-32
multi-ball ROI, robust single-ball ROI, and the sampled points):

```bash
python tools/visualize_roi_points.py \
    --checkpoint_path data/PV-Ground_ScanRefer_single.pth \
    --data_root ./data/ --output_dir visualization_results \
    --num_samples 20 --roi_points 256
```

## 6. Acknowledgement

Built upon [PV-Ground](https://github.com/AaNnWwTt/PV-Ground),
[BUTD-DETR](https://github.com/nickgkan/butd-detr) and
[Group-Free-3D](https://github.com/zeliu98/Group-Free-3D).
