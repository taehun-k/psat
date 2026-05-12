<div align="center">

### Structural dependency-aware generative design of elastic metasurfaces via pseudo-supervised attention-based transformer

#### Computer Methods in Applied Mechanics and Engineering &mdash; Volume 458, Article 119041 (2026)

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cma.2026.119041-blue)](https://doi.org/10.1016/j.cma.2026.119041) [![Journal](https://img.shields.io/badge/Journal-CMAME-red)](https://www.sciencedirect.com/journal/computer-methods-in-applied-mechanics-and-engineering)

<u>**Taehun Kim**</u><sup>1,&dagger;</sup> &nbsp; Donghyu Lee<sup>1</sup> &nbsp; Juhwan Han<sup>1</sup> &nbsp; Sayhee Kim<sup>1</sup> &nbsp; Byeng D. Youn<sup>1,\*</sup> &nbsp; Soo-Ho Jo<sup>2,\*</sup>

<sup>1</sup>Seoul National University &nbsp;&nbsp; <sup>2</sup>Dongguk University<br/>
<sup>&dagger;</sup>First author. &nbsp;&nbsp; <sup>\*</sup>Corresponding authors.

</div>

PSAAT is a generative design framework for elastic metasurfaces that explicitly models the structural dependency between unit-cell geometric parameters with a pseudo-supervised attention mechanism. Given a target transmission response, the model produces structural-parameter samples that meet the target while remaining physically consistent.

---

## Repository structure

```
psat/
├── arguments.py                 # Default runtime / material arguments
├── main.py                      # Entry point (Hydra-driven, interactive mode select)
├── train.py                     # Trainer classes: forward surrogate & PSAAT generator
├── test.py                      # Evaluation, design generation, attention analysis
├── utils.py                     # Plotting & weight-init helpers
│
├── config/                      # Hydra configs (one per generated-parameter subset)
│   ├── psaat_gen_l2l5.yaml
│   ├── psaat_gen_l1l2l5.yaml
│   ├── psaat_gen_l2l3l5.yaml
│   ├── psaat_gen_l2l4l5.yaml
│   ├── psaat_gen_l2l5l6.yaml
│   └── psaat_gen_l2l5l7.yaml
│   └── psaat_gen_l2l5l8.yaml
│
├── data/
│   ├── custom.py                # `masstiff_metasurface` Dataset
│   └── dataset/
│       ├── design_variable/     # Structural parameters: DV_<vars>.mat
│       └── response/
│           └── 100k/            # Transmission/reflection @ 100 kHz
│
├── model/
│   ├── attention.py             # Multi-head attention block, positional encoding
│   ├── forward.py              # AttnFWD / BaseFWD forward surrogates
│   └── inverse.py                 # PSAAT generator + discriminator
│
└── experiments/
    ├── saved_model/             # Pretrained checkpoints (.pt)
    ├── learning_curve/          # Training curves & history logs
    ├── fwd_evaluation/          # Forward-model evaluation plots
    ├── physical_analysis_attention/  # Attention-score visualizations
    ├── test_dataset_generation/ # Ground-truth response plots
    └── practical_case/
        ├── psaat/               # Generated designs
        └── comsol_model/        # COMSOL .mph files for FEM validation
```

---

## Requirements

- Python **3.10**
- PyTorch **2.5** with CUDA **12.1** (other CUDA builds also work; pick the wheel matching your driver)
- Other Python packages: `hydra-core`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `torchinfo`, `accelerate`, `wandb`, `nlopt`

### Installation

```bash
conda create -n psaat python=3.10 -y
conda activate psaat

# PyTorch 2.5 (CUDA 12.1 build). Swap cu121 -> cu118 / cu124 to match your environment.
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

pip install hydra-core scipy scikit-learn pandas matplotlib seaborn tqdm \
            torchinfo accelerate wandb nlopt
```

> Check your CUDA build with `nvidia-smi` (max supported CUDA from the driver) or, after install,
> `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`.

---

## How to run

The single entry point is `main.py`. The Hydra config is selected with `--config-name`:

```bash
python main.py --config-name=psaat_gen_l2l5
```

The program then prompts you to choose a mode:

| Key | Mode          | Description |
|-----|---------------|-------------|
| `a` | `model-spec`  | Print parameter counts and FLOPs for PSAAT / Discriminator / Forward surrogates |
| `b` | `fwd-train`   | Train the forward surrogate (AttnFWD, BaseFWD, or both) |
| `c` | `fwd-test`    | Evaluate the trained forward surrogates against ground-truth responses |
| `d` | `inv-train`   | Train the inverse PSAAT generator + discriminator |
| `e` | `inv-test`    | Evaluate the PSAAT generator and run physical analysis of attention scores |
| `f` | `generate`    | Generate designs for a practical case (refracting / focusing) |
| `g` | `comsol`      | Export generated designs for COMSOL FEM validation |

### Selecting a config

Each `config/psaat_gen_*.yaml` defines which structural parameters are *generated* (the rest are conditioned). For example, `psaat_gen_l2l5.yaml` sets `gen_idx: [1,4]` (i.e. ℓ₂ and ℓ₅). Available configs:

- `psaat_gen_l2l5` &nbsp; (2 generated variables)
- `psaat_gen_l2l3l5`, `psaat_gen_l2l4l5`, `psaat_gen_l2l5l6`, `psaat_gen_l2l5l7` &nbsp; (3 generated variables)

### Selecting the operating frequency

The frequency is a Hydra-overridable field (`freq`). Valid values are `100k`:

```bash
python main.py --config-name=psaat_gen_l2l5 freq=100k
```

### End-to-end example

Train PSAAT and then generate a refracting metasurface at 100 kHz:

```bash
# 1) Train the forward surrogate
python main.py --config-name=psaat_gen_l2l5 freq=100k
# > select (b) fwd-train, then (c) Both

# 2) Train the inverse generator
python main.py --config-name=psaat_gen_l2l5 freq=100k
# > select (d) inv-train

# 3) Generate a design for a target refraction angle
python main.py --config-name=psaat_gen_l2l5 freq=100k
# > select (f) generate
# > r          # refracting
# > 15        # target refracted angle in degrees
# > 10        # number of unit cells
```

Generated designs are written to `experiments/practical_case/psaat/gen_designs/`.

---

## Pretrained checkpoints

`experiments/saved_model/` contains the trained weights used in the paper:

- `best_psaat_gen_<vars>_<freq>.pt` &mdash; inverse generator (PSAAT)
- `best_attn_fwd_gen_<vars>_<freq>.pt` &mdash; attention-based forward surrogate
- `best_base_fwd_gen_<vars>_<freq>.pt` &mdash; MLP baseline forward surrogate

They are loaded automatically by modes `(c)`, `(e)`, `(f)`, and `(g)` based on the active config and `freq`.

---

## Dataset

Located in `data/dataset/`:

- `design_variable/DV_<vars>.mat` &mdash; structural parameters of each unit cell
- `response/<freq>/DV_<vars>_4cell_response.mat` &mdash; transmission/reflection coefficients computed on 4-unit-cell supercells at the corresponding frequency (`100k`)

The dataset loader (`data/custom.py`) draws non-overlapping random subsets for train / valid / test according to `data.{train,valid,test}.n_data` in the YAML config.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{kim2026psaat,
  title   = {Structural dependency-aware generative design of elastic metasurfaces via pseudo-supervised attention-based transformer},
  author  = {Kim, Taehun and Lee, Donghyu and Han, Juhwan and Kim, Sayhee and Youn, Byeng D. and Jo, Soo-Ho},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {458},
  pages   = {119041},
  year    = {2026},
  doi     = {10.1016/j.cma.2026.119041}
}
```

Plain-text reference:

> T. Kim, D. Lee, J. Han, S. Kim, B. D. Youn, S.-H. Jo, "Structural dependency-aware generative design of elastic metasurfaces via pseudo-supervised attention-based transformer," *Computer Methods in Applied Mechanics and Engineering*, vol. 458, art. 119041, 2026. https://doi.org/10.1016/j.cma.2026.119041

---

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Contact

For questions about the code or paper, please contact the corresponding authors:

- Byeng D. Youn &mdash; Seoul National University
- Soo-Ho Jo &mdash; Dongguk University
