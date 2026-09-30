# Grokking

This repo contains all code for my MS Thesis studying the impact of distributed training environments on the Grokking Phenomena of Transformer Neural Networks trained for modular arithmetic.

This implementation reproduces the **one-layer, ReLU Transformer for modular addition modulo 113** analyzed by Nanda et al. in [“Progress Measures for Grokking via Mechanistic Interpretability”](https://arxiv.org/abs/2301.05217). It is not the two-layer architecture from the original Grokking paper.

## Experimental source and configuration

The canonical configuration is `configs/modular_addition_p113.yaml`. Its architecture, full-batch training, split, AdamW settings, and Fourier ablations follow the paper and the authors’ [companion repository](https://github.com/neelnanda-io/Grokking) / [Colab notebook](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20). A source-by-source comparison, mathematical metric definitions, and disclosed implementation choices are in [`docs/progress_measures.md`](docs/progress_measures.md).

| Setting | Canonical value |
| --- | --- |
| Modulus and data | `p=113`; all ordered pairs; 30% Python-shuffled training split |
| Transformer | 1 layer, 4 heads, `d_model=128`, `d_head=32`, `d_mlp=512`, ReLU |
| Normalization/positions | No LayerNorm; learned positional embeddings; causal attention |
| Optimization | Full-batch AdamW, LR `1e-3`, weight decay `1.0`, betas `(0.9, 0.98)`, 10-step LR warmup |
| Duration | 40,000 epochs (paper; notebook training cell defaults to 50,000) |
| Mainline Fourier frequencies | `14, 35, 41, 42, 52` (configurable) |

Some details are not fixed by the paper (for example, the model-initialization seed, checkpoint cadence, and AdamW epsilon); values and source status are explicitly marked in the documentation and YAML rather than left implicit.

## Install

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a locally installed PyTorch build with Apple Silicon support, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) if the default package does not expose MPS.

## Run a Mac/MPS smoke test

The smoke configuration retains the p=113 model and data pipeline but runs only five epochs. It verifies the end-to-end pipeline; it is **not** expected to grok.

```bash
python -m scripts.train --config configs/debug_mps.yaml
```

To require a particular device, change `device.selection` to `mps`, `cuda`, or `cpu`. `auto` selects CUDA, then MPS, then CPU. An unavailable explicit device falls back to CPU only when `allow_cpu_fallback: true`.

PyTorch may warn that a particular MPS backward operation has no deterministic implementation. The code seeds all RNGs and requests deterministic algorithms in warning mode so such device-specific kernels do not prevent training; strict bitwise reproducibility across MPS, CUDA, and CPU is not guaranteed.

## Run the canonical experiment

```bash
python -m scripts.train --config configs/modular_addition_p113.yaml
```

Set `experiment.seed` to select a new data split and model initialization. The code seeds Python, NumPy, and PyTorch, records the selected device and environment, and saves the effective YAML configuration in each run directory.

At startup, the script prints a run summary with the selected device, data/split sizes, seed, architecture and parameter count, training duration and loss, optimizer settings, logging cadences, Fourier-key mode, runtime versions, and output/checkpoint paths. The YAML has independent `train_metrics_interval`, `validation_interval`, `progress_interval`, `mechanistic_metrics_interval`, `checkpoint_interval`, and `ablation_interval` values. The canonical run evaluates train/test metrics each epoch, prints a terminal progress line every 1,000 epochs, and computes progress measures every 100 epochs. Progress lines include elapsed and interval time, train/validation accuracy, and loss. Checkpoints are saved every 1,000 epochs to limit disk use, plus a final checkpoint; the complete metric curves do not depend on retaining every intermediate checkpoint.

## Run the distributed staleness experiment

The distributed simulator uses **five logical nodes** and the same p=113 model, dataset split, full-batch gradients, and canonical AdamW settings:

```bash
python -m scripts.train_distributed --config configs/modular_addition_p113.yaml --tau 5
```

The number of global updates comes from `experiment.epochs` in the YAML. Each global step samples one model version per node, computes five full-training-split gradients, averages them, and applies one AdamW update to the latest central model. For `tau > 0`, nodes use the current model synchronously until `tau` update-history steps exist; afterward, each node independently samples a lag uniformly from `1..tau`. The history keeps the current model and its `tau` predecessors.

Use `--tau 0` for the synchronous control: all five nodes calculate gradients from the same current model. Their mean is equivalent to one centralized full-batch gradient, up to floating-point summation roundoff.

Distributed runs write ordinary train/test metrics for the updated global model at every global step to `metrics.jsonl` and `metrics.csv`. `node_metrics.jsonl` contains one record per node per step, with its sampled lag, source model version, and that source model's train/test metrics. Mechanistic source-model measures are attached at `mechanistic_metrics_interval`; the new global model's mechanistic measures and Fourier spectra are computed at the same cadence. Checkpoints also save the retained model history and staleness RNG state. The simulator uses five logical nodes in one process; it does not simulate network delays or launch separate workers.

When `save_plots` is enabled, the distributed command also creates `figures/nodes/node_0/` through `node_4/`, with per-node train/test loss and accuracy, restricted/excluded metrics, weight/Gini progress, Fourier spectra, key-logit coefficients, individual-frequency ablations, and staleness plots. Mechanistic plots are sampled at the mechanistic logging cadence and describe the node's selected gradient-source model; x-axis values are global steps. To regenerate these node figures from an existing run:

```bash
python -m scripts.plot_node_metrics results/<distributed-run-name>
```

## Outputs

Runs are written under `results/` by default. A run directory contains:

- `config.yaml`, `environment.json`: effective configuration, seed, device, package/runtime details, and source commit when available.
- `metrics.jsonl`: complete, machine-readable per-epoch metrics and periodic mechanistic measures.
- `metrics.csv`: tabular form of the same records (structured frequency/spectrum values are JSON-encoded).
- `fourier_spectra.csv`: embedding and neuron-to-logit Fourier component norms at mechanistic-analysis epochs.
- `final.pt`: final checkpoint; periodic checkpoints are under `checkpoints/`.
- `figures/`: generated global loss, accuracy, restricted/excluded, weight norm, Gini, Fourier-spectrum, and logit-coefficient plots. Distributed runs additionally contain `figures/nodes/node_<id>/` per-node figures.

The fixed paper frequencies and the frequencies detected from each checkpoint are recorded separately. To use frequency discovery as the active restricted-loss definition, set `mechanistic.key_frequencies.mode: discover`; this changes the active frequency set over time and is therefore distinct from the paper-frequency curve.

## Analyze or plot an existing run

Analyze a checkpoint independently of training:

```bash
python -m scripts.analyze_checkpoint checkpoints/<run-name>/final.pt
```

Regenerate the figures from saved raw metrics without retraining:

```bash
python -m scripts.plot_results results/<run-name>
```

For the restricted/excluded loss definitions, frequency-identification algorithm, Fourier basis, and paper-versus-notebook differences, see [`docs/progress_measures.md`](docs/progress_measures.md).

## Project layout

```text
configs/       Canonical p=113 and short debug YAML configurations
docs/          Source notes and defensible mathematical metric definitions
scripts/       Train, analyze checkpoints, and regenerate global/per-node plots
src/           Dataset, model, Fourier analysis, progress measures, trainer, logging
tests/         Dataset, architecture, Fourier, ablation, and training smoke tests
results/       Run metrics and figures (generated)
checkpoints/   Periodic and final checkpoints (generated)
```

Run lightweight tests with `pytest`.
