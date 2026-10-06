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

At startup, the script prints a run summary with the selected device, data/split sizes, seed, architecture and parameter count, training duration and loss, optimizer settings, logging cadences, Fourier-key mode, runtime versions, and output/checkpoint paths. The YAML has independent `train_metrics_interval`, `validation_interval`, `progress_interval`, `mechanistic_metrics_interval`, `checkpoint_interval`, and `ablation_interval` values. The canonical run evaluates train/test metrics each epoch, prints a terminal progress line every 1,000 epochs, and computes progress measures every 100 epochs. Progress lines include elapsed and interval time, train/validation accuracy, and loss. Checkpoints are saved every 1,000 epochs plus the final checkpoint. The canonical command always starts a fresh run; loading/resuming from a checkpoint is not implemented.

The canonical `metrics.jsonl`/`metrics.csv` are the **one-node baseline**. They contain one record per epoch, including ordinary train/test metrics and periodic mechanistic measures. This path is unchanged by the distributed experiment.

## Run the distributed staleness experiment

The distributed simulator defaults to **five logical nodes** and uses the same p=113 model, dataset split, full-batch gradients, and canonical AdamW settings. Set `--nodes` to select another positive count:

```bash
python -m scripts.train_distributed --config configs/modular_addition_p113.yaml --tau 5
python -m scripts.train_distributed --config configs/modular_addition_p113.yaml --tau 5 --nodes 3
```

The number of global updates comes from `experiment.epochs` in the YAML. Each global step samples one model version per node, computes one full-training-split gradient per node, averages them, and applies one AdamW update to the latest central model. Every node uses the same complete training and test splits; data is not partitioned by node. For `tau > 0`, nodes use the current model synchronously until `tau` update-history steps exist; afterward, each node independently samples a lag uniformly from `1..tau`. The history keeps the current model and its `tau` predecessors.

Use `--tau 0` for the synchronous distributed control: all nodes calculate gradients from the same current model. Their mean is equivalent to one centralized full-batch gradient, up to floating-point summation roundoff. This is a separate run from the canonical one-node command above.

### Mixed zero-delay and fixed-delay nodes

To assign the first `N` nodes zero delay and the remaining nodes exactly `tau` global updates of delay, use the explicit mixed fixed-delay mode:

```bash
python -m scripts.train_distributed --config configs/modular_addition_p113.yaml \
  --fixed-delay 5 --zero-delay-nodes 2
```

This assigns nodes `[0, 1]` to lag 0 and nodes `[2, 3, 4]` to lag 5. As in the uniform-staleness mode, all nodes use the current model during startup until the required delay history exists; afterward the assignments are deterministic and fixed. `--zero-delay-nodes 0` makes all nodes fixed-delay nodes, while setting it equal to `--nodes` makes all nodes current-model nodes. `--nodes` configures the total count in either distributed mode and defaults to 5. A fixed delay of 0 is synchronous for all nodes. These options must be provided together and cannot be combined with `--tau`; the existing `--tau` mode remains independent per-node uniform random staleness.

The fixed-delay mode also supports `--step-size-sweep`. Its run directories identify the staleness mode, delay, zero-delay node count, and learning-rate factor/rate.

To compare learning rates at a fixed τ, add `--step-size-sweep`:

```bash
python -m scripts.train_distributed --config configs/modular_addition_p113.yaml --tau 5 --step-size-sweep
```

This runs three fresh experiments sequentially at the YAML learning rate, `0.1×` that rate, and `0.01×` that rate. The seed, split, architecture, node count, delay settings, and all other training settings are held constant. Each run's directory name includes its staleness mode/parameters, node count, and learning-rate factor/rate, and its effective `config.yaml`, metrics, checkpoints, and figures are stored separately as usual. The terminal prints each factor/rate before starting and summarizes all output paths at the end. If `--run-dir` is supplied, it is used as a parent directory for three tagged run directories; it must not already contain those run directories.

Distributed runs write ordinary train/test metrics for the updated global model at every global step to `metrics.jsonl` and `metrics.csv`. These records identify the global step and include each node's lag and source model version. `node_metrics.jsonl` contains one record per node per step, with its sampled lag, source model version, and that source model's train/test metrics. Thus node-level curves are per-node views of selected gradient-source snapshots, not separate persistent node models. Mechanistic source-snapshot measures are attached at `mechanistic_metrics_interval`; the updated global model's mechanistic measures and Fourier spectra are also computed at that cadence. Fixed paper keys and keys discovered for each analyzed model are recorded separately. Restricted loss/accuracy is reported for both key sets at the mechanistic cadence; excluded loss/accuracy for both sets and individual-frequency ablations are computed at the ablation cadence. Checkpoints also save the retained model history and staleness RNG state. The simulator uses logical nodes in one process; it does not simulate network delays or launch separate workers. Like the canonical command, it always starts a fresh run rather than resuming from a checkpoint.

When `save_plots` is enabled, the distributed command also creates `figures/nodes/node_<id>/` for each configured node, with per-node train/test loss and accuracy, separate restricted/excluded plots comparing fixed paper frequencies with frequencies discovered in that source model, weight/Gini progress, Fourier spectra, key-logit coefficients, individual-frequency ablations, and staleness plots. Mechanistic plots are sampled at the mechanistic/ablation logging cadences and describe the node's selected gradient-source model; x-axis values are global steps. To regenerate these node figures from an existing run:

```bash
python -m scripts.plot_node_metrics results/<distributed-run-name>
```

The `*_keys.png` plots deliberately separate fixed paper frequencies (`14, 35, 41, 42, 52`) from frequencies discovered in the current model. This matters in distributed runs: if the model learns a different Fourier solution, accuracy can remain high after removing the paper frequencies while restriction to those paper frequencies remains weak. Check `detected_key_frequencies`, and compare the `*_fixed` and `*_discovered` metrics, before interpreting that pattern.

For runs created before fixed-versus-discovered metric fields were added, plot regeneration can only show values that were actually saved. It cannot reconstruct absent discovered-key accuracy/exclusion values without the corresponding model checkpoints.

## Outputs

Runs are written under `results/` by default. A run directory contains:

- `config.yaml`, `environment.json`: effective configuration, seed, device, package/runtime details, and source commit when available.
- `metrics.jsonl`: canonical run records per epoch; distributed run records per global optimizer step, with periodic mechanistic measures and distributed lag/source-version details.
- `metrics.csv`: tabular form of those global records (structured frequency/spectrum values are JSON-encoded).
- `node_metrics.jsonl` (distributed only): one record per node per global step, identifying node, source model version, staleness, train/test metrics, and periodic source-model progress measures.
- `fourier_spectra.csv`: global-model embedding and neuron-to-logit Fourier component norms at mechanistic-analysis steps.
- `final.pt`: final checkpoint; periodic checkpoints are under `checkpoints/<run-name>/`. Distributed checkpoints additionally contain retained staleness history and RNG state.
- `figures/`: generated global loss, accuracy, fixed/discovered restricted/excluded curves, weight norm, Gini, Fourier-spectrum, and logit-coefficient plots. Distributed runs additionally contain `figures/nodes/node_<id>/` per-node figures.

The fixed paper frequencies and the frequencies detected from each analyzed model are recorded separately. `mechanistic.key_frequencies.mode` selects which set is used for the unqualified `restricted_*` metric: `fixed` (canonical default) uses the YAML paper frequencies; `discover` uses that model's detected keys. Explicit `*_fixed` and `*_discovered` fields are logged separately so the two analyses can be compared regardless of the active mode. Excluded-by-key-set variants require ablation metrics to be enabled and are calculated at `ablation_interval`.

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
