# Replication sources and progress-measure definitions

## Sources and scope

This repository targets the **one-layer Transformer on modular addition modulo 113** in Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability* (2023), especially Sections 3, 4.4, 5.1–5.3 and Appendix C. The authors’ companion materials are the [`neelnanda-io/Grokking` repository](https://github.com/neelnanda-io/Grokking) and its [linked Colab notebook](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20). The GitHub repository itself is a notebook/artifact companion, not a maintained training package. The notebook includes the custom Transformer and training code used as the reference here.

The target is **not** the two-layer Transformer from the original Grokking paper. Nanda et al.'s mainline experiment is explicitly a one-layer ReLU Transformer.

## Experimental configuration and provenance

“Paper” below means explicit in the Nanda et al. paper. “Notebook” means stated or reconstructed from the authors' linked Colab code. “Implementation choice” identifies a detail not fixed by those sources.

| Experimental choice | Canonical value in this repository | Provenance / qualification |
| --- | --- | --- |
| Task and modulus | \(a+b \pmod {113}\), \(p=113\) | Paper, Section 3; notebook setup |
| Dataset | Every ordered pair \((a,b)\in\{0,\ldots,112\}^2\); label \((a+b)\bmod 113\) | Notebook `gen_train_test` and `fn`; the paper specifies modular addition |
| Number of examples | 12,769 total; first `int(0.3 * 12,769) = 3,830` shuffled pairs train, 8,939 test | Inferred from paper's 30% and notebook's `int` split |
| Split and seed | Shuffle all pairs with Python `random.shuffle`, split at 30%; split seed 0 | Notebook. It seeds Python's `random` for the split only; it does not seed PyTorch in the training cell |
| Input | Three token IDs `[a, b, 113]`, representing `a|b|=`; `0..112` are numbers and 113 is the equals token | Paper, Section 3; notebook (`d_vocab=p+1`, `n_ctx=3`, input records `(i,j,p)`) |
| Output | Read final-position logits. Training cross-entropy includes all 114 output classes; evaluation drops the equals-token class. | Notebook `full_loss` trains on all model outputs, while `test_logits` truncates the equals-token class for evaluation. |
| Layers | 1 Transformer block | Paper and notebook |
| Attention | 4 causal heads, `d_model=128`, `d_head=32`; score scale \(1/\sqrt{32}\); concatenated head outputs projected to 128 | Paper and notebook architecture |
| MLP | 512 hidden units; ReLU | Paper and notebook |
| Residual/norm | Residual connections around attention and MLP; no LayerNorm (including final norm) | Notebook architecture; paper explicitly says no LayerNorm |
| Position embeddings | Learned, length 3 | Paper and notebook |
| Embedding / unembedding | Separate learned matrices; not tied; token embedding and output include the equals token, but Fourier analyses omit it | Paper and notebook |
| Dropout | None | Inferred from the complete reference architecture, which contains no dropout |
| Initialization | Explicit `randn` initializers: \(W_E,W_{pos},W_Q,W_K,W_V,W_O,W_{in},W_{out}\sim N(0,1/d_{model})\) entry variance; \(W_U\sim N(0,1/d_{vocab})\); MLP biases zero | Notebook code. These are standard deviations `1/sqrt(dimension)`; no other biases are used |
| Initialization seed | Seed all Python, NumPy, and PyTorch RNGs with the configured seed (default 0) | **Implementation choice:** required for deterministic reruns. The notebook only explicitly seeds the Python split RNG, so this does not claim the authors' original parameter draw |
| Batch size | Full batch: all 3,830 train examples per optimizer update | Paper and notebook |
| Optimizer | AdamW, learning rate \(10^{-3}\), weight decay 1.0, betas `(0.9, 0.98)` | Paper gives optimizer/LR/decay; betas from notebook |
| AdamW epsilon | `1e-8` | **Implementation choice:** PyTorch AdamW default; not explicitly overridden by the notebook |
| LR schedule | Linear warmup `min(step / 10, 1)` | Notebook. The first scheduled update uses LR 0; then the LR increases over 10 scheduler steps |
| Weight decay parameter groups | Same AdamW weight decay on every trainable parameter, including biases | Inferred from the reference's single `AdamW(model.parameters(), ...)` call |
| Epochs | 40,000 | Paper, Section 3. **Source discrepancy:** notebook training UI defaults to 50,000; its analyzed mainline model is the 40,000-epoch checkpoint, and its training-dynamics analysis spans checkpoints through 50,000 |
| Early stopping | Disabled (`null`; equivalent to notebook threshold `-1`, which positive cross-entropy never reaches) | Notebook training cell; paper does not specify an early-stop rule |
| Early stopping | Disabled (`stopping_thresh=-1`) | Notebook default |
| Objective / precision | Mean categorical cross-entropy on all final-position output logits (including the equals-token class); reference computes `log_softmax` in float64 | Inferred from notebook `full_loss` / `cross_entropy_high_precision`; paper does not specify arithmetic precision |
| Ordinary evaluation | Full train and test losses are computed each epoch; the reference prints every 100 epochs | Notebook code |
| Mechanistic cadence | Every 100 epochs | Notebook progress analysis uses checkpoints at 100-epoch increments |
| Checkpoint cadence | Every 1,000 epochs plus final model | **Implementation choice:** smaller storage footprint. Metrics and Fourier spectra are still saved every 100 epochs; set `logging.checkpoint_interval: 100` to retain every mechanistic checkpoint |
| Mainline key frequencies | `{14, 35, 41, 42, 52}` | Paper's analyzed mainline run; configurable in YAML. Not assumed universal across seeds |
| Device | `auto`, `cuda`, `mps`, or `cpu` | **Implementation choice** for this portable research repository; math and parameters are unchanged |
| MPS cross-entropy | Float64 loss is evaluated on CPU when logits are on MPS | **Implementation choice/workaround:** maintains the reference's float64 loss calculation, with autograd through the device copy |

### Data-order detail

The canonical Fourier grid must have rows ordered lexicographically by \((a,b)\). The reference creates `all_data` in that order and marks train/test membership by pair. The implementation therefore keeps `all_inputs` in lexicographic order, while the seeded shuffle determines *indices* for the train/test partition. This preserves both the reference split and the two-dimensional Fourier-grid layout.

## Fourier conventions

For \(p=113\), use an orthonormal real 1D Fourier basis ordered as

\[
\phi_0(x)=1/\sqrt p,\quad
\phi_{k,c}(x)=\sqrt{2/p}\cos(2\pi kx/p),\quad
\phi_{k,s}(x)=\sqrt{2/p}\sin(2\pi kx/p).
\]

The basis has 113 rows: DC followed by cosine/sine pairs for \(k=1,\ldots,56\). These normalization/order conventions match the companion notebook's `fourier_basis`. A 2D transform of a function on input pairs is the tensor-product projection
\[
\widehat{f}_{ij}=\sum_{a,b=0}^{p-1}\phi_i(a)\phi_j(b)f(a,b).
\]
Inputs to Fourier functions must be in lexicographic \((a,b)\) order. Activations used for frequency identification are centered across all \(p^2\) pairs before transforming, following the notebook.

## Progress measures

The paper's named **restricted loss** and **excluded loss** are Fourier-space *logit ablations*. They are not ablations of MLP neurons. “Trig loss” is the companion notebook's name for restricted loss.

| Measure | Mathematical definition and component | Dataset / cadence | Source and implementation correspondence |
| --- | --- | --- | --- |
| Restricted loss (`restricted_loss`) | For logits \(L(a,b)\in\mathbb R^p\), project the \((a,b)\) dimension onto the constant direction and, for each key \(k\), onto the two normalized directions \(\cos(2\pi k(a+b)/p)\) and \(\sin(2\pi k(a+b)/p)\). Zero all other input-Fourier directions; compute cross-entropy on the resulting logits. The constant is the original per-output mean logit. For the five mainline keys, this keeps DC plus 10 real input directions (described as 20 2D Fourier terms in the paper's product basis). | Canonical paper curve: all \(p^2\) possible pairs; also save train and test variants. Every 100 epochs by default. | Paper Section 5.1; notebook `trig_loss`, `get_component_cos_xpy`, `get_component_sin_xpy`, and `test_logits(..., bias_correction=True)`. Adding the mean logits implements the notebook's bias correction and paper's explicit constant component. |
| Restricted accuracy | Argmax accuracy on the same restricted logits. | All/train/test; additional interpretation measure, not a named paper metric. | **Implementation extension** using the exact restricted logits. |
| Excluded loss (`excluded_loss`) | Subtract the projections onto **all** key-frequency cosine/sine sum directions from the original logits, keep every other logit component, and compute cross-entropy. The canonical number is train-split loss. | Training split, every 100 epochs by default; test variant is separately labeled. | Paper Section 5.1 describes removing the key frequencies and measuring on training data. The companion code's `excl_loss` reports one curve per individual frequency; this implementation records both combined exclusion and per-frequency exclusions. |
| Per-frequency excluded loss/accuracy | For each key \(k\) independently, \(L_{\setminus k}=L-P_{k,c}L-P_{k,s}L\). Score this ablated output. No extra bias correction is applied. | Train is canonical; test is also saved as a clearly labeled extension. Every `ablation_interval`. | Companion notebook's `excl_loss`: subtracts `get_component_cos_xpy` and `get_component_sin_xpy` for one frequency and evaluates `mode='train'`, with `bias_correction=False`. This is the individual-frequency ablation requested for the appendix analysis. |
| Detected key frequencies | On each checkpoint, center final-position post-ReLU MLP activations \(A(a,b,n)\), take their 2D DFT, and for each neuron find the frequency \(k\) whose 3×3 block indexed by \(\{0,\cos k,\sin k\}\) in both input dimensions has greatest squared-energy fraction. A frequency is classified as a key cluster when at least one neuron assigned to it has fraction \(\ge 0.85\). The searched reference range is \(k=1,\ldots,\lfloor p/2\rfloor-1\). | Computed on all pairs at each mechanistic checkpoint; detected key list and per-neuron scores are logged. | Companion notebook cells “Neuron Clusters” and “Always Firing Neurons”: states the `frac > 0.85` cluster rule and computes this 3×3 fraction. The explicit threshold is source-derived, not newly tuned. |
| Frequency-set choice | Fixed mode uses the YAML list (canonical default `{14,35,41,42,52}`) for the progress curves; discover mode uses the checkpoint's detected set. Both sets and the active mode are logged. | Every mechanistic checkpoint. | The paper's principal time-series analysis holds frequencies identified from the trained mainline model fixed across its checkpoints. Checkpoint-local discovery is additionally exposed for seed variants; it can yield a moving target and must not be conflated with fixed-key curves. |
| Fixed-vs-discovered ablation variants | The implementation logs restricted loss/accuracy for both the fixed YAML set and the checkpoint-local detected set. At ablation cadence, it likewise removes each set as a whole and reports excluded loss/accuracy for both. `restricted_*` follows `frequency_mode`; `*_fixed` and `*_discovered` fields identify their sets explicitly. | All-pair restricted metrics every mechanistic step; excluded metrics at ablation steps. | **Implementation extension** added to disambiguate whether observed progress uses the paper's fixed mainline keys or keys detected in the current model. The paper's canonical curves remain the fixed-key variants. |
| Embedding Fourier spectrum | \(W_E\in\mathbb R^{d\times(p+1)}\); discard the equals-token column and compute \(\|\Phi W_E^\top\|_{2,\text{row}}\) for each of the 113 real Fourier basis rows \(\Phi\). Save both L2 and squared-L2 norms. | All number tokens; every 100 epochs. | Paper Section 5.1 and notebook `fourier_embed` (which stores squared component norms). The raw CSV stores both forms so plots can use either. |
| Neuron-to-logit map Fourier spectrum | \(W_L=W_UW_{out}\in\mathbb R^{p\times d_{mlp}}\), using only the \(p\) number-output rows of \(W_U\). Transform vocabulary/output rows as \(\Phi W_L\), and take each row's L2 norm across neurons. | Output vocabulary; every 100 epochs. | Paper Section 3 defines \(W_L=W_UW_{out}\); the notebook refers to the corresponding object as `W_logit = W_U @ W_out` and plots its 1D Fourier transform. |
| Fourier Gini: \(W_E,W_L\) | For the vector of 113 nonnegative Fourier-component L2 norms \(x\), sort \(x_{(i)}\) ascending and compute \(G=\frac{2\sum_{i=1}^{n}i x_{(i)}}{n\sum_i x_i}-\frac{n+1}{n}\). Zero vectors return 0. DC and each separate sine/cosine component are included. | Same matrices / cadence as their Fourier spectra. | Paper Section 5.1 names the Gini coefficient of Fourier-component norms and cites Hurley & Rickard. The paper and Colab do not provide executable Gini code or spell out binning/normalization; this uses the standard finite-sample Gini on L2 component norms. |
| Weight norm | `weight_l2_norm = sqrt(sum(parameter**2))` over all trainable parameters. Also save `weight_squared_norm = sum(parameter**2)`. | Whole model; every 100 epochs. | Paper Sections 5.1–5.3 discuss \(\ell_2\) norm / sum of squared weights. The companion's `sum_sq_weights` explicitly plots the sum of squares of every parameter. Both values are recorded to remove the terminology ambiguity. |
| Key-frequency logit coefficient | For each \(k\), \(c_k=\langle L(a,b,c),\cos(2\pi k(a+b-c)/p)\rangle/\|\cos(2\pi k(a+b-c)/p)\|_2^2\), over all pairs and output classes. | All pairs, one value per fixed and discovered key; every 100 epochs. | **Analysis extension:** this scalar coefficient is not specified as a paper progress metric or implemented as a time series in the notebook. It is a direct least-squares projection of the requested cosine template and should be described as such, not attributed to Nanda et al. |

## Key-frequency detection details and caveats

The frequency detector reproduces the notebook's activation-based approach rather than ranking the embedding spectrum or imposing a magnitude cutoff on output logits:

1. Run the model on all ordered pairs in lexicographic order and take each final-position MLP post-activation.
2. Subtract each neuron's mean across the \(p^2\) pairs; transform each neuron's \((a,b)\) activation table in the real 2D Fourier basis.
3. For each neuron and each \(k=1,\ldots,55\), sum squared coefficients in the 3×3 sub-block with indices \(\{DC,\cos k,\sin k\}\) for both \(a\) and \(b\), and divide by that neuron's total centered Fourier energy.
4. Assign the neuron to the frequency with the largest fraction. Include a frequency in the detected set only if an assigned neuron reaches the notebook's documented 0.85 explained-fraction cluster criterion. Zero-energy neurons are unassigned.

The notebook first computes `np.unique(neuron_freqs)` and only afterward marks neurons below 0.85 as the diffuse `-1` cluster; as written, that ordering can leave unsupported frequencies in the earlier `key_freqs` array. This implementation applies the stated 0.85 cluster criterion before forming the key set, and records all neuron assignments/scores for audit. This is a small correction to the notebook's ordering, not a new threshold.

The paper explicitly cautions that runs can learn different Fourier solutions. The canonical YAML's paper frequencies are a fixed baseline, not a claim that every seed must discover those exact modes. Checkpoint-local discovered frequencies are logged independently. For strict comparison of a non-mainline seed's entire trajectory, identify its key set on the designated trained checkpoint and then use that **same fixed set** when recomputing earlier checkpoints; a dynamic per-checkpoint set is a different diagnostic.

For distributed runs, node-level loss/accuracy describes the node's selected gradient-source checkpoint, not a persistent node-specific model: the five nodes contribute gradients to one shared central model. A node plot can therefore show different points at the same global step because each node may have selected a different source version. The restricted/excluded `*_keys` plots distinguish the fixed paper frequencies from each source checkpoint's detected frequencies; do not interpret a low fixed-key restricted accuracy as evidence that a run's learned circuit is absent until checking the discovered-key restriction too.

Older runs may not contain the newer `*_fixed`/`*_discovered` metric fields. Plot regeneration aliases the earlier active restricted metrics to fixed keys only when the saved mode was fixed, and aliases earlier excluded metrics to the fixed set. It cannot reconstruct discovered-key accuracies or discovered-set exclusions without the original model checkpoint; it leaves those curves absent rather than fabricating values.

## Paper/code differences and implementation limits

- **Epoch count:** the paper says 40,000; the training cell in the linked notebook has a 50,000 default. The analyzed mainline checkpoint is at 40,000, while its training-dynamics dataset spans 50,000. This repository uses the paper's 40,000 for canonical training.
- **Restricted terminology:** the paper calls the measure “restricted loss”; the notebook calls the same logit projection “trig loss”. The YAML frequencies determine its retained directions.
- **Excluded loss:** paper prose describes removing the key frequencies as a set on the training split; the notebook's training-dynamics implementation removes each one independently. Both aggregate-all-key and per-frequency versions are saved, alongside extra test metrics.
- **Gini:** no complete Gini implementation was found in the paper or linked notebook. The formula above is therefore a disclosed standard interpretation, and Gini results should not be claimed as bitwise reproduction of an unpublished implementation.
- **Weight norm:** paper prose uses \(\ell_2\)-norm wording; the notebook computes sum of squared weights. This code logs both.
- **Seeds:** the notebook fixes the split seed to 0 by default but does not seed Torch in its training cell. This code seeds Torch and NumPy too; model initialization is consequently deterministic but cannot be asserted to match the authors' original random draw.
- **MPS precision:** MPS does not provide float64 tensors for the loss. Logits are copied to CPU for the reference-style float64 cross-entropy computation; this affects performance, not model architecture or mathematical objective.
- **MPS deterministic kernels:** the pipeline seeds all supported RNGs and requests PyTorch deterministic algorithms with `warn_only=True`. On Apple Silicon, PyTorch may warn that an operation such as `index_put_with_accumulate_mps` has no deterministic implementation. MPS runs are therefore seeded/reproducible to the extent supported by their kernels, but bitwise determinism across repeated runs or device types is not promised.

## Distributed asynchronous-staleness extension

`python -m scripts.train_distributed --config configs/modular_addition_p113.yaml --tau 5` runs the thesis extension without changing the canonical single-run trainer. It is a **simulator of centralized, delayed-gradient aggregation**, not a multi-process or networked distributed implementation:

1. One central model is initialized exactly as in the canonical run. Five logical nodes all use the same complete training split and compute full-batch gradients.
2. At global model version \(t\), node \(i\) selects one source version \(s_i\). For \(\tau=0\), \(s_i=t\) for every node. For \(\tau>0\), global updates remain synchronous while \(t<\tau\). Once the required history exists, each node independently samples \(d_i\sim\mathrm{Uniform}\{1,\ldots,\tau\}\), and uses \(s_i=t-d_i\). This avoids inventing unavailable pre-run checkpoints and makes the configured staleness bound achievable from the first asynchronous update.
3. Each node evaluates the ordinary full-batch training loss at its selected source model and backpropagates there. Its parameter gradients are copied, then gradients from all five nodes are averaged parameter-wise.
4. A single canonical AdamW optimizer applies that averaged gradient once to the latest global model, with the same weight decay and warmup schedule as the reference experiment. The result is global model version \(t+1\), which is appended to history.

The five node gradients are calculated sequentially in one process/device to emulate stale-model gradient aggregation. No communication delays, networking, process-level parallelism, client dropout, or node-specific data partitioning are simulated. Since all nodes have identical full-batch data, differences between their gradients arise from their source model versions alone. At \(\tau=0\), all five gradients are identical; averaging them reproduces one centralized gradient (up to floating-point accumulation roundoff) while preserving the same AdamW optimizer steps.

Distributed metrics have two scopes:

- `metrics.jsonl` / `metrics.csv` store train/test loss and accuracy for the **new global model** after every global update, along with the sampled node lags and source versions.
- `node_metrics.jsonl` stores one record per node per global step. Its train/test metrics describe the model used for that node's gradient, identified by `source_model_version` and `staleness`. Mechanistic source-model measures are attached at the configured mechanistic cadence; the new global model's mechanistic metrics and Fourier spectra are also computed at that cadence. Thus ordinary global and node evaluation is per-step, while expensive Fourier/ablation analysis follows `mechanistic_metrics_interval` and `ablation_interval`.

Periodic distributed checkpoints include the current model/optimizer/scheduler, split indices, staleness configuration, retained model-version history, and Python RNG state used for staleness sampling. The current CLI always starts a fresh run; checkpoint resume is not implemented. Compare the canonical trainer against the `--tau 0` distributed control before interpreting the `--tau > 0` runs. Tests verify tau-zero parameter equivalence, synchronous warmup, valid uniform-lag bounds, and per-node metric records.
- **Mechanistic snapshots and checkpoints:** progress measures are calculated every 100 epochs by default. To constrain checkpoint storage, model-state checkpoints are saved every 1,000 epochs plus the final state; use a 100-epoch checkpoint interval when the actual intermediate states are needed for later recomputation.
