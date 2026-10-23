# Related-Key Multi-Pair Neural Distinguishers

Command-line guide for related-key perturbation screening, Inception–ECA training, staged fine-tuning, and checkpoint evaluation.


Run commands from the repository root, containing:

- `train_nets.py`: the matching version exporting `binary_label_generator`, `build_callbacks`, `evaluate_model`, `train_model`, and `write_json`.
- `RKmcp.py`: the matching flat-input model factory accepting `pairs`, `plain_bits`, `reg_param`, and `dropout`.
- `make_data_train.py`: `NDCMultiPairGenerator`.
- `YuanWang_RK_SENet.py`: `make_multipair_senet` 
- `cipher/`: `present80.py`, `simmeck3264.py`, `hight.py`, and `lea.py`.

**Model import prerequisite:** the supplied `eval_nets.py` imports factories from `RKmcp_fixed`, while training imports them from `RKmcp`. Before running evaluation, make the evaluator import the exact model implementation used for training. If `RKmcp.py` is authoritative, replace its `from RKmcp_fixed import ...` line with:

```python
from RKmcp import make_model_inception, make_model_inception_no_eca
```

## Environment

Use the Python/TensorFlow/Keras/CUDA versions of the original experiment when reproducing saved checkpoints. The commands here have been checked against script argument definitions; full GPU execution has not been validated as part of this documentation.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Bash, `.venv\Scripts\activate` in Windows CMD, or `.venv\Scripts\Activate.ps1` in PowerShell. Install a validated dependency set, then record it:

```bash
python -m pip install -r requirements.txt
python -m pip freeze > environment.txt
```

Dependencies used by these workflows include NumPy, TensorFlow/Keras, SciPy, scikit-learn, tqdm, and CuPy for the GPU paths. Install a CuPy distribution compatible with your CUDA environment. The supplied screening script calls CuPy even with `--no-gpu`; that flag does not currently provide a CuPy-free workflow. Do not assume the older repository requirements file is a verified environment lock for the newer scripts.

Commands below use single lines and explicit arguments. Replace checkpoint placeholders before execution. Create output directories for evaluation logs:

```bash
python -c "from pathlib import Path; Path('results/evaluation').mkdir(parents=True, exist_ok=True)"
```

## Cipher settings and data conventions

| Cipher | CLI module | Plaintext difference | Key bit index |
|---|---|---|---:|
| PRESENT-80 | `present80` | `0x80` | 56 |
| SIMECK-32/64 | `simmeck3264` | `0x40` | 57 |
| HIGHT | `hight` | `0x4000000000000000` | 97 |
| LEA-128 | `lea` | `0x8000000080` | 94 |

A sample is a group of `k = --pairs` ciphertext pairs. All pairs in a group share one related-key pair. Positive groups use the chosen plaintext difference; negative groups use independently sampled plaintexts under the same related-key relation. The generator emits a flat vector of length `k * 3 * plain_bits`, with per-pair features `[delta_C, C, C_star]`.

Use `--input-diff` and `--delta-key-bit` consistently. The key-bit argument is a zero-based array index; in the training convention, index 0 is the most significant bit. The supplied evaluation and staged-training hex-key helper fills the array least-significant-bit first, so `--delta-key-hex` is not interchangeable with the training hex representation. The examples deliberately use the bit-index option.

## 1. Related-key perturbation screening

The following recipes scan from round 1 through at most one round beyond the selected screening round. They use `k=1`, two PCA components (hard-coded), a silhouette threshold of 0.1, and 10,000 samples per candidate. This sample budget is an explicit example, not a verified historical setting.

### PRESENT-80

```bash
python finding_key_bit.py --cipher-name present80 --input-diff 0x80 --pairs 1 --start-round 1 --max-rounds 8 --stop-score-threshold 0.1 --n-samples 10000 --batch-size 10000 --csv-output results/screening/present80.csv
```

### SIMECK-32/64

```bash
python finding_key_bit.py --cipher-name simmeck3264 --input-diff 0x40 --pairs 1 --start-round 1 --max-rounds 11 --stop-score-threshold 0.1 --n-samples 10000 --batch-size 10000 --csv-output results/screening/simmeck3264.csv
```

### HIGHT

```bash
python finding_key_bit.py --cipher-name hight --input-diff 0x4000000000000000 --pairs 1 --start-round 1 --max-rounds 12 --stop-score-threshold 0.1 --n-samples 10000 --batch-size 10000 --csv-output results/screening/hight.csv
```

### LEA-128

```bash
python finding_key_bit.py --cipher-name lea --input-diff 0x8000000080 --pairs 1 --start-round 1 --max-rounds 8 --stop-score-threshold 0.1 --n-samples 10000 --batch-size 10000 --csv-output results/screening/lea.csv
```

## 2. Training from scratch

Each command below runs Inception–ECA with up to 30 epochs, fresh training and validation data, Adam AMSGrad, BCE, plateau learning-rate reduction, and early stopping. The best checkpoint is selected by `val_bce` and is reloaded before automatic evaluation. These are current-code settings, not a claim that every manuscript result used them.

### PRESENT-80

```bash
python main.py --cipher present80 --model inception --rounds 8 --pairs 32 --input-diff 0x80 --delta-key-bit 56 --epochs 30 --num-samples-train 1000000 --num-samples-val 250000 --batch-size 2048 --val-batch-size 2048 --loss bce --lr 0.0003 --lr-schedule plateau --min-lr 0.000001 --lr-patience 3 --patience 10 --reg-param 0.0001 --dropout 0.2 --seed 42 --data-backend cupy --encrypt-backend cupy --test-samples 1000000 --test-repeats 5 --test-batch-size 2048 --out-dir checkpoints
```

### SIMECK-32/64

```bash
python main.py --cipher simmeck3264 --model inception --rounds 11 --pairs 32 --input-diff 0x40 --delta-key-bit 57 --epochs 30 --num-samples-train 1000000 --num-samples-val 250000 --batch-size 2048 --val-batch-size 2048 --loss bce --lr 0.0003 --lr-schedule plateau --min-lr 0.000001 --lr-patience 3 --patience 10 --reg-param 0.0001 --dropout 0.2 --seed 42 --data-backend cupy --encrypt-backend cupy --test-samples 1000000 --test-repeats 5 --test-batch-size 2048 --out-dir checkpoints
```

### HIGHT

```bash
python main.py --cipher hight --model inception --rounds 12 --pairs 32 --input-diff 0x4000000000000000 --delta-key-bit 97 --epochs 30 --num-samples-train 1000000 --num-samples-val 250000 --batch-size 2048 --val-batch-size 2048 --loss bce --lr 0.0003 --lr-schedule plateau --min-lr 0.000001 --lr-patience 3 --patience 10 --reg-param 0.0001 --dropout 0.2 --seed 42 --data-backend cupy --encrypt-backend cupy --test-samples 1000000 --test-repeats 5 --test-batch-size 2048 --out-dir checkpoints
```

### LEA-128

```bash
python main.py --cipher lea --model inception --rounds 8 --pairs 32 --input-diff 0x8000000080 --delta-key-bit 94 --epochs 30 --num-samples-train 1000000 --num-samples-val 250000 --batch-size 2048 --val-batch-size 2048 --loss bce --lr 0.0003 --lr-schedule plateau --min-lr 0.000001 --lr-patience 3 --patience 10 --reg-param 0.0001 --dropout 0.2 --seed 42 --data-backend cupy --encrypt-backend cupy --test-samples 1000000 --test-repeats 5 --test-batch-size 2048 --out-dir checkpoints
```

The examples explicitly request CuPy data and encryption paths; use backends supported by your cipher implementation. TensorFlow GPU execution is separate from the generator backend flags.

For multi-pair comparisons, repeat a run with `--pairs 1`, `2`, `4`, `8`, `16`, or `32` at each intended round. Each setting is a separate training experiment. Do not treat omitted manuscript configurations as reported measurements.

Output directory: `checkpoints/<cipher>/<timestamp-uuid>/`.

| File | Content |
|---|---|
| `config.json` | Resolved settings, differences, run ID, TensorFlow version |
| `architecture.json` | Serialized model architecture |
| `best.weights.h5` | Selected checkpoint |
| `history.json` | Training/validation metric histories |
| `summary.json` | Training duration, completed epochs, selection criterion |
| `evaluation.json` | Repeated evaluation output; absent with `--skip-eval` |

Use `--patience 0` to disable early stopping, `--lr-schedule constant` to disable plateau scheduling, or `--skip-eval` to defer evaluation. These change the protocol and should be recorded.

## 3. Staged fine-tuning

Example: initialize from a compatible SIMECK round-11 checkpoint with `k=32`, and fine-tune at round 12. This is an execution example, not an assertion of successful round-12 distinguishing.

```bash
python staged_train.py --cipher simmeck3264 --model inception --pairs 32 --input-diff 0x40 --delta-key-bit 57 --init-weights checkpoints/simmeck3264/REPLACE_WITH_SOURCE_RUN_ID/best.weights.h5 --stages-rounds 12,12,12 --stages-epochs 3,5,8 --stages-lrs 0.00005,0.00002,0.000005 --stages-modes freeze,partial,full --partial-blocks 1 --train-samples 10000000 --val-samples 250000 --batch-size 5000 --val-batch-size 20000 --use-gpu --save-final --out-dir staged_runs/simmeck3264_r12_k32_RUN_A
```

- All four stage lists must have equal lengths. No stage is automatically skipped.
- `freeze` freezes the shared pair encoder; `partial` opens the final specified pair residual blocks and final ECA/embedding while keeping encoder BatchNorm frozen; `full` enables all layers.
- Freeze/partial require the matching architecture with `pair_encoder_td`. The older Flatten-based model is incompatible with these modes.
- Each phase creates a fresh Adam AMSGrad optimizer and uses BCE.
- Each phase selects its checkpoint by `val_loss` (including model regularization), then reloads it before the next phase. This differs from scratch training's `val_bce` criterion.
- Keep the cipher, differences, architecture, and pair count consistent with the source checkpoint in this recipe.
- Use a distinct `--out-dir` for every run: named final/stage artifacts can otherwise be overwritten.
- This script has no CLI seed, automatic final evaluation, LR scheduler, or early stopping. It does not accept the older `--use-chunks`/`--chunk-size` flags.

Per-phase files are saved under `<out-dir>/<run_id>/stage<index>_<round>r_<mode>/`, including `best.weights.h5` and `training.csv`. Final files for this example include:

- `simmeck3264_inception_k32_staged_final.weights.h5`
- `simmeck3264_inception_k32_staged_final.keras` (with `--save-final`)
- `simmeck3264_inception_k32_staged_history.json`

## 4. Independent checkpoint evaluation

First resolve the model-import prerequisite above. This example evaluates one fixed SIMECK round-15 checkpoint on five fresh test sets of one million groups each:

```bash
python eval_nets.py --cipher-module cipher.simmeck3264 --model inception --model-path checkpoints/simmeck3264/REPLACE_WITH_R15_K32_RUN_ID/best.weights.h5 --rounds 15 --pairs 32 --input-diff 0x40 --delta-key-bit 57 --n-repeat 5 --test-samples 1000000 --batch-size 5000 --log-path results/evaluation/simmeck3264_r15_k32.txt
```

Evaluate the staged model from the previous example with:

```bash
python eval_nets.py --cipher-module cipher.simmeck3264 --model inception --model-path staged_runs/simmeck3264_r12_k32_RUN_A/simmeck3264_inception_k32_staged_final.keras --rounds 12 --pairs 32 --input-diff 0x40 --delta-key-bit 57 --n-repeat 5 --test-samples 1000000 --batch-size 5000 --log-path results/evaluation/simmeck3264_staged_r12_k32.txt
```

Weights-only files must end in `.weights.h5` and use the correct factory. Full `.keras` models retain their saved architecture. The evaluator does not expose all architecture overrides from scratch training, so do not assume it can reconstruct arbitrary modified models.

Reported standard deviation is `numpy.std` with `ddof=0` across fresh test sets for one fixed checkpoint; it is not variation across independently trained models. Throughput measures model prediction on synthetic bit inputs and excludes cipher/data-generation cost.

**Statistical reporting limitation:** this evaluator hard-codes null standard deviation to 0.0005, corresponding to the elementary Bernoulli(0.5) approximation at one million observations. It does not adjust it when `--test-samples` changes or validate the null assumptions. Do not use its printed p-values as general significance evidence without correcting and validating the test. The recipes use one million groups explicitly; the CLI default is only 100,000.

## 5. Cross-round evaluation without retraining

Keep `--model-path`, `--pairs 32`, and both differences fixed. Change only `--rounds` and the log filename. The manuscript checkpoint mapping is:

| Cipher | Checkpoint training round | Evaluation rounds |
|---|---:|---|
| PRESENT-80 | 10 | 10–15 |
| SIMECK-32/64 | 15 | 15–18 |
| HIGHT | 14 | 14–17 |
| LEA-128 | 10 | 10–12 |

The source checkpoint itself must be supplied; its training history is not established by its filename. In particular, the LEA checkpoint was trained at **round 10**, not round 9. The following commands show the first target round beyond each source round:

```bash
python eval_nets.py --cipher-module cipher.present80 --model inception --model-path checkpoints/present80/REPLACE_WITH_R10_K32_RUN_ID/best.weights.h5 --rounds 11 --pairs 32 --input-diff 0x80 --delta-key-bit 56 --n-repeat 5 --test-samples 1000000 --batch-size 5000 --log-path results/evaluation/present80_train10_test11_k32.txt
```

```bash
python eval_nets.py --cipher-module cipher.simmeck3264 --model inception --model-path checkpoints/simmeck3264/REPLACE_WITH_R15_K32_RUN_ID/best.weights.h5 --rounds 16 --pairs 32 --input-diff 0x40 --delta-key-bit 57 --n-repeat 5 --test-samples 1000000 --batch-size 5000 --log-path results/evaluation/simmeck3264_train15_test16_k32.txt
```

```bash
python eval_nets.py --cipher-module cipher.hight --model inception --model-path checkpoints/hight/REPLACE_WITH_R14_K32_RUN_ID/best.weights.h5 --rounds 15 --pairs 32 --input-diff 0x4000000000000000 --delta-key-bit 97 --n-repeat 5 --test-samples 1000000 --batch-size 5000 --log-path results/evaluation/hight_train14_test15_k32.txt
```

```bash
python eval_nets.py --cipher-module cipher.lea --model inception --model-path checkpoints/lea/REPLACE_WITH_R10_K32_RUN_ID/best.weights.h5 --rounds 11 --pairs 32 --input-diff 0x8000000080 --delta-key-bit 94 --n-repeat 5 --test-samples 1000000 --batch-size 5000 --log-path results/evaluation/lea_train10_test11_k32.txt
```

Repeat for all target rounds in the table using the same checkpoint. Do not run staged training between evaluations: that would test adaptation rather than fixed-checkpoint transfer. Near-chance performance is a transfer result, not an achieved distinguishing depth.



## ResNet baseline

`resnet.py` exports `make_resnet(...)` (the supplied architecture and defaults)
and `make_multipair_resnet(pairs, plain_bits)`. The latter sets
`dimension = 3 * pairs * plain_bits` for the existing generator.
Inception remains the default model. Select ResNet with `--model resnet`
in `main.py`, `eval_nets.py` (weights-only reconstruction), or `staged_train.py`.

```bash
python main.py --model resnet --cipher present80 --rounds 7 --pairs 4 --input-diff 0x80 --delta-key-bit 22
```

Use the plaintext/key differences corresponding to your actual experiment.
For weights-only evaluation, pass `--model resnet` with the same cipher and
pair count used in training. Full model loading remains architecture-driven.
ResNet staged fine-tuning supports full training only: explicitly pass
`--stages-modes full,full,full` for a three-stage schedule. Freeze/partial
modes implement the Inception shared pair encoder and do not apply to ResNet.

Architecture: pointwise Conv1D (32 filters), five residual blocks each with
two Conv1D layers (32 filters, kernel 3), flatten, Dense 64, Dense 64, and
one sigmoid output. Each convolution and hidden dense layer uses BN then
ReLU; the residual addition follows the second ReLU, with no post-addition
activation. All convolution/dense kernels use L2 = 1e-4. No dropout is used.
**Input semantics:** the original reshape `(dimension/2, 2)` and permutation
are retained, yielding `(batch, 2, dimension/2)` before Conv1D. This is NOT a
`(batch, k*n, 3)` convolutional representation. Do not describe it as such in
the manuscript. The factory does not compile the model or print a summary;
call `model.summary()` explicitly when needed.
