"""BCE staged fine-tuning for the current flat-input Inception-ECA model.

Each list entry is one phase; repeat a target round for freeze/partial/full.
Defaults: freeze 3 epochs @ 5e-5, partial 5 @ 2e-5, full 8 @ 5e-6.
Freeze requires pretrained weights. For scratch use --stages-modes full
and matching one-entry rounds/epochs/LR lists. No stages are auto-skipped.
Partial opens the last N pair residual blocks plus final ECA/embedding,
while encoder BatchNorm remains frozen. Full also updates encoder BN.
A fresh Adam is compiled at EVERY phase (including round changes).
Place this script beside RKmcp.py, make_data_train.py, and cipher/.
"""
import argparse
import os
import json
import datetime as dt
import numpy as np
import tensorflow as tf
from pathlib import Path
from RKmcp import make_model_inception, make_model_inception_no_eca

TRAIN_NUM_SAMPLES = 10_000_000
VAL_NUM_SAMPLES = 250_000
# CuPy compatibility check
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = np
    _CUPY_AVAILABLE = False
    print("[WARNING] CuPy not available, falling back to NumPy (CPU mode)")

from make_data_train import NDCMultiPairGenerator


def integer_to_binary_array(int_val, num_bits):
    if int_val < 0 or int_val >= 1 << num_bits:
        raise ValueError("Input difference exceeds plaintext width")
    return _int_to_bits(int_val, num_bits).reshape(1, num_bits)


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-stage training for neural distinguisher")
    ap.add_argument("--cipher", type=str, default="present80", help="Cipher module under cipher/ (e.g. present80, simon3264)")
    ap.add_argument("--pairs", type=int, default=8, help="Pairs per sample")
    ap.add_argument("--model", choices=("inception", "senet", "resnet"), default="inception",
                    help="Model architecture (default: inception)")
    ap.add_argument("--input-diff", type=str, default="0x00000080", help="Hex input difference (e.g. 0x80)")
    ap.add_argument(
        "--difference",
        type=str,
        default=None,
        help="Optional hex/int difference. If bit-length > plain_bits or --combined-diff set, treated as concatenated (plain_bits + key_bits).",
    )
    ap.add_argument(
        "--combined-diff",
        action="store_true",
        help="Force --difference to be interpreted as combined (plain_bits + key_bits).",
    )
    ap.add_argument("--delta-key-bit", type=int, default=None, help="Optional delta key bit index (0-based). If omitted -> all zeros")
    ap.add_argument("--delta-key-hex", type=str, default=None, help="Optional delta key hex mask (e.g., 0x1). Overrides combined delta_key if provided.")
    ap.add_argument("--stages-rounds", type=str, default="7,7,7", help="Comma list of rounds per stage")
    ap.add_argument("--stages-epochs", type=str, default="3,5,8", help="Comma list of epochs per stage")
    ap.add_argument("--stages-lrs", type=str, default="5e-5,2e-5,5e-6", help="Comma list of learning rates per stage")
    ap.add_argument("--batch-size", type=int, default=5000, help="Train batch size for generator")
    ap.add_argument("--val-batch-size", type=int, default=20000, help="Validation batch size")
    ap.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True, help="Use GPU data generation (CuPy) if available")
    ap.add_argument("--no-eca", action="store_true", help="Use no-ECA ablation variant (applies when building or loading weights)")
    ap.add_argument("--out-dir", type=str, default="staged_runs", help="Output directory for final artifacts")
    ap.add_argument("--save-final", action="store_true", help="Save final full model (.keras) additionally to weights")
    ap.add_argument("--init-model", type=str, default=None, help="Path to full .keras model to start from (takes priority over --init-weights)")
    ap.add_argument("--init-weights", type=str, default=None, help="Path to weights file (.h5/.weights.h5) to initialize model")
    ap.add_argument("--stages-modes", default="freeze,partial,full",
                    help="One of freeze,partial,full per stage; length must match rounds")
    ap.add_argument("--partial-blocks", type=int, default=1,
                    help="Last N pair residual blocks to unfreeze in partial mode")
    ap.add_argument("--train-samples", type=int, default=TRAIN_NUM_SAMPLES)
    ap.add_argument("--val-samples", type=int, default=VAL_NUM_SAMPLES)
    return ap.parse_args()


def _comma_list_to_numbers(s, cast_fn):
    return [cast_fn(x.strip()) for x in s.split(",") if x.strip()]


def build_delta_key(bit_idx, key_bits, use_gpu=False):
    """Build delta key array compatible with the chosen backend (CuPy/NumPy)"""
    lib = cp if (use_gpu and _CUPY_AVAILABLE) else np
    dk = lib.zeros(key_bits, dtype=lib.uint8)
    if bit_idx is not None:
        if bit_idx < 0 or bit_idx >= key_bits:
            raise ValueError(f"--delta-key-bit {bit_idx} out of range (0..{key_bits-1})")
        dk[bit_idx] = 1
    return dk


def _int_to_bits(int_val: int, num_bits: int) -> np.ndarray:
    return np.array([int(b) for b in bin(int_val)[2:].zfill(num_bits)], dtype=np.uint8)


def _parse_delta_key_from_hex(hex_str: str, key_bits: int) -> np.ndarray:
    mask = int(hex_str, 16)
    arr = np.zeros(key_bits, dtype=np.uint8)
    for i in range(key_bits):
        arr[i] = (mask >> i) & 1
    return arr


def _split_combined_difference(diff_int: int, plain_bits: int, key_bits: int):
    if diff_int < 0 or diff_int >= 1 << (plain_bits + key_bits):
        raise ValueError("Combined difference exceeds plaintext and key bit widths")
    bits = _int_to_bits(diff_int, plain_bits + key_bits)
    return diff_int >> key_bits, bits[:plain_bits], bits[plain_bits:]



def compile_fresh(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    )


def build_or_load_initial_model(pairs, plain_bits, init_model_path=None,
                                init_weights_path=None, use_no_eca=False,
                                model_name="inception"):
    if init_model_path:
        # Never restore an old loss or Adam moments.
        return tf.keras.models.load_model(init_model_path, compile=False)
    if use_no_eca and model_name != "inception":
        raise ValueError("--no-eca applies only to inception")
    if model_name == "resnet":
        from resnet import make_multipair_resnet
        factory = make_multipair_resnet
    elif model_name == "senet":
        from YuanWang_RK_SENet import make_multipair_senet
        factory = make_multipair_senet
    else:
        factory = make_model_inception_no_eca if use_no_eca else make_model_inception
    model = factory(pairs=pairs, plain_bits=plain_bits)
    if init_weights_path:
        model.load_weights(init_weights_path)
    return model


def configure_trainable(model, mode, partial_blocks=1):
    # Reset parent flags FIRST: Keras propagates them to nested layers.
    model.trainable = True
    if mode == "full":
        return
    try:
        wrapper = model.get_layer("pair_encoder_td")
        encoder = wrapper.layer
    except (ValueError, AttributeError) as exc:
        raise ValueError("freeze/partial require the current Inception pair_encoder_td; "
                         "use full for unsupported architectures") from exc
    if mode == "freeze":
        wrapper.trainable = False
        return
    if mode != "partial":
        raise ValueError(f"Unknown mode: {mode}")
    import re
    blocks = sorted({int(m.group(1)) for layer in encoder.layers
                     if (m := re.match(r"pair_res_(\d+)_", layer.name))})
    if not 1 <= partial_blocks <= len(blocks):
        raise ValueError(f"--partial-blocks must be 1..{len(blocks)}")
    prefixes = tuple(f"pair_res_{i}_" for i in blocks[-partial_blocks:])
    prefixes += ("pair_eca_final_", "pair_embedding_")
    for layer in encoder.layers:
        layer.trainable = (layer.name.startswith(prefixes)
                           and not isinstance(layer, tf.keras.layers.BatchNormalization))


def validate_schedule(args):
    rounds = _comma_list_to_numbers(args.stages_rounds, int)
    epochs = _comma_list_to_numbers(args.stages_epochs, int)
    lrs = _comma_list_to_numbers(args.stages_lrs, float)
    modes = _comma_list_to_numbers(args.stages_modes, str)
    if not rounds or len({len(rounds), len(epochs), len(lrs), len(modes)}) != 1:
        raise ValueError("rounds, epochs, lrs, and modes must be nonempty and equally long")
    if any(n <= 0 for n in rounds + epochs):
        raise ValueError("Rounds and epochs must be positive")
    if any(not np.isfinite(lr) or lr <= 0 for lr in lrs):
        raise ValueError("Learning rates must be finite and positive")
    if any(m not in ("freeze", "partial", "full") for m in modes):
        raise ValueError("Modes must be freeze, partial, or full")
    if args.model == "resnet" and any(m != "full" for m in modes):
        raise ValueError("ResNet supports full stages only; use --stages-modes full,full,full")
    if not (args.init_model or args.init_weights) and modes[0] != "full":
        raise ValueError("Freeze/partial require a pretrained model; start scratch with full")
    for value in (args.pairs, args.batch_size, args.val_batch_size,
                  args.train_samples, args.val_samples, args.partial_blocks):
        if value <= 0:
            raise ValueError("Pairs, batch sizes, sample counts, partial-blocks must be positive")
    return rounds, epochs, lrs, modes


def run_stage_training(args):
    rounds_list, epochs_list, lrs_list, modes_list = validate_schedule(args)
    # Import cipher dynamically
    import importlib
    cipher_mod = importlib.import_module(f"cipher.{args.cipher}")
    encrypt = cipher_mod.encrypt
    plain_bits = cipher_mod.plain_bits
    key_bits = cipher_mod.key_bits

    # Prepare deltas (supports combined differences with optional override)
    delta_key = None
    # Handle user-provided difference first
    if args.difference:
        try:
            user_diff_int = int(args.difference, 0)
        except Exception:
            user_diff_int = int(args.difference)

        if args.combined_diff or user_diff_int.bit_length() > plain_bits:
            input_diff_int, d_plain_bits, d_key_bits = _split_combined_difference(user_diff_int, plain_bits, key_bits)
            print(f"[difference] Using combined difference (plain+key): 0x{user_diff_int:X}")
            print(f"            delta_plain HW={int(d_plain_bits.sum())} bits={[i for i,v in enumerate(d_plain_bits) if v]}")
            print(f"            delta_key  HW={int(d_key_bits.sum())} bits={[i for i,v in enumerate(d_key_bits) if v]}")
            delta_plain = integer_to_binary_array(input_diff_int, plain_bits)
            delta_key = d_key_bits.astype(np.uint8)
            # Overrides
            if args.delta_key_bit is not None:
                mb = int(args.delta_key_bit)
                if mb < 0 or mb >= key_bits:
                    raise ValueError(f"--delta-key-bit out of range (0..{key_bits-1})")
                delta_key = np.zeros(key_bits, dtype=np.uint8)
                delta_key[mb] = 1
                print(f"[override] Replacing combined delta_key with manual bit={mb}")
            elif args.delta_key_hex:
                delta_key = _parse_delta_key_from_hex(args.delta_key_hex, key_bits)
                print("[override] Replacing combined delta_key with provided --delta-key-hex mask")
        else:
            # Plain-only
            input_diff_int = user_diff_int
            print(f"[difference] Using provided plain input difference: 0x{input_diff_int:X}")
            delta_plain = integer_to_binary_array(input_diff_int, plain_bits)
            # Determine delta_key from overrides or default to zeros
            if args.delta_key_bit is not None:
                delta_key = _parse_delta_key_from_hex("0x0", key_bits)  # start zeros
                mb = int(args.delta_key_bit)
                if mb < 0 or mb >= key_bits:
                    raise ValueError(f"--delta-key-bit out of range (0..{key_bits-1})")
                delta_key[mb] = 1
            elif args.delta_key_hex:
                delta_key = _parse_delta_key_from_hex(args.delta_key_hex, key_bits)
            else:
                delta_key = np.zeros(key_bits, dtype=np.uint8)
                print("[info] No delta-key override provided; using delta_key = 0 (no related-key).")
    else:
        # Backward-compatible path using --input-diff and delta-key-bit
        input_diff_int = int(args.input_diff, 16)
        delta_plain = integer_to_binary_array(input_diff_int, plain_bits)
        if args.delta_key_bit is not None:
            delta_key = _parse_delta_key_from_hex("0x0", key_bits)
            mb = int(args.delta_key_bit)
            if mb < 0 or mb >= key_bits:
                raise ValueError(f"--delta-key-bit out of range (0..{key_bits-1})")
            delta_key[mb] = 1
        elif args.delta_key_hex:
            delta_key = _parse_delta_key_from_hex(args.delta_key_hex, key_bits)
        else:
            delta_key = np.zeros(key_bits, dtype=np.uint8)
            print("[info] No --delta-key-bit/--delta-key-hex provided; using delta_key = 0 (no related-key).")

    # Ensure delta_plain uses correct backend
    if args.use_gpu and _CUPY_AVAILABLE:
        if isinstance(delta_plain, np.ndarray):
            delta_plain = cp.asarray(delta_plain)
        if isinstance(delta_key, np.ndarray):
            delta_key = cp.asarray(delta_key)
    else:
        if hasattr(delta_plain, '__cuda_array_interface__'):
            delta_plain = cp.asnumpy(delta_plain)
        if hasattr(delta_key, '__cuda_array_interface__'):
            delta_key = cp.asnumpy(delta_key)

    n_stages = len(rounds_list)
    os.makedirs(args.out_dir, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Build or load initial model
    model = build_or_load_initial_model(
        pairs=args.pairs, 
        plain_bits=plain_bits,
        init_model_path=args.init_model,
        init_weights_path=args.init_weights,
        use_no_eca=bool(args.no_eca),
        model_name=args.model,
    )

    expected_shape = (None, args.pairs * 3 * plain_bits)
    if tuple(model.input_shape) != expected_shape:
        raise ValueError(f"Expected flat generator input {expected_shape}, got {model.input_shape}")
    # Check all requested phase configurations before starting expensive training.
    for mode in modes_list:
        configure_trainable(model, mode, args.partial_blocks)
    all_stage_histories = []
    for idx in range(n_stages):
        stage_nr = rounds_list[idx]
        stage_epochs = epochs_list[idx]
        stage_lr = lrs_list[idx]
        mode = modes_list[idx]
        stage_train_samples = args.train_samples
        stage_val_samples = args.val_samples
        configure_trainable(model, mode, args.partial_blocks)
        compile_fresh(model, stage_lr)
        print(f"\n[Stage {idx+1}/{n_stages}] round={stage_nr} mode={mode} "
              f"lr={stage_lr} epochs={stage_epochs}; fresh Adam + BCE")
        print("Trainable parameters:", sum(int(np.prod(w.shape)) for w in model.trainable_weights))

        gen_train = NDCMultiPairGenerator(
            encryption_function=encrypt,
            plain_bits=plain_bits, key_bits=key_bits, nr=stage_nr,
            delta_state=delta_plain, delta_key=delta_key,
            pairs=args.pairs,
            n_samples=stage_train_samples, batch_size=args.batch_size,
            use_gpu=args.use_gpu,
        )
        gen_val = NDCMultiPairGenerator(
            encryption_function=encrypt,
            plain_bits=plain_bits, key_bits=key_bits, nr=stage_nr,
            delta_state=delta_plain, delta_key=delta_key,
            pairs=args.pairs,
            n_samples=stage_val_samples, batch_size=args.val_batch_size,
            use_gpu=args.use_gpu,
        )

        phase_dir = Path(args.out_dir) / run_id / f"stage{idx+1}_{stage_nr}r_{mode}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        best_path = phase_dir / "best.weights.h5"
        stage_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss",
                                               save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.CSVLogger(str(phase_dir / "training.csv")),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        hist = model.fit(gen_train, validation_data=gen_val, epochs=stage_epochs,
                         callbacks=stage_callbacks, verbose=1)
        values = hist.history.get("val_loss", [])
        if (not values or not all(np.isfinite(v) for v in values)
                or not all(np.isfinite(v) for v in hist.history.get("loss", []))
                or not best_path.is_file()):
            raise RuntimeError(f"Stage {idx+1} has no valid checkpoint or non-finite loss")
        # The next phase starts from this phase's best validation checkpoint.
        model.load_weights(str(best_path))
        all_stage_histories.append({
            "stage": idx+1, "rounds": stage_nr, "mode": mode, "lr": stage_lr,
            "optimizer_reset": True, "loss": "binary_crossentropy",
            "best_checkpoint": str(best_path), "history": hist.history,
        })
        with open(Path(args.out_dir) / f"{run_id}_history.json", "w", encoding="utf-8") as f:
            json.dump(all_stage_histories, f, indent=2)

        # Save intermediate weights
        suffix = "_noECA" if args.no_eca else ""
        weights_path = os.path.join(args.out_dir, f"{args.cipher}_{args.model}_k{args.pairs}_stage{idx+1}_{stage_nr}r_best{suffix}.weights.h5")
        model.save_weights(weights_path)
        print(f"Saved stage {idx+1} weights: {weights_path}")

    # Final artifacts
    suffix = "_noECA" if args.no_eca else ""
    final_weights = os.path.join(args.out_dir, f"{args.cipher}_{args.model}_k{args.pairs}_staged_final{suffix}.weights.h5")
    model.save_weights(final_weights)
    print(f"Saved final staged weights: {final_weights}")

    if args.save_final:
        final_model_path = os.path.join(args.out_dir, f"{args.cipher}_{args.model}_k{args.pairs}_staged_final{suffix}.keras")
        model.save(final_model_path)
        print(f"Saved final staged full model: {final_model_path}")

    # Save combined history
    history_path = os.path.join(args.out_dir, f"{args.cipher}_{args.model}_k{args.pairs}_staged_history{suffix}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(all_stage_histories, f, ensure_ascii=False, indent=2)
    print(f"Saved staged training history: {history_path}")


if __name__ == "__main__":
    args = parse_args()
    run_stage_training(args)
