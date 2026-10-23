"""Train a neural distinguisher FROM SCRATCH only.

Requires train_nets.py, RKmcp.py, make_data_train.py and cipher/ beside this file.
Validation is freshly generated each epoch (not a cached fixed dataset).
"""
import argparse
import datetime as dt
import importlib
import json
from pathlib import Path
import time
import uuid

from train_nets import (
    binary_label_generator, build_callbacks, evaluate_model, train_model, write_json,
)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    p.add_argument("--cipher", default="present80", help="Module name under cipher/")
    p.add_argument("--rounds", "-r", type=positive_int, required=True)
    p.add_argument("--pairs", "-p", type=positive_int, default=16)
    p.add_argument("--input-diff", type=lambda value: int(value, 0), required=True,
                   help="Plaintext difference as an integer, e.g. 0x0040")
    p.add_argument("--delta-key-bit", type=int, required=True,
                   help="Zero-based array index; index 0 is the MSB of the key bit vector")
    p.add_argument("--model", choices=("inception", "senet", "resnet"), default="inception")
    p.add_argument("--epochs", type=positive_int, default=30)
    p.add_argument("--num-samples-train", type=positive_int, default=1_000_000,
                   help="Fresh training samples per epoch")
    p.add_argument("--num-samples-val", type=positive_int, default=250_000)
    p.add_argument("--batch-size", type=positive_int, default=2048)
    p.add_argument("--val-batch-size", type=positive_int, default=2048)
    p.add_argument("--loss", choices=("bce", "mse"), default="bce")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=("plateau", "constant"), default="plateau")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--lr-patience", type=positive_int, default=3)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience; 0 disables early stopping")
    p.add_argument("--reg-param", type=float, default=1e-4,
                   help="Inception L2 coefficient; architecture default retained")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="Inception dropout; architecture default retained")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-backend", choices=("numpy", "cupy"), default="numpy",
                   help="Generator RNG backend; independent of TensorFlow's GPU use")
    p.add_argument("--encrypt-backend", choices=("numpy", "cupy", "auto"), default="numpy",
                   help="Cipher input backend; use the backend supported by your cipher")
    p.add_argument("--test-samples", type=positive_int, default=100_000)
    p.add_argument("--test-repeats", type=positive_int, default=5)
    p.add_argument("--test-batch-size", type=positive_int, default=2048)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    args = p.parse_args(argv)
    import math
    if not (math.isfinite(args.lr) and math.isfinite(args.min_lr)
            and 0 < args.min_lr <= args.lr):
        p.error("require finite 0 < --min-lr <= --lr")
    if args.patience < 0:
        p.error("--patience must be >= 0")
    if (args.lr_schedule == "plateau" and args.patience
            and args.patience <= args.lr_patience):
        p.error("--patience must exceed --lr-patience, or be 0")
    if not math.isfinite(args.reg_param) or args.reg_param < 0 or not 0 <= args.dropout < 1:
        p.error("require finite --reg-param >= 0 and 0 <= --dropout < 1")
    if args.model != "inception" and (args.reg_param != 1e-4 or args.dropout != 0.2):
        p.error("--reg-param and --dropout overrides apply only to inception")
    if not 0 <= args.seed < 2**32:
        p.error("--seed must be in [0, 2**32)")
    return args


def run(argv=None):
    args = parse_args(argv)
    import numpy as np
    import tensorflow as tf
    from make_data_train import NDCMultiPairGenerator

    # Configure TF before any generator can initialize CUDA allocations.
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(args.seed)
    if args.data_backend == "cupy" or args.encrypt_backend == "cupy":
        import cupy as cp
        cp.random.seed(args.seed)

    cipher = importlib.import_module(f"cipher.{args.cipher}")
    plain_bits, key_bits = int(cipher.plain_bits), int(cipher.key_bits)
    if not 0 <= args.input_diff < 1 << plain_bits:
        raise ValueError(f"--input-diff must fit in {plain_bits} bits")
    if not 0 <= args.delta_key_bit < key_bits:
        raise ValueError(f"--delta-key-bit must be in [0, {key_bits - 1}]")
    delta_key = np.zeros(key_bits, dtype=np.uint8)
    delta_key[args.delta_key_bit] = 1

    def make_generator(samples, batch_size):
        return binary_label_generator(NDCMultiPairGenerator(
            encryption_function=cipher.encrypt, plain_bits=plain_bits,
            key_bits=key_bits, nr=args.rounds,
            delta_state=args.input_diff, delta_key=delta_key,
            pairs=args.pairs, n_samples=samples, batch_size=batch_size,
            use_gpu=args.data_backend == "cupy", encrypt_backend=args.encrypt_backend))

    if args.model == "inception":
        from RKmcp import make_model_inception
        model = make_model_inception(pairs=args.pairs, plain_bits=plain_bits,
                                    reg_param=args.reg_param, dropout=args.dropout)
    elif args.model == "resnet":
        from resnet import make_multipair_resnet
        model = make_multipair_resnet(pairs=args.pairs, plain_bits=plain_bits)
    else:
        from YuanWang_RK_SENet import make_multipair_senet
        model = make_multipair_senet(pairs=args.pairs, plain_bits=plain_bits)
    expected_shape = (None, args.pairs * 3 * plain_bits)
    if tuple(model.input_shape) != expected_shape:
        raise ValueError(f"Model input {model.input_shape} != generator input {expected_shape}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr, amsgrad=True),
        loss="binary_crossentropy" if args.loss == "bce" else "mse",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.BinaryCrossentropy(name="bce")])

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = args.out_dir / args.cipher / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(plain_bits=plain_bits, key_bits=key_bits, run_id=run_id,
                  initialization="scratch", tensorflow_version=tf.__version__,
                  validation="fresh samples each epoch", monitor="val_bce",
                  delta_key_bits=delta_key.tolist(),
                  delta_key_hex=hex(1 << (key_bits - 1 - args.delta_key_bit)))
    write_json(run_dir / "config.json", config)
    (run_dir / "architecture.json").write_text(model.to_json(), encoding="utf-8")
    print(json.dumps(config, indent=2))
    print(f"Artifacts: {run_dir.resolve()}")

    started = time.perf_counter()
    history = train_model(
        model, make_generator(args.num_samples_train, args.batch_size),
        make_generator(args.num_samples_val, args.val_batch_size),
        epochs=args.epochs, callbacks=build_callbacks(run_dir, args))
    seconds = time.perf_counter() - started
    write_json(run_dir / "history.json",
               {key: [float(v) for v in values] for key, values in history.history.items()})
    losses = np.asarray(history.history.get("loss", []), dtype=float)
    if not losses.size or not np.all(np.isfinite(losses)):
        raise RuntimeError(f"Non-finite training loss; inspect {run_dir}")
    best_path = run_dir / "best.weights.h5"
    if not best_path.is_file():
        raise RuntimeError(f"No valid best checkpoint saved; inspect {run_dir}")
    # Always evaluate the saved best checkpoint, regardless of early stopping.
    model.load_weights(best_path)
    write_json(run_dir / "summary.json", {
        "training_seconds": seconds, "epochs_completed": len(history.epoch),
        "selected_checkpoint": "best.weights.h5", "monitor": "val_bce"})
    if not args.skip_eval:
        # Pass the integer difference through make_generator: no CuPy -> NumPy cast.
        stats = evaluate_model(model, make_generator, samples=args.test_samples,
                               batch_size=args.test_batch_size, repeats=args.test_repeats)
        write_json(run_dir / "evaluation.json", stats)
        print("Test means:", stats["mean"])
    print(f"Done: {run_dir.resolve()}")


if __name__ == "__main__":
    run()
