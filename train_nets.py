"""Small helpers for scratch training. Data is streamed by the generator.

Replaces the old chunk loop; no callback objects are created at import time.
"""
import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def binary_label_generator(generator):
    """Adapt Sequence batches to sigmoid output (batch, 1), without caching data."""
    import numpy as np
    from tensorflow.keras.utils import Sequence

    if getattr(generator, "_binary_column_labels", False):
        return generator

    class BinaryLabelSequence(Sequence):
        _binary_column_labels = True

        def __init__(self, source):
            super().__init__()
            self.source = source

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            batch = self.source[index]
            if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
                raise ValueError("Expected (inputs, labels[, sample_weight])")
            x, y = batch[:2]
            y = np.asarray(y, dtype=np.float32)
            if y.ndim == 1:
                y = y[:, None]
            if y.ndim != 2 or y.shape[1] != 1:
                raise ValueError(f"Binary labels must have shape (batch,) or (batch, 1); got {y.shape}")
            if y.shape[0] != x.shape[0]:
                raise ValueError("Inputs and labels have different batch sizes")
            return (x, y) if len(batch) == 2 else (x, y, batch[2])

        def on_epoch_end(self):
            hook = getattr(self.source, "on_epoch_end", None)
            if hook is not None:
                hook()

        def on_epoch_begin(self):
            hook = getattr(self.source, "on_epoch_begin", None)
            if hook is not None:
                hook()

    return BinaryLabelSequence(generator)


def build_callbacks(run_dir, args):
    import tensorflow as tf

    c = tf.keras.callbacks
    callbacks = [
        c.TerminateOnNaN(),
        c.ModelCheckpoint(str(run_dir / "best.weights.h5"), monitor="val_bce",
                          mode="min", save_best_only=True, save_weights_only=True),
        c.CSVLogger(str(run_dir / "training.csv")),
    ]
    if args.lr_schedule == "plateau":
        callbacks.append(c.ReduceLROnPlateau(
            monitor="val_bce", mode="min", factor=0.5,
            patience=args.lr_patience, min_lr=args.min_lr, verbose=1))
    if args.patience:
        callbacks.append(c.EarlyStopping(
            monitor="val_bce", mode="min", patience=args.patience,
            restore_best_weights=True, verbose=1))
    return callbacks


def train_model(model, train_generator, validation_generator, *, epochs, callbacks):
    """One fit call: Keras manages batch/epoch callbacks and streaming."""
    return model.fit(binary_label_generator(train_generator),
                     validation_data=binary_label_generator(validation_generator),
                     epochs=epochs, callbacks=callbacks, verbose=2)


def evaluate_model(model, make_generator, *, samples, batch_size, repeats):
    """Fresh test sets; SD measures test-sampling variation, not training seeds."""
    import numpy as np

    runs = []
    for index in range(repeats):
        values = model.evaluate(binary_label_generator(make_generator(samples, batch_size)),
                                verbose=0, return_dict=True)
        values = {key: float(value) for key, value in values.items()}
        runs.append(values)
        print(f"Test {index + 1}/{repeats}: {values}")
    return {
        "test_samples_per_repeat": samples,
        "repeats": repeats,
        "runs": runs,
        "mean": {key: float(np.mean([r[key] for r in runs])) for key in runs[0]},
        "std": {key: float(np.std([r[key] for r in runs])) for key in runs[0]},
        "std_note": "Across fresh test sets for one trained model, not training seeds.",
    }
