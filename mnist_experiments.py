"""COSC 31000 Assignment 3: four MNIST experiments tracked with MLflow.

Run: python mnist_experiments.py --owner YOUR_DAGSHUB_USERNAME
Use --local for an offline rehearsal. See README.md for setup.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Keep downloaded datasets and temporary files inside this project.
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


SEED = 42
REPO_NAME = "cosc31000-assignment3-mnist-mlflow"
EXPERIMENT_NAME = "Assignment3_MNIST"

# Each additional experiment changes one factor from the baseline.
CONFIGS = [
    {"name": "01_baseline_128_adam_bs64_ep50", "nodes": [128],
     "optimizer": "Adam", "batch_size": 64, "epochs": 50},
    {"name": "02_wider_256_adam_bs64_ep50", "nodes": [256],
     "optimizer": "Adam", "batch_size": 64, "epochs": 50},
    {"name": "03_deeper_128_64_adam_bs64_ep50", "nodes": [128, 64],
     "optimizer": "Adam", "batch_size": 64, "epochs": 50},
    {"name": "04_optimizer_128_rmsprop_bs64_ep50", "nodes": [128],
     "optimizer": "RMSprop", "batch_size": 64, "epochs": 50},
]


def load_data():
    """Keep the official test set separate from all model selection."""
    (images, labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    indices = np.arange(len(images))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=labels
    )
    x_train = images[train_idx].astype("float32") / 255.0
    x_val = images[val_idx].astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train, y_val = labels[train_idx], labels[val_idx]
    assert x_train.shape == (48000, 28, 28)
    assert x_val.shape == (12000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert 0 <= x_train.min() <= x_train.max() <= 1
    split_hash = hashlib.sha256(train_idx.tobytes() + val_idx.tobytes()).hexdigest()
    return x_train, y_train, x_val, y_val, x_test, y_test, split_hash


def build_model(config):
    model = tf.keras.Sequential(name="mnist_dense_classifier")
    model.add(tf.keras.Input(shape=(28, 28), name="digit_image"))
    model.add(tf.keras.layers.Flatten(name="flatten"))
    for number, nodes in enumerate(config["nodes"], start=1):
        model.add(tf.keras.layers.Dense(nodes, activation="relu", name=f"hidden_{number}"))
    model.add(tf.keras.layers.Dense(10, activation="softmax", name="digit_probabilities"))
    optimizer_class = getattr(tf.keras.optimizers, config["optimizer"])
    model.compile(
        optimizer=optimizer_class(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, destination, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for axis, metric, label in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        axis.plot(epochs, history[metric], label="Training")
        axis.plot(epochs, history["val_" + metric], label="Validation")
        axis.set(xlabel="Epoch", ylabel=label)
        axis.legend()
        axis.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def configure_tracking(args):
    if args.local:
        uri = (Path(args.output_dir).resolve() / "mlruns").as_uri()
        mlflow.set_tracking_uri(uri)
    else:
        if not args.owner:
            raise ValueError("Provide --owner with your DagsHub username, or use --local.")
        import dagshub
        dagshub.init(repo_owner=args.owner, repo_name=args.repo, mlflow=True)
        uri = f"https://dagshub.com/{args.owner}/{args.repo}.mlflow"
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(args.experiment)
    # Autolog records per-epoch accuracy/loss, optimizer details, model summary,
    # TensorBoard logs and the trained model. Custom parameters below make the
    # assignment's architecture settings easy to compare in the interface.
    mlflow.tensorflow.autolog(log_models=True, log_datasets=False,
                              log_every_epoch=True, checkpoint=False)
    return uri


class Progress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"epoch {epoch + 1:02d}: accuracy={logs.get('accuracy', 0):.4f} "
              f"val_accuracy={logs.get('val_accuracy', 0):.4f} "
              f"loss={logs.get('loss', 0):.4f} val_loss={logs.get('val_loss', 0):.4f}",
              flush=True)


def verify_run(run_id, config, epoch_count):
    """Check recorded data rather than assuming that autologging succeeded."""
    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED", run.info.status
    assert run.data.params["hidden_layers"] == str(len(config["nodes"]))
    assert run.data.params["optimizer"] == config["optimizer"]
    for metric in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        points = client.get_metric_history(run_id, metric)
        assert len({p.step for p in points}) == epoch_count, (metric, len(points))
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    assert "model" in artifacts, artifacts
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=os.getenv("DAGSHUB_USER"))
    parser.add_argument("--repo", default=REPO_NAME)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="One short rehearsal; never used for reported results.")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    uri = configure_tracking(args)
    x_train, y_train, x_val, y_val, x_test, y_test, split_hash = load_data()
    if args.smoke_test:
        x_train, y_train = x_train[:1024], y_train[:1024]
        x_val, y_val = x_val[:256], y_val[:256]
    configs = [dict(CONFIGS[0], epochs=2, name="smoke_test")] if args.smoke_test else CONFIGS
    results = []
    for config in configs:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = build_model(config)
        run_dir = output / config["name"]
        run_dir.mkdir(exist_ok=True)
        print(f"\nStarting {config['name']}", flush=True)
        start = time.perf_counter()
        with mlflow.start_run(run_name=config["name"]) as active:
            run_id = active.info.run_id
            mlflow.set_tags({"assignment": "COSC 31000 Assignment 3", "dataset": "MNIST digits",
                             "selection_rule": "final val_accuracy descending; val_loss ascending",
                             "smoke_test": str(args.smoke_test).lower()})
            params = {
                "hidden_layers": len(config["nodes"]),
                "hidden_nodes": json.dumps(config["nodes"]),
                "optimizer": config["optimizer"], "batch_size": config["batch_size"],
                "epochs": config["epochs"], "learning_rate": 0.001,
                "activation": "relu", "output_activation": "softmax",
                "seed": SEED, "validation_fraction": 0.2,
                "train_samples": len(x_train), "validation_samples": len(x_val),
                "test_samples": len(x_test), "split_sha256": split_hash,
                "parameter_count": model.count_params(),
            }
            for i, nodes in enumerate(config["nodes"], start=1):
                params[f"hidden_{i}_nodes"] = nodes
            mlflow.log_params(params)
            history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                                batch_size=config["batch_size"], epochs=config["epochs"],
                                shuffle=True, callbacks=[Progress()], verbose=0)
            h = history.history
            assert len(h["loss"]) == config["epochs"]
            assert all(np.isfinite(values).all() for values in h.values())
            final = {"training_accuracy": float(h["accuracy"][-1]),
                     "validation_accuracy": float(h["val_accuracy"][-1]),
                     "training_loss": float(h["loss"][-1]),
                     "validation_loss": float(h["val_loss"][-1]),
                     "best_validation_accuracy": float(max(h["val_accuracy"])),
                     "best_validation_epoch": int(np.argmax(h["val_accuracy"]) + 1),
                     "training_seconds": time.perf_counter() - start}
            mlflow.log_metrics(final)
            pd.DataFrame(h, index=np.arange(1, len(h["loss"]) + 1)).rename_axis("epoch").to_csv(run_dir / "history.csv")
            plot_history(h, run_dir / "learning_curves.png", config["name"])
            (run_dir / "config.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
            (run_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
            model.save(run_dir / "final_model.keras")
            mlflow.log_artifacts(str(run_dir), artifact_path="assignment_results")
            if Path(__file__).is_file():
                mlflow.log_artifact(__file__, artifact_path="source")
        verify_run(run_id, config, config["epochs"])
        results.append({"run_name": config["name"], "run_id": run_id,
                        "hidden_layers": len(config["nodes"]), "hidden_nodes": config["nodes"],
                        "optimizer": config["optimizer"], "batch_size": config["batch_size"],
                        "epochs": config["epochs"], **final})
        (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        pd.DataFrame(results).to_csv(output / "comparison.csv", index=False)

    ranked = sorted(results, key=lambda r: (-r["validation_accuracy"], r["validation_loss"]))
    best = ranked[0]
    # Only the winning final model sees the official test set, after selection.
    mlflow.tensorflow.autolog(disable=True)
    selected_model = tf.keras.models.load_model(output / best["run_name"] / "final_model.keras")
    test_loss, test_accuracy = selected_model.evaluate(x_test, y_test, batch_size=256, verbose=0)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected_best_model", "true")
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
    metadata = {"tracking_uri": uri, "experiment_name": args.experiment,
                "best_run": best, "test_accuracy": test_accuracy, "test_loss": test_loss,
                "selection_rule": "Highest final validation accuracy; lowest final validation loss breaks ties",
                "python_version": platform.python_version(), "tensorflow_version": tf.__version__,
                "keras_version": tf.keras.__version__, "mlflow_version": mlflow.__version__,
                "smoke_test": args.smoke_test}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    print(f"\nSelected: {best['run_name']}; test accuracy: {test_accuracy:.4%}", flush=True)
    print(f"Tracking URI: {uri}", flush=True)


if __name__ == "__main__":
    main()
