# MNIST experiment tracking for Assignment 3

This project trains four fully connected neural networks on the MNIST handwritten **digits** dataset and tracks the experiments using TensorFlow/Keras autologging in MLflow and DagsHub.

## Files

- `mnist_experiments.py`: data preparation, all four experiments, automatic logging, comparison export and final test evaluation.
- `requirements.txt`: package versions used for the experiments.
- `results/comparison.csv`: hyperparameters and actual final-epoch training/validation results.
- `results/results.json`: results with MLflow run identifiers.
- `results/metadata.json`: selected run, test results, tracking URI and software versions.
- Each run folder includes epoch history, configuration, metrics and learning curves. Trained models are stored as MLflow artifacts on DagsHub rather than in Git.

## Setup and run

Use Python 3.12 in a virtual environment. From this project folder:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS or Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python mnist_experiments.py --owner kaps-is-it
```

The official DagsHub client opens its authentication flow if needed. Sign in to the account that owns the repository. Do not put tokens or passwords in the Python file, report or Git repository. The repository is created by `dagshub.init()` if it does not already exist.

If a connection interrupts the pipeline after a run has finished, repeat the command with `--resume` and the same output directory. It reuses completed runs only after checking their logged metrics and saved model; unfinished configurations are trained again. On Windows, the script uses UTF-8 output so MLflow's run-link messages also work when output is redirected to a log file.

To rehearse without DagsHub:

```bash
python mnist_experiments.py --local --output-dir local_results
mlflow ui --backend-store-uri local_results/mlruns
```

An optional `--smoke-test` runs only two epochs on a small subset to check the pipeline. Smoke-test numbers are not the assignment results; use the full command above to reproduce the submitted experiments. Choose a separate experiment name for rehearsals, for example `--experiment Pipeline_Test`.

## Experiment design

| Run | Hidden Dense nodes | Optimizer | Batch size | Epochs | Change |
| --- | --- | --- | --- | --- | --- |
| 01 baseline | 128 | Adam | 64 | 50 | Required baseline |
| 02 wider | 256 | Adam | 64 | 50 | Width only |
| 03 deeper | 128, 64 | Adam | 64 | 50 | Add a hidden layer |
| 04 optimizer | 128 | RMSprop | 64 | 50 | Optimizer only |

Every model uses Flatten, ReLU in each hidden Dense layer, a 10-node softmax output and sparse categorical cross-entropy. The learning rate is 0.001. No early stopping is used: every submitted run completes all 50 epochs.

The official 60,000 training images are split into 48,000 training and 12,000 validation examples, stratified by digit with seed 42. Pixel values are converted to float32 and divided by 255. All four runs use exactly the same split. The original 10,000 test images are kept separate and are evaluated only for the selected model after selection. Seeding and deterministic operations reduce variation; hardware and software changes can still affect reproducibility.

## Logging and comparison

`mlflow.tensorflow.autolog()` records epoch metrics, optimizer configuration, the model summary, TensorBoard logs and the trained model. Explicit parameters also record hidden-layer counts, each layer's node count, optimizer, batch size, epochs, seed, data sizes and split checksum. Custom final metrics have clear names: `training_accuracy`, `validation_accuracy`, `training_loss`, and `validation_loss`. The standard automatically logged metrics `accuracy`, `val_accuracy`, `loss`, and `val_loss` retain the full epoch curves.

The comparison uses epoch 50 consistently. The rule is highest **final validation accuracy**, with lower final validation loss as the tie-breaker. Peak validation accuracy is recorded as extra context, but it is not substituted into the final-epoch table. Only the selected final model is evaluated on the official test set. One seed per configuration gives a useful assignment comparison, not proof that an architecture is always superior.

After each run the script checks that MLflow reports FINISHED, all 50 epoch steps are present for each standard metric, required parameters exist and the autologged model artifact exists. It writes the results locally as well as to DagsHub.

## Links

- GitHub: https://github.com/kaps-is-it/cosc31000-assignment3-mnist-mlflow
- DagsHub: https://dagshub.com/kaps-is-it/cosc31000-assignment3-mnist-mlflow
- MLflow Tracking URI: https://dagshub.com/kaps-is-it/cosc31000-assignment3-mnist-mlflow.mlflow

## References

- Keras MNIST dataset: https://keras.io/api/datasets/mnist/
- MLflow TensorFlow integration: https://mlflow.org/docs/latest/ml/deep-learning/tensorflow/
- DagsHub MLflow setup: https://dagshub.com/docs/integration_guide/mlflow_tracking/
