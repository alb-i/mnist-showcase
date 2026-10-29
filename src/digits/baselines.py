"""Baselines on flattened pixels (plan 7.1, experiment E1): logistic regression and k-NN.

    micromamba run -n mnist-showcase python -m digits.baselines            # both
    micromamba run -n mnist-showcase python -m digits.baselines --only knn

Each model is one run in $RUNS_DIR/mnist/ (digits.runlog): one metrics record per split
(val, test) with acc, macro_f1, loss and predictions/{val,test}.parquet. No step loop and no
weights in the PyTorch sense, so no weight_delta here.
"""

import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from digits.data import load_dataset
from digits.evaluate import classification_metrics, predictions_frame
from digits.runlog import RunLogger

BASELINES = {
    "logreg": {"model": "logreg", "params": {"C": 1.0, "max_iter": 200}},
    "knn": {"model": "knn", "params": {"n_neighbors": 3}},
}


def make_model(config: dict):
    if config["model"] == "logreg":
        return LogisticRegression(**config["params"])
    if config["model"] == "knn":
        return KNeighborsClassifier(**config["params"], n_jobs=-1)
    raise ValueError(f"unknown baseline {config['model']}")


def flatten(images: np.ndarray) -> np.ndarray:
    """(N, 28, 28) uint8 -> (N, 784) float32 in [0, 1]."""
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def run_baseline(config: dict, dataset: dict, root=None) -> dict[str, dict]:
    """Fit on train, evaluate on val and test; returns the metrics per split."""
    config = {**config, "features": "pixels/255", "n_train": len(dataset["train"][2])}
    results = {}
    with RunLogger(config, root=root) as run:
        _, images, labels = dataset["train"]
        model = make_model(config)
        t = time.time()
        model.fit(flatten(images), labels)
        fit_s = time.time() - t
        for split in ("val", "test"):
            ids, images, labels = dataset[split]
            t = time.time()
            probs = model.predict_proba(flatten(images))
            predict_s = time.time() - t
            frame = predictions_frame(ids, labels, probs)
            frame.to_parquet(run.dir / "predictions" / f"{split}.parquet", index=False)
            metrics = classification_metrics(labels, frame["pred"].to_numpy())
            results[split] = metrics
            run.log(
                step=0,
                split=split,
                loss=float(frame["loss"].mean()),
                acc=metrics["acc"],
                macro_f1=metrics["macro_f1"],
                fit_s=round(fit_s, 2),
                predict_s=round(predict_s, 2),
            )
        print(
            f"{run.run_id}: val acc {results['val']['acc']:.4f}, "
            f"test acc {results['test']['acc']:.4f} (fit {fit_s:.1f} s)"
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=list(BASELINES), action="append")
    args = parser.parse_args(argv)
    dataset = load_dataset()
    for name in args.only or list(BASELINES):
        run_baseline(BASELINES[name], dataset)


if __name__ == "__main__":
    main()
