"""Train a baseline pass/fail classifier on the UCI SECOM dataset.

Accuracy is not the goal here -- this is the simplest model that can sit
behind a real API. The infrastructure around it is what's being graded.

Data is split 60/20/20 into train/val/test. The model is trained epoch by
epoch via SGDClassifier.partial_fit so we can log train/val loss after
every pass; the test set is never touched until final evaluation.
"""
import csv
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

TOP_N_FEATURES = 20
EPOCHS = 200
MODEL_PATH = "model.pkl"
DATASET_PATH = "dataset/secom.csv"
LOGS_DIR = "logs"
METRICS_DIR = "metrics"


def load_and_split():
    raw = pd.read_csv(DATASET_PATH)
    X = raw.drop(columns=["class", "timestamp"])
    y = raw["class"]

    # 60% train / 20% val / 20% test, stratified so each split keeps the
    # same pass/fail ratio as the full (imbalanced) dataset.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(X_train, X_val, X_test):
    # Fit imputer, feature selection, and scaler using TRAIN data only,
    # so no information from val/test leaks into preprocessing decisions.
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    top_features = X_train_imputed.var().sort_values(ascending=False).head(TOP_N_FEATURES).index.tolist()
    X_train_top = X_train_imputed[top_features]
    X_val_top = X_val_imputed[top_features]
    X_test_top = X_test_imputed[top_features]

    # SGDClassifier (unlike lbfgs-based LogisticRegression) is very sensitive
    # to feature scale, since it takes literal gradient steps sized by raw
    # feature magnitude. Scaling is required here, not just nice-to-have.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_top), columns=top_features)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_top), columns=top_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_top), columns=top_features)

    return X_train_scaled, X_val_scaled, X_test_scaled, top_features, scaler


def train_with_logging(X_train, y_train, X_val, y_val):
    classes = np.array(sorted(y_train.unique()))
    # partial_fit doesn't accept class_weight="balanced" directly (it can't
    # recompute class frequencies from a single batch) -- so we compute the
    # balanced weights once, up front, from the full training set instead.
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    model = SGDClassifier(loss="log_loss", class_weight=class_weight, random_state=42)
    rows = []
    for epoch in range(1, EPOCHS + 1):
        # One partial_fit call per epoch = one full-batch gradient step.
        model.partial_fit(X_train, y_train, classes=classes)
        train_proba = model.predict_proba(X_train)
        val_proba = model.predict_proba(X_val)
        rows.append({
            "epoch": epoch,
            "train_loss": log_loss(y_train, train_proba, labels=model.classes_),
            "val_loss": log_loss(y_val, val_proba, labels=model.classes_),
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
            "val_accuracy": accuracy_score(y_val, model.predict(X_val)),
        })

    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(f"{LOGS_DIR}/training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in rows], label="train loss")
    plt.plot(epochs, [r["val_loss"] for r in rows], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("log loss")
    plt.title("Training vs validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{LOGS_DIR}/loss_curve.png")
    plt.close()

    print(f"Final epoch -- train_loss={rows[-1]['train_loss']:.3f}  val_loss={rows[-1]['val_loss']:.3f}")
    print(f"Saved training log + loss curve to {LOGS_DIR}/")
    return model


def evaluate_and_save(model, X_test, y_test):
    y_pred = model.predict(X_test)
    fail_idx = list(model.classes_).index(1)
    y_proba = model.predict_proba(X_test)[:, fail_idx]

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": {
            "labels": model.classes_.tolist(),
            "matrix": cm.tolist(),
        },
        "test_set_size": len(y_test),
        "test_set_fail_count": int((y_test == 1).sum()),
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(f"{METRICS_DIR}/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test metrics: precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
          f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}  pr_auc={metrics['pr_auc']:.3f}")
    print(f"Saved evaluation metrics to {METRICS_DIR}/evaluation_metrics.json")


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
    X_train, X_val, X_test, top_features, scaler = preprocess(X_train, X_val, X_test)

    model = train_with_logging(X_train, y_train, X_val, y_val)
    evaluate_and_save(model, X_test, y_test)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": top_features, "scaler": scaler}, f)
    print(f"Saved model + feature list + scaler to {MODEL_PATH}")


if __name__ == "__main__":
    main()
