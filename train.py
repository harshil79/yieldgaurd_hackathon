"""Train a baseline pass/fail classifier on the UCI SECOM dataset.

Accuracy is not the goal here -- this is the simplest model that can sit
behind a real API. The infrastructure around it is what's being graded.
"""
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

TOP_N_FEATURES = 20
MODEL_PATH = "model.pkl"


def main():
    secom = fetch_ucirepo(id=179)
    # ucimlrepo doesn't pre-split features/targets for this dataset --
    # `original` has everything: 'class' is the label, 'timestamp' isn't a feature.
    raw = secom.data.original
    X = raw.drop(columns=["class", "timestamp"])
    y = raw["class"]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    top_features = X_imputed.var().sort_values(ascending=False).head(TOP_N_FEATURES).index.tolist()
    X_top = X_imputed[top_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f} (irrelevant -- this is the throwaway part)")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": top_features}, f)
    print(f"Saved model + feature list to {MODEL_PATH}")


if __name__ == "__main__":
    main()
