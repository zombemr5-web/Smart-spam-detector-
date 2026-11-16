"""Train, compare and save spam-detection models.

Run from the project root:  python -m src.train
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "spam.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def build_pipeline(classifier):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=1)),
            ("clf", classifier),
        ]
    )


def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (calibrated)": CalibratedClassifierCV(LinearSVC(), cv=3),
    }


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA).dropna()
    df["y"] = (df["label"] == "spam").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    results, fitted = {}, {}
    for name, clf in get_models().items():
        pipe = build_pipeline(clf)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "cv_f1": round(float(cv_f1), 4),
            "test_f1": round(float(f1_score(y_test, pred)), 4),
        }
        fitted[name] = pipe
        print(f"{name:22s} CV F1={cv_f1:.3f}  Test F1={results[name]['test_f1']:.3f}")

    best_name = max(results, key=lambda n: (results[n]["cv_f1"], results[n]["test_f1"]))
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best.predict(X_test), target_names=["ham", "spam"]))

    # Save models. The explainer is always Logistic Regression (it exposes word weights).
    joblib.dump(best, MODELS / "best_model.joblib")
    joblib.dump(fitted["Logistic Regression"], MODELS / "explainer.joblib")
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # Plot 1: model comparison
    names = list(results)
    cv_scores = [results[n]["cv_f1"] for n in names]
    test_scores = [results[n]["test_f1"] for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.2 for i in x], cv_scores, 0.4, label="Cross-val F1")
    ax.bar([i + 0.2 for i in x], test_scores, 0.4, label="Test F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "model_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 2: confusion matrix of the best model
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best.predict(X_test), display_labels=["ham", "spam"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved models and reports.")


if __name__ == "__main__":
    main()
