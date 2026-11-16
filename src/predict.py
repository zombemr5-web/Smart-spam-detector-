"""Prediction + explanation helpers (used by the app and tests)."""
from pathlib import Path

import joblib
import numpy as np

from src.preprocess import clean_text

MODELS = Path(__file__).resolve().parent.parent / "models"


def load_models():
    best = joblib.load(MODELS / "best_model.joblib")
    explainer = joblib.load(MODELS / "explainer.joblib")
    return best, explainer


def predict(model, text: str):
    """Return (label, spam_probability)."""
    proba = float(model.predict_proba([text])[0][1])
    return ("spam" if proba >= 0.5 else "ham"), proba


def explain(explainer, text: str, top_n: int = 5):
    """Words in the message that pushed it toward spam (+) or ham (-).

    contribution = tfidf value of the word * logistic-regression weight.
    """
    tfidf = explainer.named_steps["tfidf"]
    clf = explainer.named_steps["clf"]
    vec = tfidf.transform([text])
    names = np.array(tfidf.get_feature_names_out())
    idx = vec.nonzero()[1]
    contrib = vec[0, idx].toarray().ravel() * clf.coef_[0][idx]
    order = np.argsort(contrib)
    spam_words = [(names[idx[i]], float(contrib[i])) for i in order[::-1] if contrib[i] > 0][:top_n]
    ham_words = [(names[idx[i]], float(contrib[i])) for i in order if contrib[i] < 0][:top_n]
    return spam_words, ham_words
