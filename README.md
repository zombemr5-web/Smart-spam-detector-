# 🛡️ Smart Spam Detector (Arabic + English)

A machine-learning web app that classifies messages as **spam** or **normal**, in Arabic and English, and explains *why*.

## Features
- 🌐 Interactive **Streamlit** web app
- 🤖 Compares **3 models** (Naive Bayes, Logistic Regression, calibrated Linear SVM) using cross-validation and picks the best automatically
- 🧹 Text preprocessing with **Arabic normalization** (diacritics, alef/ya/ta-marbuta variants), URL and number tokens
- 🔍 **Explainability**: shows which words pushed the message toward spam or normal
- 📊 Confusion matrix and model comparison charts
- ✅ Unit tests

## Project structure
```
app.py                  # Streamlit app
src/preprocess.py       # text cleaning (EN + AR)
src/train.py            # trains, compares, saves models and reports
src/predict.py          # prediction + explanation helpers
data/spam.csv           # dataset (label,text)
data/make_sample_data.py# generates the demo dataset
models/                 # saved models
reports/                # metrics + charts
tests/                  # unit tests
```

## Run it
```bash
pip install -r requirements.txt
python -m src.train
streamlit run app.py
pytest
```

## ⚠️ About the dataset
The included `data/spam.csv` is a **small synthetic bilingual dataset** generated from templates, so the project runs out of the box. Scores near 100% on it are **expected and not meaningful**, because the messages are repetitive.

To get realistic results, replace `data/spam.csv` with a real dataset (for example the *SMS Spam Collection* from UCI/Kaggle) with columns `label` (`spam`/`ham`) and `text`, then run `python -m src.train` again.

## Tech stack
Python, scikit-learn, pandas, matplotlib, Streamlit
