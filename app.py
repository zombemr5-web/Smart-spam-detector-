"""Streamlit web app. Run:  streamlit run app.py"""
import json
from pathlib import Path

import streamlit as st

from src.predict import explain, load_models, predict

ROOT = Path(__file__).parent

st.set_page_config(page_title="Smart Spam Detector", page_icon="🛡️")
st.title("🛡️ Smart Spam Detector")
st.caption("Arabic + English | TF-IDF + Machine Learning")


@st.cache_resource
def get_models():
    return load_models()


if not (ROOT / "models" / "best_model.joblib").exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

best, explainer = get_models()
tab_check, tab_perf = st.tabs(["🔍 Check a message", "📊 Model performance"])

with tab_check:
    examples = {
        "Choose an example...": "",
        "Spam (EN)": "Congratulations! You won a free iPhone. Click http://bit.ly/x7Kd2 to claim now",
        "Spam (AR)": "مبروك! ربحت سيارة. اضغط على الرابط لاستلام الجائزة الان",
        "Normal (EN)": "Hey, are we still meeting at 5 for the project?",
        "Normal (AR)": "هلا، تكدر ترسللي محاضرة البرمجة اذا ممكن؟",
    }
    choice = st.selectbox("Examples", list(examples))
    text = st.text_area("Message", value=examples[choice], height=120)

    if st.button("Analyze", type="primary") and text.strip():
        label, p = predict(best, text)
        if label == "spam":
            st.error(f"🚨 SPAM  ({p:.0%} confidence)")
        else:
            st.success(f"✅ Normal message  ({1 - p:.0%} confidence)")
        st.progress(p, text=f"Spam probability: {p:.0%}")

        spam_words, ham_words = explain(explainer, text)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Words pushing toward spam**")
            for w, v in spam_words or [("-", 0)]:
                st.write(f"🔴 {w}")
        with c2:
            st.markdown("**Words pushing toward normal**")
            for w, v in ham_words or [("-", 0)]:
                st.write(f"🟢 {w}")

with tab_perf:
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        st.subheader(f"Best model: {m['best_model']}")
        st.table(m["results"])
    for img in ["model_comparison.png", "confusion_matrix.png"]:
        p = ROOT / "reports" / img
        if p.exists():
            st.image(str(p))
