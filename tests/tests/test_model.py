from src.predict import explain, load_models, predict


def test_spam_and_ham_en():
    best, _ = load_models()
    assert predict(best, "Congratulations! You won a free iPhone, click http://x.com now")[0] == "spam"
    assert predict(best, "Can you send me the lecture notes for math?")[0] == "ham"


def test_spam_and_ham_ar():
    best, _ = load_models()
    assert predict(best, "مبروك! ربحت سيارة. اضغط على الرابط لاستلام الجائزة الان")[0] == "spam"
    assert predict(best, "تكدر ترسللي محاضرة الفيزياء اذا ممكن؟")[0] == "ham"


def test_explain_returns_words():
    _, explainer = load_models()
    spam_words, _ = explain(explainer, "free iPhone click now")
    assert len(spam_words) > 0
