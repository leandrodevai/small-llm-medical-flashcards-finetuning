from medical_flashcards.metrics import clean_answer, normalize_answer, token_f1


def test_clean_answer_removes_common_prefixes():
    assert clean_answer("Answer: hypertension") == "hypertension"
    assert clean_answer("assistant: beta blocker") == "beta blocker"


def test_normalize_answer_removes_articles_and_punctuation():
    assert normalize_answer("The left-sided, chest pain.") == "left sided chest pain"


def test_token_f1_handles_overlap():
    assert token_f1("acute kidney injury", "acute renal injury") == 2 / 3


def test_token_f1_handles_empty_answers():
    assert token_f1("", "") == 1.0
    assert token_f1("", "aspirin") == 0.0
