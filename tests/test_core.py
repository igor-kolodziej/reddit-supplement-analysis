import json

import spacy

from sentiment_analysis import (
    build_matchers,
    choose_overall_label,
    contains_pronoun,
    normalize_text,
    preprocess_reddit_text,
    stream_records,
)


def test_text_normalization_and_reddit_cleanup():
    assert normalize_text("it’s useful—sometimes") == "it's useful-sometimes"
    cleaned = preprocess_reddit_text("First point\n- Second point")
    assert "Second point" in cleaned
    assert "- Second" not in cleaned


def test_overall_label_rules():
    assert choose_overall_label("question", 0.8, False) == "neutral (question)"
    assert choose_overall_label("experience", 0.6, False) == "positive"
    assert choose_overall_label("experience", -0.6, False) == "negative"
    assert choose_overall_label("experience", 0.6, True) == "ineffective"


def test_pronoun_detection_has_word_boundaries():
    assert contains_pronoun("It worked for me")
    assert not contains_pronoun("The biscuit was fresh")


def test_matchers_detect_exact_and_synonym_mentions():
    nlp = spacy.blank("en")
    phrase_matcher, keyword_matcher = build_matchers(
        nlp,
        ["magnesium", "lion's mane"],
        {"lion's mane": ["lions mane"]},
    )
    doc = nlp("I compared magnesium with lions mane.")
    phrase_hits = [doc[start:end].text.lower() for _, start, end in phrase_matcher(doc)]
    keyword_hits = [name for name, _, _ in keyword_matcher.extract_keywords(doc.text, span_info=True)]
    assert "magnesium" in phrase_hits
    assert "lion's mane" in keyword_hits


def test_stream_records_reads_jsonl(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"id": "one"}) + "\n" + json.dumps({"id": "two"}) + "\n")
    assert [record["id"] for record in stream_records(str(path))] == ["one", "two"]

