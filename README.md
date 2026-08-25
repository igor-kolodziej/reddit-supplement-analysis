# Reddit Supplement NLP

A streaming NLP pipeline for exploring how people discuss supplements: which products are mentioned, whether a sentence describes an experience or asks a question, what sentiment it carries, and which benefit or side-effect aspects appear.

The project combines exact and fuzzy mention matching, zero-shot intent classification, VADER sentiment, dependency-aware negation rules, and sentence-level aspect extraction. It was developed as a text-analysis project and is presented here as an auditable research workflow rather than a source of health advice.

> Reddit posts are anecdotal user-generated content. The output cannot establish safety, efficacy, dosage, or causality and must not be used for medical decisions.

## Pipeline

```text
Reddit JSONL
   |
   +-- optional submission/comment join
   +-- normalization -> sentence parsing -> supplement matching
                                      -> intent + sentiment + aspects
                                      -> streaming JSONL output
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[models,dev]"
python -m spacy download en_core_web_sm

# Run a small privacy-safe example on CPU
reddit-supplement-analyze \
  --input examples/synthetic_reddit.jsonl \
  --output synthetic-output.jsonl \
  --device cpu \
  --max-records 2

pytest
```

The default intent model is `facebook/bart-large-mnli`; the first run downloads its public weights. Use `--intent-model` to select a compatible Hugging Face zero-shot classifier.

## Joining separate dumps

When submissions and comments are stored separately, pass explicit paths rather than editing source code:

```bash
reddit-supplement-join \
  --submissions data/Supplements_submissions \
  --comments data/Supplements_comments \
  --output combined_reddit_data.jsonl
```

Multiple `--submissions` and `--comments` arguments are supported. Input records are expected to be one JSON object per line; comments are connected through `link_id`.

## Output schema

Every output line describes one sentence containing an explicit or context-inferred supplement mention:

```json
{
  "supplements": ["magnesium"],
  "classification": "positive",
  "sentiment_score": 0.4404,
  "is_question": false,
  "intent": "experience",
  "intent_score": 0.91,
  "aspects": [{"aspect": "sleep_pos", "polarity": "positive"}],
  "sentence_context": "Magnesium helped me sleep.",
  "doc_type": "comment",
  "doc_id": "synthetic-comment",
  "author": "synthetic-user",
  "submission_id": "synthetic-post",
  "subreddit": "SyntheticSupplements"
}
```

The example is synthetic. Real usernames or post text are not committed to this repository.

## Configuration

`params.py` contains the supplement vocabulary, synonyms, aspect terms, and intent labels. Keeping the vocabulary in source control makes review and versioning possible. Domain terms are heuristic and should be validated before a new analysis.

## Analysis notebook

`analysis.ipynb` contains the exploratory aggregation and visualization workflow. Outputs are stripped from the public notebook so that usernames, excerpts, and machine-specific paths are not embedded in Git history. Install the optional analysis environment with:

```bash
pip install -e ".[analysis]"
```

## Data and ethics

- Obtain data lawfully and review Reddit's current terms, API rules, and research requirements.
- Minimize retained fields and remove usernames where identity is not necessary.
- Do not republish deleted content or quote sensitive health disclosures.
- Treat sentiment and aspect labels as fallible model outputs, not clinical evidence.
- Report aggregate results with uncertainty and document collection dates and coverage.

## Limitations

- Zero-shot intent labels and lexicon/rule-based aspects can be wrong, especially with sarcasm and domain slang.
- Fuzzy matching trades recall for false positives.
- Discussion frequency reflects the sampled communities, not population prevalence.
- Historical dump coverage and moderation practices can change over time.

## Author

[Igor Kołodziej](https://igor-kolodziej.github.io/) | [LinkedIn](https://www.linkedin.com/in/igor-kolodziej/)
