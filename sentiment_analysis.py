# main.py
import argparse
import json
import gzip
import os
import sys
import unicodedata
from typing import Iterable, Dict, Any, List, Tuple
from collections import defaultdict

import spacy
from spacy.matcher import PhraseMatcher
from flashtext import KeywordProcessor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from params import SUPPLEMENT_LIST, SUPPLEMENT_SYNONYMS, ASPECT_TERMS, INTENT_LABELS
import regex as re
from rapidfuzz import process, fuzz

# --------------------------------
# IO utilities (JSON / JSON Lines)
# --------------------------------
def open_maybe_gzip(path, mode="rt", encoding="utf-8"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)
    return open(path, mode, encoding=encoding)

def stream_records(path: str) -> Iterable[Dict[str, Any]]:
    # Accept JSON Lines (.jsonl/.ndjson) or a single JSON dict file.
    if path.endswith(".jsonl") or path.endswith(".ndjson") or path.endswith(".jsonl.gz") or path.endswith(".ndjson.gz"):
        with open_maybe_gzip(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open_maybe_gzip(path, "rt") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for _k, v in data.items():
                    yield v
            elif isinstance(data, list):
                for v in data:
                    yield v
            else:
                raise ValueError("Unsupported JSON structure; expect dict of submissions or JSON Lines.")

def write_jsonl(path: str, records: Iterable[Dict[str, Any]], batch_size: int = 5000):
    batch = []
    with open_maybe_gzip(path, "wt") as f:
        for r in records:
            # Convert record to JSON string and add to batch
            batch.append(json.dumps(r, ensure_ascii=False))
            
            # When the batch is full, write it to the file
            if len(batch) >= batch_size:
                f.write("\n".join(batch) + "\n")
                batch.clear() # Clear the batch to start fresh

        # After the loop, write any remaining records
        if batch:
            f.write("\n".join(batch) + "\n")

# -----------------
# Text normalization
# -----------------
def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "")
    t = t.replace("’", "'").replace("‘", "'").replace("—", "-").replace("–", "-")
    return t

def preprocess_reddit_text(text: str) -> str:
    """Clean Reddit formatting for proper sentence segmentation."""
    if not text: return text
    
    # Remove standalone periods (Reddit paragraph separator)
    text = re.sub(r'\n\s*\.\s*\n', '\n\n', text)
    
    # Convert bullet points to periods for sentencizer
    text = re.sub(r'\n\s*[-•*]\s+', '. ', text)
    
    # Remove quote markers
    text = re.sub(r'\n\s*>\s*', '\n', text)
    
    # Normalize multiple newlines to double
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Add period before capitalized text after newline
    text = re.sub(r'(?<![.!?])\s*\n+\s*(?=[A-Z])', '. ', text)
    
    return text.strip()


# --------------
# NLP components
# --------------
def build_nlp(disable_ner=True):
    exclude = ["ner", "lemmatizer", "attribute_ruler", "tok2vec", "morphologizer"]
    if not disable_ner:
        exclude = [e for e in exclude if e != "ner"]
    nlp = spacy.load("en_core_web_sm", exclude=exclude)
    
    @spacy.Language.component("reddit_boundaries")
    def set_reddit_boundaries(doc):
        for i, token in enumerate(doc[:-1]):
            if '\n\n' in token.text or token.text == '\n':
                doc[i+1].is_sent_start = True
        return doc
    
    # Add reddit_boundaries FIRST (before parser runs)
    nlp.add_pipe("reddit_boundaries", first=True)
    
    # Add sentencizer before parser for additional boundary rules
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", before="parser")
    
    return nlp


def build_matchers(nlp, supplement_list: List[str], synonyms: Dict[str, List[str]]):
    # PhraseMatcher for precise multi-word matching
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(x) for x in supplement_list]
    pm.add("SUPPLEMENTS", patterns)

    # FlashText for large synonym sets / variants
    kp = KeywordProcessor(case_sensitive=False)
    for base in supplement_list:
        kp.add_keyword(base, base)
    for k, vs in synonyms.items():
        for v in vs:
            kp.add_keyword(v, k)
    return pm, kp

def build_intent_pipeline(model_name: str = "facebook/bart-large-mnli", device: str = "gpu"):
    from transformers import pipeline

    if device == "cpu":
        device_id = -1
    elif device == "auto":
        try:
            import torch

            device_id = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device_id = -1
    else:
        device_id = int(device)
    return pipeline("zero-shot-classification", model=model_name, device=device_id)

def build_vader():
    return SentimentIntensityAnalyzer()

# ----------------
# Core processing
# ----------------
def detect_supplements(pmatcher, kw, doc):
    hits = set()
    
    # 1. Exact Matches (Keep this!)
    for match_id, start, end in pmatcher(doc):
        span = doc[start:end]
        hits.add((span.text, span.start_char, span.end_char))
    for found, start, end in kw.extract_keywords(doc.text, span_info=True):
        hits.add((found, start, end))

    found_texts = {h[0].lower() for h in hits} 

    # 2. Robust Fuzzy Recovery
    for token in doc:
        # OPTIMIZATION: Only check Nouns/Proper Nouns
        if token.pos_ in ["PROPN", "NOUN"] and not token.is_stop:
            
            t_text = token.text
            t_len = len(t_text)
            
            # SAFETY 1: Ignore very short words (too much noise)
            if t_len < 4:
                continue
                
            # Skip if we already found this exact word
            if t_text.lower() in found_texts:
                continue

            # SAFETY 2: Variable Thresholds
            # If word is short (4-5 chars), require 90% match (avoid Icon -> Iron)
            # If word is long (6+ chars), allow 85% match (allow Magnesiuim -> Magnesium)
            threshold = 92 if t_len < 6 else 85

            match = process.extractOne(
                t_text, 
                SUPPLEMENT_LIST, 
                scorer=fuzz.ratio, 
                score_cutoff=threshold
            )

            if match:
                best_match_name = match[0]
                
                # SAFETY 3: Sanity check - don't match a single word to a huge phrase
                # If token is "Wrt" and match is "St John's Wort", ignore it.
                # The match length shouldn't be more than 2x the token length.
                if len(best_match_name) > (t_len * 2):
                    continue

                hits.add((best_match_name, token.idx, token.idx + len(t_text)))

    return sorted(hits, key=lambda x: (x[1], x[2]))

def classify_intent(zero_shot, text: str) -> Tuple[str, float]:
    """Classify one sentence without retaining a model object in a global cache key."""
    res = zero_shot(text, candidate_labels=INTENT_LABELS, multi_label=False)
    return res["labels"][0], float(res["scores"][0])

def is_grammatically_negated(token) -> bool:
    """Checks dependency tree for negation (children, head verbs, prepositions)."""
    # 1. Direct modification ("no", "not", "never")
    for child in token.children:
        if child.dep_ == "neg" or child.lower_ in {"no", "none", "zero", "without", "free"}:
            return True

    # 2. Head Verb Analysis ("did not cause", "stopped")
    head = token.head
    if head != token:
        # Inherently negative verbs (removal)
        removal_verbs = {"prevent", "cure", "stop", "eliminate", "kill", "banish", 
                         "destroy", "remove", "avoid", "block", "reduce", "lower", "decrease"}
        
        if head.pos_ in ("VERB", "AUX"):
            # "did NOT cause"
            for child in head.children:
                if child.dep_ == "neg": return True
            # "STOPPED the pain"
            if head.lemma_.lower() in removal_verbs: return True

    # 3. Prepositional ("without side effects")
    if head.dep_ == "prep" and head.lower_ in {"without", "minus", "except"}:
        return True

    return False

def find_aspects_dependency(doc: spacy.tokens.Span) -> Dict[str, bool]:
    """
    Matches keywords using Regex, then uses SpaCy dependency parsing to 
    determine if they are Negated, Cured, or Avoided.
    """
    present = defaultdict(bool)
    text_lower = doc.text.lower()
    
    # 1. DEFINE EXCEPTIONS
    # These are terms that, when found in a Negative list but negated ("No X"),
    # should flip to a specific Positive category (Mood/Sleep) rather than generic Side Effects.
    # Add ALL your major symptoms here.
    direct_flip_exceptions = {
        "anxiety": "mood_pos",
        "panic": "mood_pos",
        "depression": "mood_pos",
        "stress": "mood_pos",
        "insomnia": "sleep_pos",
        "fatigue": "energy_pos",
        "tiredness": "energy_pos",
        "lethargy": "energy_pos",
        "brain fog": "cognition_pos",
        "confusion": "cognition_pos",
        "acne": "skin_hair_nails_pos",
        "bloat": "digestion_pos",
        "constipation": "digestion_pos",
        "diarrhea": "digestion_pos"
    }

    curative_verbs = {"cure", "fix", "stop", "prevent", "eliminate", "resolve", "kill", "help", "heal", "treat"}
    
    categories = list(ASPECT_TERMS.keys())

    for category in categories:
        is_negative_category = "_neg" in category
        
        for term in ASPECT_TERMS.get(category, []):
            if not term: continue
            
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span is None: continue
                root_token = span.root
                head = root_token.head

                # Check Negation
                if is_grammatically_negated(root_token):
                    
                    if is_negative_category:
                        # --- SMART FLIP LOGIC ---
                        
                        # 1. Check Specific Flip Exceptions ("No anxiety" -> Mood Pos)
                        # We check if the found term matches one of our exceptions
                        flipped_category = None
                        for exc, target_cat in direct_flip_exceptions.items():
                            if exc in term:
                                flipped_category = target_cat
                                break
                        
                        if flipped_category:
                            present[flipped_category] = True
                        
                        # 2. Check Curative Verbs ("Cured the pain")
                        # If the verb is curative, we flip to the generic positive counterpart
                        elif head.lemma_.lower() in curative_verbs:
                             # Try to calculate the positive key (mood_neg -> mood_pos)
                            target_pos = category.replace("_neg", "_pos")
                            if target_pos in ASPECT_TERMS:
                                present[target_pos] = True
                            else:
                                present["effectiveness_pos"] = True

                        # 3. Default: Negated Symptom ("No headache") -> Side Effects Positive
                        else:
                            present["side_effects_pos"] = True
                        
                    else:
                        # Negated Positive ("No energy", "Didn't work") -> Negative
                        target_neg = category.replace("_pos", "_neg")
                        if target_neg in ASPECT_TERMS:
                            present[target_neg] = True
                        else:
                            present["effectiveness_neg"] = True
                
                else:
                    # Not negated -> Original category found
                    present[category] = True
                
                break 
                
    return present

def choose_overall_label(intent_label: str, vader_compound: float, ineffect: bool) -> str:
    if intent_label == "question":
        return "neutral (question)"
    if intent_label in ("informational", "other"):
        return "neutral (non-experiential)"
    if ineffect:
        return "ineffective"
    if vader_compound >= 0.1:
        return "positive"
    if vader_compound <= -0.1:
        return "negative"
    return "neutral"

def process_sentence(sent: spacy.tokens.Span,
                     mentions: List[Tuple[str, int, int]],
                     intent_pipe,
                     vader,
                     doc_meta: Dict[str, Any]) -> Dict[str, Any]:
    stext = sent.text.strip()
    if not stext:
        return None
    intent_label, intent_score = classify_intent(intent_pipe, stext)
    text_lower = stext.lower()

    # Ineffective fast rule
    ineffective_keywords = [
    "no effect",
    "no effects",
    "didn't work",
    "didnt work",
    "did not work",
    "did nothing",
    "does nothing",
    "nothing at all",
    "felt nothing",
    "no benefit",
    "no benefits",
    "no noticeable effect",
    "no noticeable effects",
    "no noticeable difference",
    "didn't notice anything",
    "didnt notice anything",
    "didn't help",
    "didnt help",
    "did not help",
    "no help",
    "no improvement",
    "no improvements",
    "zero results",
    "no results",
    "waste of money",
    "waste of time",
    "total waste",
    "useless",
    "pointless",
    "worthless",
    "overhyped",
    "over hyped",
    "just placebo",
    "only placebo",
    "placebo only",
    "scam"
]

    is_ineffective = any(k in text_lower for k in ineffective_keywords)

    # Sentiment only if experiential/promotional
    vader_score = vader.polarity_scores(stext)["compound"] if intent_label in ("experience", "promotional") else 0.0

    # Aspect presence
    aspects_present = find_aspects_dependency(sent)
    aspects_payload = []

    for a in ["effectiveness_pos",
                "effectiveness_neg",
                "side_effects_pos",
                "side_effects_neg",
                "sleep_pos",
                "sleep_neg",
                "energy_pos",
                "energy_neg",
                "mood_pos",
                "mood_neg",
                "cognition_pos",
                "cognition_neg",
                "cost_pos",
                "cost_neg",
                "physical_performance_pos",
                "physical_performance_neg",
                "appetite_pos",
                "appetite_neg",
                "libido_pos",
                "libido_neg",
                "skin_hair_nails_pos",
                "skin_hair_nails_neg",
                "digestion_pos",
                "digestion_neg",
                "onset_duration_pos",
                "onset_duration_neg"]:
        if aspects_present[a]:
            # polarity derived from VADER where applicable
            pol = "positive" if vader_score >= 0.1 else ("negative" if vader_score <= -0.1 else "neutral")
            aspects_payload.append({"aspect": a, "polarity": pol})

    classification = choose_overall_label(intent_label, vader_score, is_ineffective)

    result = {
        "supplements": sorted({m[0] for m in mentions}),
        "classification": classification,
        "sentiment_score": round(vader_score, 4),
        "is_question": intent_label == "question",
        "intent": intent_label,
        "intent_score": round(intent_score, 4),
        "aspects": aspects_payload,
        "sentence_context": stext,
        "doc_type": doc_meta.get("doc_type"),
        "doc_id": doc_meta.get("doc_id"),
        "author": doc_meta.get("author"),
        "submission_id": doc_meta.get("submission_id"),
        "subreddit": doc_meta.get("subreddit"),
    }
    return result

def contains_pronoun(text: str) -> bool:
    # Matches common pronouns used to refer to supplements
    # \b ensures we don't match "it" inside "biscuit"
    return bool(re.search(r'\b(it|this|these|they|stuff|product|pill|pills|powder|capsule)\b', text, re.IGNORECASE))

def iter_text_items(submission: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    sub_id = submission.get("id") or submission.get("submission_id")
    author = submission.get("author", "N/A")
    subreddit = submission.get("subreddit", "N/A")

    if submission.get("title"):
        yield {
            "text": submission["title"],
            "doc_type": "submission_title",
            "doc_id": sub_id,
            "author": author,
            "submission_id": sub_id,
            "subreddit": subreddit
        }
    if submission.get("selftext"):
        yield {
            "text": submission["selftext"],
            "doc_type": "submission_body",
            "doc_id": sub_id,
            "author": author,
            "submission_id": sub_id,
            "subreddit": subreddit
        }
    for c in submission.get("comments", []):
        body = c.get("body")
        if not body:
            continue
        yield {
            "text": body,
            "doc_type": "comment",
            "doc_id": c.get("id", "N/A"),
            "author": c.get("author", "N/A"),
            "submission_id": sub_id or submission.get("id"),
            "subreddit": subreddit
        }

def run(args):
    nlp = build_nlp(disable_ner=True)
    pmatcher, kw = build_matchers(nlp, SUPPLEMENT_LIST, SUPPLEMENT_SYNONYMS)
    intent_pipe = build_intent_pipeline(model_name=args.intent_model, device=args.device)
    vader = build_vader()

    def records_generator():
        count_records = 0
        
        # We group processing by document to ensure context flows correctly
        for rec in stream_records(args.input):
            if args.max_records and count_records >= args.max_records:
                break
            count_records += 1

            # 1. Aggregate text items for this specific record
            # We need to keep track of the doc_id to reset context
            items = list(iter_text_items(rec))
            if not items:
                continue
                
            texts = []
            metas = []
            
            for item in items:
                text = normalize_text(item["text"])
                if not text or text in ("[deleted]", "[removed]"):
                    continue
                text = preprocess_reddit_text(text)
                texts.append(text)
                metas.append(item)

            if not texts:
                continue

            # 2. Process the batch for this specific document/submission
            # We use nlp.pipe but we iterate carefully to maintain order
            
            active_context_supplements = [] # Stores supplements seen in previous sentence
            current_doc_id = None

            for doc, meta in zip(nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process), metas):
                
                # Reset context if we switch to a completely different comment/post ID
                # (Though items from the same 'rec' are usually related, a comment thread implies distinct authors)
                if meta.get("doc_id") != current_doc_id:
                    active_context_supplements = []
                    current_doc_id = meta.get("doc_id")

                for sent in doc.sents:
                    # Detect mentions in THIS sentence
                    mentions = detect_supplements(pmatcher, kw, sent.as_doc())
                    
                    # --- IMPROVEMENT: Context Propagation ---
                    # Case A: We found explicit mentions. Update context.
                    if mentions:
                        # Update the 'active' context to these new findings
                        # We store just the names (item[0])
                        active_context_supplements = sorted(list({m[0] for m in mentions}))
                    
                    # Case B: No explicit mentions, but we have a pronoun AND valid context
                    elif active_context_supplements and contains_pronoun(sent.text):
                        # We "inject" the previous supplements into this sentence as if they were found
                        # Format: (Name, StartChar, EndChar) - using -1 to indicate inferred
                        mentions = [(sup, -1, -1) for sup in active_context_supplements]
                    
                    # ----------------------------------------

                    if not mentions:
                        continue
                        
                    out = process_sentence(sent, mentions, intent_pipe, vader, meta)
                    if out is not None:
                        yield out

    # Stream write outputs
    write_jsonl(
        args.output, 
        tqdm(records_generator(), desc="Processing", unit="sent"),
        batch_size=args.write_batch_size
    )

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Streaming supplement sentiment with intent gating, negation, and fast matching")
    ap.add_argument("--input", default = "combined_reddit_data.json", help="Path to combined_reddit_data.json or .jsonl/.jsonl.gz")
    ap.add_argument("--output", default="data_for_analysis_final.jsonl", help="Path to output .jsonl or .jsonl.gz")
    ap.add_argument("--batch-size", type=int, default=256, help="spaCy pipe batch size")
    ap.add_argument("--n-process", type=int, default=1, help="spaCy n_process for parallelism")
    ap.add_argument("--max-records", type=int, default=0, help="Limit submissions processed (0 = all)")
    ap.add_argument("--write-batch-size", type=int, default=4096, help="Number of records to batch before writing to file")
    ap.add_argument("--intent-model", default="facebook/bart-large-mnli", help="HF zero-shot model name")
    ap.add_argument("--device", default="auto", help="cpu or GPU id via transformers (cpu/auto)")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
