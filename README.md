# Demo: Semantic Aspect Matching + ModernBERT Sentiment

This is a runnable demonstration of the **robustness pipeline** described in
Section VI-C of the report ("semantic+ModernBERT"), shown live alongside the
**primary keyword+FinBERT pipeline** whose output populates `aspect_sentiments.csv`
and which was intrinsically evaluated in Section 7.

## Why two pipelines?

The project's primary ABSA pipeline (Section 6 of the report) does aspect
matching with keyword/pattern rules and sentiment scoring with FinBERT. That
is the pipeline evaluated for accuracy in Section 7 (76.5% aspect accuracy,
75.8% sentiment accuracy).

This demo runs a second, independent pipeline on the **same sentences**:

| Step | Primary pipeline | Demo pipeline |
|---|---|---|
| Aspect matching | Keyword/pattern rules | Semantic similarity (sentence embeddings) |
| Sentiment scoring | FinBERT | ModernFinBERT (ModernBERT fine-tuned for financial sentiment) |

Running both side by side is the point, not a limitation — it's a live
version of the extraction-method sensitivity check the paper already reports:
switching pipelines changes some tags and scores, which is expected and is
itself evidence for the paper's claim that findings should be checked against
more than one pipeline before being trusted. **The two pipelines are not
expected to agree on every sentence, and Section 7's accuracy numbers apply
only to the primary (keyword+FinBERT) pipeline**, not to this one.

## How it works

1. **Aspect taxonomy as anchors.** Each of the 42 aspects (6 universal + 36
   sector-specific) is given a one-line natural-language definition
   (`ASPECT_DEFINITIONS` in `semantic_modernbert_demo.py`).
2. **Semantic matching.** Both the input sentence and every aspect definition
   are embedded with a sentence-transformer (`all-MiniLM-L6-v2`). The aspect
   whose definition has the highest cosine similarity to the sentence is
   selected. If the best similarity is below a threshold (0.30), the sentence
   is tagged `none` — no aspect confidently applies.
3. **Sentiment scoring.** The sentence is scored with
   [`tabularisai/ModernFinBERT`](https://huggingface.co/tabularisai/ModernFinBERT),
   a ModernBERT checkpoint fine-tuned for financial sentiment (positive /
   negative / neutral).
4. **Side-by-side comparison.** The script pulls a random sample from
   `aspect_sentiments.csv`, which already carries the primary pipeline's
   aspect/sentiment output, and prints both pipelines' results for the same
   sentences.

## Setup

```bash
pip install sentence-transformers transformers torch pandas
```

Requires internet access on first run (downloads `all-MiniLM-L6-v2` and
`tabularisai/ModernFinBERT` from Hugging Face — a few hundred MB total,
cached locally after that).

## Running it

Place `aspect_sentiments.csv` in the same folder, then:

```bash
python semantic_modernbert_demo.py --csv aspect_sentiments.csv --n 5
```

`--n` controls how many random sentences to demo (default 5). Increase it
for a longer walkthrough, or point `--csv` at a different file with the same
`sentence` / `aspect` / `sentiment_score` columns.

## Expected output shape

```
====================================================================================================
SENTENCE                                                | KEYWORD+FinBERT (primary)   | SEMANTIC+ModernBERT (demo)
====================================================================================================
The board has accepted his resignation with appreci... | guidance / +0.77             | none (0.21) / neutral (0.81)
...
====================================================================================================
```

Column 2 is what's already in the CSV (primary pipeline). Column 3 is
computed live by this script (demo pipeline).

## What to point out live (viva talking points)

- Where the two pipelines **agree** on both aspect and sentiment — shows the
  finding is not an artifact of one specific implementation.
- Where they **disagree on aspect** — usually a keyword-matcher false
  positive (e.g. the primary pipeline tags "guidance" on a sentence that
  merely contains the word, not a forward-looking statement) that the
  semantic matcher avoids by reasoning over meaning rather than surface
  keywords.
- Where they **disagree on sentiment** — illustrates the neutral-bias found
  in Section 7 (the primary pipeline's sentiment score is exactly 0.0 for
  56% of the full corpus); ModernFinBERT's disagreement on those exact rows
  is a natural discussion point.

## Files

- `semantic_modernbert_demo.py` — the runnable script.
- `README.md` — this file.
- `aspect_sentiments.csv` — the dataset to test against (not included here;
  use the same file already used for Section 7's intrinsic evaluation).
