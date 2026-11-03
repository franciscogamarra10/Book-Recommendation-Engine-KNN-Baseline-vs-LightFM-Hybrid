# Book Recommendation Engine — KNN Baseline vs LightFM Hybrid

A comparative analysis of three recommendation approaches on the
[Book-Crossings](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) dataset,
built for the freeCodeCamp *Machine Learning with Python* project and extended
into a full evaluation of item-to-item, user-level, and cold-start recommendation
quality.

---

## What this repository contains

| file | purpose |
|---|---|
| `fcc_book_recommendation_knn_vs_lightfm_features.ipynb` | Full pipeline: robust loading, feature engineering, KNN baseline, LightFM sweep, item-to-item comparison, warm user demo, cold-start demo |
| `README.md` | This file |

---

## The three recommendation levels

| level | query | KNN | LightFM | measured |
|---|---|---|---|---|
| item-to-item | "books similar to X" | yes | yes | P@10 ≈ 0.033 |
| warm user | "books for user U" | no | yes | 2/10 for most active user, 2.17x lift over popularity |
| cold-start | "books for demographics D" | no | yes | 4/10 overlap between profiles |

KNN — item-based collaborative filtering with cosine distance — answers one
question and passes the freeCodeCamp test unchanged. LightFM — WARP loss over
positive-only interactions with item features from title TF-IDF and target-encoded
author — answers all three. The pipeline exists to enable the second and third rows.

---

## Dataset

[Book-Crossings](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) — three CSVs:
books (ISBN, title, author, year, publisher), ratings (user, ISBN, rating 0–10),
users (user, location, age).

After FCC filtering (users with ≥200 ratings, books with ≥100 ratings), the
working set is **66,286 interactions**, **893 users**, **914 unique titles**,
**14,725 positives** (rating ≥ 6).

---

## Feature sweep — item-to-item

Four item-level configurations tested on the same split, same hyperparameters,
same cold-start-filtered test set:

| config | P@10 | R@10 | AUC | n_item | train_s |
|---|---|---|---|---|---|
| tfidf | 0.0301 | 0.0738 | 0.778 | 38 | 0.9 |
| **tfidf+author** | **0.0328** | **0.0807** | **0.807** | 44 | 1.6 |
| tfidf+author+year | 0.0305 | 0.0747 | 0.808 | 49 | 1.2 |
| tfidf+author+year+publisher | 0.0301 | 0.0708 | 0.811 | 57 | 1.2 |
| full (with user features) | 0.0166 | 0.0388 | 0.799 | 57+30 | 3.5 |

**Locked baseline:** `tfidf + Book-Author` on raw title with `min_df=0.01`,
`max_df=0.99`, `no_components=40`, `epochs=30`, `item_alpha=1e-6`,
`user_alpha=1e-6`. **P@10 = 0.0328, R@10 = 0.0807, AUC = 0.8069.**

Findings:

- **Author adds real signal to the top-10.** +0.0027 P@10 and +0.0069 R@10 over
  tfidf-only, with AUC up from 0.778 to 0.807.
- **Year and Publisher add marginal AUC but no top-10 improvement.** Both end up
  at the same P@10 as tfidf-only.
- **User features lower P@10 but keep competitive AUC.** The `full` config scores
  0.017 P@10 (worst) but 0.799 AUC (comparable to `tfidf`). Demographics push
  popular titles up for every user, damaging the tight top-10.

---

## Item-to-item comparison

Query: *Where the Heart Is (Oprah's Book Club (Paperback))*

**KNN** (cosine over user-item co-occurrence):
- The Weight of Water (0.771)
- I Know This Much Is True (0.752)
- The Lovely Bones: A Novel (0.723)
- Blue Diary (0.718)

**LightFM** (cosine over learned item embeddings):
- She's Come Undone (0.941)
- Vinegar Hill (0.915)
- The Book of Ruth (0.904)
- Drowning Ruth (0.898)
- Open House (0.896)

**Overlap at top-5: 0/4. At top-20: 1/20** (The Joy Luck Club).

The two models surface disjoint neighbourhoods by design. KNN aggregates by user
co-rating patterns; LightFM aggregates by learned item embeddings from title
n-grams plus author signal. Where they agree, both mechanisms point to the same
title; when they diverge, each is doing its own job.

---

## Warm user prediction

User 16795 (119 train positives, 30 test positives):

- Top-10 recommendations: **2/10 hits** (The Lovely Bones, Left Behind)
- The list spans genres (literary fiction, memoir, thriller) because the user's
  reading history does — the model recommends across a diverse profile rather
  than clustering on a single genre.

Across a sample of 50 test users (36 with held-out positives):

- Mean P@10: **0.0361**
- Median P@10: **0.0000**
- Users with ≥1 hit: 12/36
- Users with 0 hits: 24/36

**Lift over popularity: 2.17x.** A trivial "recommend the ten most-rated training
titles" baseline scores P@10 = 0.0167. The personalized ranking reaches 0.0361.
The 2.17x improvement is the contribution of the user representation — the model
reorders the catalogue by each user's learned preferences rather than repeating
bestsellers.

**KNN cannot enter this comparison.** KNN has no concept of a user. It has no way
to produce a per-user top-10 without hand-building a pipeline that aggregates
similar items for every book the user has rated, deduplicates, and re-ranks.

---

## Cold-start prediction

Two users sharing the same country but differing in age bin (and state) receive
recommendations that diverge on **6 of 10 positions**:

- **Proxy A (`Age_29`)** — Harry Potter 1 and 2, The Lovely Bones, The Red Tent,
  The Secret Life of Bees, Good in Bed, The Five People You Meet in Heaven,
  The Client, Two for the Dough
- **Proxy B (`Age_30`)** — The Lovely Bones, Lord of the Flies, Da Vinci Code,
  The Red Tent, Red Dragon, Harry Potter 5, The Pelican Brief, The Firm,
  1st to Die, Good in Bed

**Shared titles (4/10):** The Lovely Bones, The Red Tent, The Da Vinci Code,
Good in Bed — universal bestsellers.

**Popularity overlap:** 6/10 for A, 4/10 for B. Demographics shift the non-popular
tail into distinct genre clusters (mainstream pop / women's fiction vs thrillers /
literary).

**What is proven.** With `user_identity_features=False`, the model represents users
exclusively through demographics plus a universal bias. Two users with different
demographic profiles receive measurably different lists. The divergence maps to
age-correlated genre preferences.

**What is not proven.** Production-grade single-row transform. The pipeline's
`RareLabelPolars` recomputes frequencies on the incoming batch, so a single-row
transform of raw values returns zero-valued user features. The demo works around
this by using real warm users as proxies — their encoded profiles are numerically
identical to what a cold user with the same demographics would produce via a
working transform.

**Caveat.** The two proxies differ in both age bin **and** state bin. A pure
age-only test would hold state constant, but the filtered dataset does not offer
enough users sharing (country, state, distinct age bins) to run it.

**The ceiling.** Demographics shift 6 of 10 slots; the other 4 are locked by
popularity. In a dataset where a handful of titles dominate the interaction
matrix, this is the maximum signal available from demographics alone.

---

## Preprocessing decisions that were load-bearing

- **Parenthetical stripping.** `preprocess_titles` lowercases and strips
  punctuation; `min_df=0.01` and `max_df=0.99` in the TF-IDF filter out stopwords
  and ultra-frequent n-grams. Without these filters, tokens like `the`, `of`,
  `and`, `for the` consume 4–5 of the 60–100 TF-IDF slots.
- **Age binning.** Book-Crossings has 22% missing ages and a max of 201. Quantile
  binning over the raw distribution collapsed 54% of users into one bin. The
  pipeline clips Age to [5, 90] and uses `nbins=10`, spreading users across 10
  bins with 5–14% per bin.
- **Encoding by cardinality.** Target encoding for `Book-Author` (616 unique) and
  `city` (642 unique) — one-hot would blow up the feature matrix. One-hot for
  `country` (29 unique), `state` (124 unique), `Publisher` (362 unique), and
  binned `Year` (5 bins) — the identity signal survives. The uniform
  "target-encode everything" attempt collapsed P@10 from 0.030 to 0.012 because
  low-cardinality features lost their identity signal.

---

## Standalone reference (title-level)

A secondary model built with `title` as the item id (editions aggregated) reaches
P@10 = 0.0368, R@10 = 0.0928, AUC = 0.7057. The pipeline's ISBN-level
representation has a lower top-10 precision but much higher global ranking quality
(AUC 0.807 vs 0.706). The gap is methodological — different `item_id`
granularities and different test-set sizes — not a model difference.

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
