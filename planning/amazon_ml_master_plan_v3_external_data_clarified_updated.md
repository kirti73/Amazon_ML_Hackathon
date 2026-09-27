# MASTER PLAN — Amazon ML Challenge 2026, Business Entity Resolution

**Version: Submission-Structure-Aligned Guardrailed Master Plan v3**

Grounded in the team's actual training examples (`training_match_examples.md`), not generic entity-resolution theory. Evidence-driven decisions are flagged inline as **[EVIDENCE]**.

---

## 1. Current Understanding

- **[TRAIN EVIDENCE]** The train audit found that an S2/S3 record is not claimed by two S1 entities. Treat this as a train-only empirical property: it may motivate reverse-consistency diagnostics, but it must **not** become an assumed test-time one-to-one constraint. Any reverse-consistency suppression must be validated empirically and remain removable.
- Match-count distribution is **not** dominated by singletons or 1:1 pairs — the mode is 3 matches, and the bulk of mass sits at 2–6. Top-1 or "one match per entity" logic would be wrong on the majority of entities, not just an edge case.
- **[EVIDENCE]** Example 69 (Oncology Associates) contains a true match, `NEXGILD`, with essentially zero name similarity to the S1 name. It matches purely because it shares the address. This is not a one-off — it's proof that name-only candidate generation has a real, non-hypothetical recall hole, and address-based retrieval routes are load-bearing, not optional.
- **[EVIDENCE]** Example 71 shows a true match with address `703 Beacon Court` vs `70 Beacon Court` (digit dropped) — numeric address tokens are strong evidence but not infallible; treat numeric mismatch as a soft negative signal, not a hard filter.
- **[EVIDENCE]** Example 66 (`Kalyani`) shows one true match is just the single word "Kalyani" — a strict subset of "Kalyani Welfare Society". Confirms containment-style features matter more than symmetric similarity here.
- **[EVIDENCE]** Example 71 (`S3-889312697`) has a blank address and is still a true match — confirms missing ≠ negative evidence, must be preserved as NaN not 0.
- **[EVIDENCE]** Example 68 (Damani Technologies) shows the same true-match cluster with an address partly in Devanagari (`दिल्ली`) and partly in Latin script within the same record set — multilingual handling can't assume a record is "in one language."
- Country in test includes France (~15%, confirmed by the test audit) with zero training exposure. Every country-dependent design choice needs a fallback that doesn't require having seen the country.

---

## 1A. Official Challenge Contract — Non-Negotiable

Everything in this master plan is subordinate to the official challenge specification. The following rules are hard constraints, not optimization preferences.

### Data and task

- There are three independent sources. Source 1 is the deduplicated reference source.
- For **every** Source 1 entity, predict **all** matching Source 2 and/or Source 3 entity IDs.
- A Source 1 entity can have zero, one, or many matches. Never assume one-to-one at the S1→S2/S3 output level.
- Input files are TSV. Always read them with an explicit tab separator (`sep="\t"`). Treat all entity IDs as strings.
- The source is available from the filename and/or entity-ID prefix (`S1-`, `S2-`, `S3-`); there is no separate source column in the official TSVs. Our internal `records.parquet` may materialize a `source` column for convenience.
- Training contains US and India. Test additionally contains France. Treat `country` as an open set of string labels. **Never hard-code, filter, or one-hot the pipeline to `{US, India}`.**
- Every test S1 entity, including France, must appear in `matching_results.tsv`.

### Official outputs

`output/matching_results.tsv` must:

- contain exactly one row for every test S1 entity;
- use exactly the official column names `source1_entity_id` and `matched_entity_ids`;
- use a comma-separated ID list with no quoting inside the field;
- leave `matched_entity_ids` empty when there is no predicted match;
- contain no duplicate IDs inside one list;
- contain only S2/S3 IDs that actually exist in the test set;
- never contain an S1 ID as a prediction.

`output/candidate_pairs.tsv` must:

- contain exactly one row for every test S1 entity;
- use exactly `source1_entity_id` and `candidate_entity_ids`;
- contain the **final candidate set actually passed to the final matching model for inference**, not an earlier intermediate blocking output;
- contain no duplicate candidate IDs;
- contain only test S2/S3 IDs;
- allow an empty candidate list;
- contain every ID that appears in `matching_results.tsv`.

The internal `candidates.parquet` is the canonical candidate table from which the final `candidate_pairs.tsv` is generated. **Never independently regenerate `candidate_pairs.tsv` after scoring.** The exact same final candidate DataFrame must feed model inference and output writing.

**Candidate-set optimization rule:** `candidate_pairs.tsv` is a first-class final-submission artifact. Subject to maintaining a sufficiently high candidate recall / oracle ceiling, the team should prefer the configuration that produces **smaller candidate sets per Source 1 entity**. Candidate-generation experiments must therefore report both recall and candidate volume; do not compare configurations on recall alone.

### Evaluation

- The leaderboard scores `matching_results.tsv`, not `candidate_pairs.tsv`.
- The official metric is macro-averaged F0.5 over S1 entities.
- `F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`.
- Precision is weighted more heavily than recall; false merges are therefore expensive.
- Singleton/no-match S1 entities matter. Correctly predicting an empty list for a true singleton earns 1.0 for that entity; predicting a false match earns 0.0.
- Test labels are unavailable. All thresholding, model selection, and decision-rule tuning must therefore be based on training data with leakage-safe validation.
- The public leaderboard is feedback, not a training signal. Never tune thresholds/features/hyperparameters directly on leaderboard scores.

### Submission package

The final ZIP **must be exactly**:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── data/
│       │   ├── preprocessing/
│       │   ├── blocking/
│       │   ├── features/
│       │   ├── validation/
│       │   ├── models/
│       │   ├── decision/
│       │   ├── inference/
│       │   └── utils/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

`requirements.txt` belongs **inside** `code/business_entity_resolution/`, not in `output/` and not at ZIP root.

The `code/business_entity_resolution/` directory must be self-contained and runnable: a reviewer should be able to use the supplied training/test data and only this folder to reproduce both output TSVs. The README must contain exact commands and expected data layout. The requirements file must pin exact dependency versions used by the final clean-environment run.

### Fair-play / model constraints

- The official constraint is **no external data lookup / external data augmentation**. The challenge does **not** explicitly say that all APIs are prohibited.
- Do not use external services to enrich, resolve, geocode, look up, or translate challenge records unless the organizers explicitly confirm that use is allowed. In particular, an external translation API should not become a pipeline dependency by assumption, because it introduces an external service/data transformation and may fall under the external-data restriction.
- Use only the provided challenge data and computations derived from it for the default pipeline.
- A local/offline translation or language-processing component is a separate question: it may be evaluated only if its license/model constraints comply with the challenge rules and the organizers' rules permit it. It is **not required for V1**.
- Any advanced model adopted into the final pipeline must comply with the official model license/parameter constraints: MIT or Apache 2.0 and at most 8B parameters.
- Do not introduce a dependency or pretrained component whose license/weights violate the challenge rules.
- Keep a record of all external packages used and verify their licenses before packaging.

## 2. Architecture

```
Raw data
  → Normalization (conservative + aggressive, both kept)
  → Candidate generation (multiple routes, unioned, S1→pool and pool→S1)
  → Candidate recall / oracle evaluation (GATE)
  → **Freeze final candidate set**
  → Pairwise feature engineering on exactly that set
  → LightGBM binary classifier (Logistic Regression baseline first)
  → Entity-level decision layer (thresholds tuned on OOF, not top-1)
  → matching_results.tsv + candidate_pairs.tsv
  → Submission validator
  → ZIP packaging
```

Two hard gates in this pipeline, not soft suggestions:

- **Gate 1 (after candidate generation):** if oracle-ceiling F0.5 on validation is low, nothing downstream matters — go back to blocking, not to the model.
- **Gate 2 (after decision layer):** if OOF macro-F0.5 is stable across folds and countries (LOCO), further complexity needs to beat this by more than the noise floor (Section 12).

---

## 2A. Artifact-to-Submission Mapping

Internal artifacts and official submission files are not interchangeable.

| Internal artifact | Purpose | Final submission relation |
|---|---|---|
| `records.parquet` | normalized source records | not directly submitted |
| `folds.parquet` | frozen validation split | not directly submitted |
| `retrieval_events.parquet` | long-form per-route retrieval evidence | not directly submitted |
| `candidates.parquet` | final aggregated candidate pairs actually scored | source of `candidate_pairs.tsv` |
| `features.parquet` | model input rows for candidates | not directly submitted |
| `train_labels.parquet` | training labels | not directly submitted |
| `scores.parquet` | OOF/test scores + decision inputs | not directly submitted |
| `matching_results.tsv` | final predictions | submitted under `output/` |
| `candidate_pairs.tsv` | final candidate set actually scored | submitted under `output/` |

**Guardrail:** `candidate_pairs.tsv` is not "whatever Route 1/2/3 happened to retrieve." It is the **last candidate set immediately before final model inference**. If a later filtering/union stage changes the candidates, that later set is what must be written.

**Guardrail:** every final prediction must be traceable:

```text
matching_results.tsv
        ↓ candidate membership
candidate_pairs.tsv
        ↓ model input
features.parquet
        ↓ candidate provenance
candidates.parquet
        ↓ route provenance
retrieval_events.parquet
```

If this traceability breaks, stop and fix the pipeline before submitting.

## 3. Candidate Generation V1

All routes computed **globally**, not partitioned by country as a hard rule (see 3.8). Forward retrieval is S1→(S2∪S3); reverse retrieval is (S2∪S3)→S1. Union candidates by `(s1_id, candidate_id)`. Country is an input feature, never an assumption that the country universe is `{US, India}`.

**Candidate-generation guardrails:**
- Never construct a full S1×(S2+S3) dense similarity matrix.
- Every retrieval route must have an explicit cap/top-k/block-size control.
- Log candidate count and runtime by route before unioning.
- Keep route provenance in `retrieval_events.parquet`; do not collapse it prematurely.
- **Primary objective:** first satisfy the team-defined candidate-recall / oracle-ceiling gate; among configurations that satisfy it, **minimize candidate volume / maximize reduction ratio**.
- Report total candidate pairs and the per-S1 distribution: **mean, median, p95, p99, max**, plus the number of zero-candidate S1 entities.
- Prefer a smaller candidate set **per S1** when recall is comparable and the downstream model remains stable.
- A country mismatch is not an automatic rejection.
- A numeric mismatch is not an automatic rejection.
- A missing address is not an automatic rejection.
- A common exact name is not an automatic match.

### Route 1 — Exact normalized-name key
- **What:** group by conservative-normalized name (Unicode NFKC, casefold, punctuation→space, whitespace collapse).
- **Why:** catches Examples 3, 4, 6, 10 — identical or near-identical names with only case/spacing noise.
- **Implementation:** hash-map lookup, O(1). Country-aware: no — build globally, since a misassigned country shouldn't cost a true match.
- **Volume control:** cap block size (e.g. 200) to stop generic-name hubs from exploding.
- **Failure mode:** two unrelated entities with the same common name collide — exactly why exact-match is retrieval, not acceptance.

### Route 2 — Character n-gram TF-IDF, name
- **What:** 3–4 char n-grams, cosine top-k.
- **Why:** handles Examples 12 (typo), 16 (word reorder — moderate recall via shared n-grams).
- **Weak spot per evidence:** does **not** catch Example 69's NEXGILD case (zero character overlap) — this route alone is insufficient, which is why Routes 4/5 exist.
- **Metadata:** rank, cosine score. Starting k: 20.

### Route 3 — Token-based / rare-token name retrieval
- **What:** inverted index on name tokens, weighted by IDF; retrieve records sharing at least one low-document-frequency token.
- **Why:** catches Example 15 (`Digitalprivatepranya.Com` vs `Pranya Digital Private Limited`) — no shared substring pattern a char n-gram would rank highly, but shares the rare token "pranya". A different failure mode than Route 2; measure unique recall separately, don't assume redundancy.

### Route 4 — Character n-gram TF-IDF, address
- **What:** same as Route 2 but on address.
- **Why:** this is what recovers Example 69's NEXGILD case — the address string is nearly identical even though the name is not. **Not optional for this dataset; tier-A priority, same as name TF-IDF.**

### Route 5 — Numeric-token / address blocking
- **What:** index by the set of digit-tokens extracted from address (house numbers, plot/khata numbers — Examples 66/68 show these survive noisily across variants).
- **Why:** cheap, catches cases where the name is heavily corrupted but a plot number persists.
- **Caveat (Example 71):** don't require exact numeric match — digits themselves get corrupted (703→70). Use loosely (any shared numeric token) for retrieval, not as a hard feature rule.

### Route 6 — Rare-token blocking (address)
- **What:** same idea as Route 3, on address tokens (locality names, landmarks).
- **Why:** for India especially, locality/area names are often the most stable token when both name and house-numbering are noisy.

### Route 7 — Reverse retrieval (pool→S1)
- **What:** run TF-IDF/rare-token retrieval with S2/S3 as queries against the S1 pool, union results in.
- **Why:** a true S1 partner might rank outside top-k on the forward pass but inside top-k on the reverse pass. Also produces the raw material for the reverse-conflict decision rule (Section 9), since matching is verified one-to-one.

### Phonetic retrieval — not in V1
The evidence file contains no clear phonetic-only case (errors look like keyboard/OCR-style typos and truncation, not Soundex-style phonetic drift), and Soundex/Metaphone are English-only — risky for India, useless for France. Revisit only if Section 4's error analysis surfaces a specific missed-pair category phonetic retrieval would catch.

### 3.8 Country: feature, not a hard blocking rule
Do not filter candidates by country match. France is unseen, so any country-conditioned logic is untested there by construction; nothing in the evidence shows country as unreliable, but this can't be verified for France before test, and the downside of being wrong (silently losing all French recall) outweighs the upside (less candidate volume). Retrieve globally, add `country_eq` as a **feature**, and run one explicit experiment — hard-blocking vs. feature-only — before deciding. Even if hard-blocking shows zero recall loss on US/India, don't trust it for France; keep it feature-only for the submission.

---

## 4. Candidate Recall Evaluation

**Protocol:**

1. Build candidates on the validation fold only (never on data the model will be tuned against — see Section 5).
2. For every S1 entity in the validation fold, compute: pair recall, any-match recall, all-match recall, candidates per S1 (mean/median/p95/max), total candidate pairs, runtime, peak memory.
3. Break down by: 0-match S1 (measure candidate volume even here — that volume is false-positive risk), 1-match, 2–4-match, 5+-match, US, India, and (once test-only) France proxy via LOCO.
4. **Oracle ceiling:** assume a perfect classifier picks exactly the true matches that made it into the candidate set; compute macro-F0.5 under that assumption. The single most important number — it upper-bounds everything downstream.
5. Multilingual/noisy subset: tag validation entities where name or address in either record contains non-ASCII characters, and report recall separately. Also tag "mixed-script" separately from "fully non-Latin" given the Damani cluster's within-record script mixing.
6. **Candidate-volume release report:** always record total candidate pairs, mean/median/p95/p99/max candidates per S1, zero-candidate S1 count, and reduction ratio.

**"Good enough to move on" rule:** move to feature engineering when (a) any-match recall ≥ ~0.9 overall with no cliff in any breakdown slice, (b) oracle-ceiling macro-F0.5 clears the **team-internal recall gate**, and (c) candidate volume is not materially larger than necessary. If two configurations clear the recall/oracle gate, prefer the one with the smaller per-S1 candidate set / higher reduction ratio. If oracle ceiling and model score are already close, the bottleneck is the model, not blocking.

**Important:** the numeric oracle gate (for example, 0.92) is a **team-internal acceptance threshold**, not an Amazon requirement.

---

## 5. Validation Strategy (no leakage)

### 5A. Leakage Guardrails — Stop-the-Line Rules

The following are **stop-the-line violations**:

1. Using a held-out S1 entity's ground-truth match IDs to generate its candidates.
2. Computing a feature from whether a candidate was a true match.
3. Computing pool frequency from `train_ground_truth.tsv` rather than raw/candidate appearance.
4. Selecting a threshold using predictions from a model that trained on the same S1 entities.
5. Using a reverse-consistency score from a model/fold context that has seen either of the compared S1 labels.
6. Selecting a feature/model/config because it improved the leaderboard while the OOF result did not support it.
7. Treating test labels, if discovered indirectly, as training/validation information.
8. Building a hard country filter from the training countries and silently excluding France.

When in doubt, ask: **could this value have been computed without knowing the ground-truth match list for this S1?** If no, it cannot be a label-free global feature or blocking rule.


- **Split unit:** S1 entity, with its complete truth list traveling with it (verified one-to-one, so no connected-component merging needed across S1 entities).
- **S2/S3 handling:** shared pool, not split. Every fold's candidate generation runs against the entire training S2/S3 pool (mirrors test). Labels are attached after generation, using only that fold's held-out S1 truth.
- **Computed globally (label-free, safe):** normalization rules, TF-IDF vocabulary/IDF, rare-token document-frequency cutoffs, corpus-derived suffix/DBA-marker lists mined by frequency.
- **Fit only inside training folds:** the classifier, calibration, thresholds, any second-stage/reverse-conflict model consuming OOF scores.
- 5-fold GroupKFold by S1 entity, stratified by country × has-match × match-count bucket (the mode is 3 — don't let one fold skew toward singletons).
- Fit blocking + IDF once on the training universe (label-free), get per-fold OOF model predictions, tune thresholds on pooled OOF, report honest held-out numbers.

---

## 6. Feature Engineering — V1 vs. wait

**Build now (V1):**

- **Name:** char-cosine, token-set/token-sort similarity, IDF-weighted bidirectional containment (justified by Example 66's "Kalyani" subset match), exact-match flags (conservative + aggressive), length/token-count diff.
- **Address:** char-cosine, IDF-weighted bidirectional containment, numeric-token Jaccard, numeric-token conflict flag (softened per Example 71), length diff.
- **Country:** `country_eq` (equal/unequal/NaN) only — no identity one-hots.
- **Missingness:** explicit flags for name/address blank on each side — required by Example 71's blank-address true match; NaN-safe, not zero-fill.
- **Retrieval:** per-route rank/score (NaN if not retrieved), `n_routes`.

**Wait (add only if error analysis after V1 demands it):**

- Soft-TF-IDF / Jaro-Winkler token matching — not evidenced as necessary yet.
- Alias/DBA variant-max similarity — prototype given Example 70's `M/s Shirdi Corp Services`; hold until Section 10.
- Entity/pool-frequency features — useful for precision but needs a full-pool frequency table; add after V1 baseline exists.

**Missing vs. conflicting address:** missing = NaN feature value, model learns "no evidence" via its own split logic (LightGBM handles NaN natively — don't impute 0). Conflicting = both sides present, no numeric overlap — real feature (`addr_num_conflict`), but a soft signal per Example 71's digit-drop case, never a hard exclusion rule.

---

## 7. Baseline Model

- Build Logistic Regression first, briefly — a fast sanity check that features and labels are wired correctly, giving a floor to beat. No more than an hour.
- **Primary: LightGBM.**
- **Target:** binary, positive iff `(s1, rec)` is in that S1's ground-truth list.
- **Positives:** every true pair, including all matches for multi-match entities — don't subsample the 3–9 match tail, it's the mode of the distribution.
- **Negatives:** every non-true candidate that survived blocking — realistic hard negatives by construction, matching the inference distribution. Not random negatives by default; a random S2 record has near-zero overlap with most S1 names and teaches nothing about the real decision boundary (would never surface something as hard as Kalyani-subset or NEXGILD-address-only).
- **Class ratio:** report it, don't force-balance via resampling; consider per-entity example weighting (1 / candidates for that S1) so large candidate sets don't dominate the loss, since the eval metric is macro over entities.
- **Train/val:** the GroupKFold from Section 5, OOF predictions for calibration/threshold tuning.
- **Initial hyperparameters:** `learning_rate=0.05, num_leaves=31, min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1–10`, early stopping (~100 rounds patience) on average precision, with the two-pass "find rounds, then retrain honestly" procedure.
- **Evaluation:** report pairwise AUC/AP as a sanity check, but the metric that matters is entity-level macro-F0.5 after the decision layer (Section 9).

---

## 8. Decision Layer

### 8A. Decision Guardrails

- Never assume one match per S1. The model must be able to output zero, one, or many.
- Never output every candidate above a permissive threshold just to maximize recall; F0.5 is precision-heavy and singleton false positives are especially costly.
- Never use top-1 as the default decision rule.
- Every threshold is chosen on pooled OOF predictions, not training predictions and not leaderboard feedback.
- Any two-threshold or reverse-consistency rule must be evaluated as an entity-level rule on OOF predictions.
- A decision rule must be reproducible from saved configuration/model artifacts; no manual "looks right" edits to individual S1 entities.
- Before final inference, assert that every predicted ID exists in the final candidate set.
- After final inference, assert one output row per test S1 and deduplicate each predicted list.


**In V1:**

- Global threshold, tuned on OOF macro-F0.5 (not 0.5 by default).
- Two-threshold rule: lower threshold for accepting the *first* candidate per entity, higher threshold for additional candidates. F0.5's math makes abstaining on a true singleton worth a full point, and turning a true match from 0→1 (accepting one correct candidate) is worth more than the marginal gain of a second correct candidate — directly relevant given the matched-entity mode is 3.
- Reverse-consistency: if an S2/S3 record scores meaningfully higher against a different S1 entity, treat this as an **experimental conflict filter**, motivated by the train-only one-to-one observation. It must be enabled only if OOF evidence shows a precision gain without unacceptable recall loss; never assume the property holds on unseen test data.

**Hold as an experiment, not default:**

- Per-country thresholds — until US/India actually show different calibration; premature for France regardless.
- Score-margin/gap rules and top-k safeguards — test on OOF, keep only if they beat the two-threshold baseline outside noise.
- A learned second-stage entity model on OOF features — only after the simple rules plateau.

Always report the "predict nothing" baseline (should score exactly the singleton rate, ~5.58% on train) as the floor every experiment must clear.

---

## 9. Hard Negatives

Example 68's cluster (`Damani Technologies`, `Damani Technologies L.L.P.`, `Sri Damani LLP Technologies`, all true matches to the same S1) is itself the argument for why retrieved-candidate negatives are non-negotiable: any near-miss S2/S3 record sharing that address but belonging to a *different* S1 (guaranteed to exist as an orphan somewhere in the pool, given the one-to-one structure) is exactly the hard negative a random-negative strategy would never surface.

- **Identify negatives:** every candidate a route retrieves for a given S1 that is not in that S1's truth list. Safe because it relies only on the ground truth file, never on model output.
- **One-to-many handling:** all listed IDs for an S1 are positive; everything else retrieved and not listed is a negative.
- **Avoiding accidental positive-as-negative:** never derive negatives from similarity scores or the model's own predictions. After training v1, manually review the highest-scoring false positives; if any look like true matches, treat it as a ground-truth-completeness question to raise with the team, not something to silently relabel.
- **Volume:** use all retrieved negatives per entity by default; if too large, cap per-entity by fused retrieval score, dropping the easiest (lowest-score) negatives first.
- **Random negatives:** skip for the main training set; optional small-scale ablation only.

---

## 10. Multilingual / France Strategy

**Can evaluate now:** Leave-one-country-out (train US → validate India, and reverse) is the only real proxy for unseen-country transfer, and the evidence file already shows script-mixing *within* India (Damani cluster's Devanagari locality inside an otherwise-Latin address) — so LOCO isn't just a France proxy, it directly tests something the data already contains.

**Cannot know before test labels:** whether French legal-suffix patterns (SARL, SAS, EURL — visible in the test audit samples) behave like US/India suffixes for matching purposes, or whether French address formatting breaks numeric-token or containment features in some new way.

**Language-independent operations (safe to rely on):**
- Unicode NFKC normalization, case-folding, accent-folding — script-agnostic by construction.
- Character n-gram similarity — works on any script without needing to know what the script is.
- Numeric-token extraction — digits are digits regardless of language.
- IDF-weighted containment — doesn't require understanding the tokens, just their corpus frequency.

**Higher-risk / avoid depending on:**
- Any suffix or DBA-marker dictionary — mine it from each file's own corpus statistics, never hard-code an English/Indian list and hope it covers French too.
- External translation/transliteration services — do not depend on them unless the organizers explicitly confirm they are allowed under the external-data rule. The challenge does not explicitly ban all APIs; the issue is external lookup/augmentation.
- Local/offline translation or transliteration — not required for V1. Consider only as a later experiment if error analysis demonstrates a real translation-dependent recall gap and the component satisfies the license/model constraints and challenge rules.
- Do not assume translation is necessary: Unicode normalization, character n-grams, numeric tokens, containment, and address-based retrieval already provide language-independent matching signals.
- Multilingual embeddings — only test once the lexical baseline is stable with hours to spare (Section 11, E9); treat as an experiment, not a default.

**Avoiding overfitting to US/India:** exclude country-identity features (Section 3.8), monotone-constrain similarity features in LightGBM so "more similar → more likely match" holds regardless of country, and accept that France-specific tuning is impossible — ship the LOCO-robust version.

---

## 11. Experiment Ladder

| # | Hypothesis | Change | Metric | Failure mode | Keep if | Discard if |
|---|---|---|---|---|---|---|
| E0 | Pipeline wired correctly | Minimal blocking (exact + 1 TF-IDF route) + LR | OOF macro-F0.5 vs "predict nothing" | Format bugs, ID mismatches | Beats "predict nothing" clearly | N/A — correctness check |
| E1 | Candidate Gen V1 (7 routes) clears the recall gate efficiently | Full route union + volume caps | Any-match recall, oracle ceiling, median/p95/max candidates/S1, reduction ratio | Recall plateau or excessive candidate volume | Recall gate met with the smallest practical candidate set | A route adds little unique recall at high volume cost |
| E2 | V1 features beat char-cosine-only | Add containment, numeric, missingness | OOF pairwise AP + entity F0.5 | Overfitting to train-only patterns | Entity F0.5 improves outside noise floor | No change beyond noise |
| E3 | LightGBM beats LR | Same features, model swap | OOF macro-F0.5 | LGBM overfits small positive set | Clear, stable gain across folds | Marginal/unstable gain |
| E4 | Hard negatives beat random negatives | Negative source swap | OOF macro-F0.5, precision | Precision doesn't actually improve | Precision rises without recall collapse | No measurable change |
| E5 | Two-threshold decision beats global threshold | Decision rule swap | OOF macro-F0.5 | Overfits to validation's match-count mix | Stable gain, especially on multi-match entities | No gain or fold-unstable |
| E6 | Reverse-consistency improves precision | Add rule using one-to-one structure | OOF macro-F0.5, FP rate | Suppresses true matches when scores are close | FP rate drops, TP rate roughly held | Recall drops more than FP saves |
| E7 | LOCO gap is small | Train US, test India (and reverse) | Cross-country OOF F0.5 gap | Large, unexplained gap | Gap small → lower France risk | Large gap → prioritize robustification before anything else |
| E8 | Rare-token/numeric address routes earn their volume cost | Ablate Routes 5/6 | Unique recall per route | Redundant with TF-IDF routes | Meaningful unique recall (motivated by NEXGILD-style case) | Fully redundant |
| E9 | Advanced retrieval/model beats the frozen strong system | Embeddings/cross-encoder | OOF macro-F0.5, paired bootstrap CI | Doesn't clear noise floor, eats remaining hours | Statistically real, positive-CI gain | Anything else — keep the simple system |

Stop rule: any experiment whose gain doesn't clear the noise floor (rerun the baseline with different seeds to measure it) gets discarded, not debated.

---

## 11A. Final Repository and Production Source Contract

The development repository and the submission ZIP are intentionally different layers.

### Development repository

```text
amazon-ml-challenge/
├── src/                         # production source; mirrors final submission src/
│   ├── __init__.py
│   ├── data/
│   ├── preprocessing/
│   ├── blocking/
│   ├── features/
│   ├── validation/
│   ├── models/
│   ├── decision/
│   ├── inference/
│   └── utils/
├── tests/                       # development only
├── configs/                     # development-only configs
├── docs/                        # development notes/contracts
├── logs/                        # person1..4 experiment logs + submission history
├── experiments/                # tracker/results; development only
├── scripts/                    # development/packaging helpers
├── README.md                   # team/development README
├── .gitignore
└── requirements-dev.txt        # optional; not the submission requirements file
```

### Development `.gitignore` rule

Do **not** use a blanket `*.tsv` ignore rule because the final submission requires `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Ignore data/artifacts and temporary TSVs specifically, for example:

```gitignore
dataset/
artifacts/
*.parquet
__pycache__/
.venv/
submission_staging/
tmp/
tmp/**/*.tsv
```

The two official files under `output/` must remain trackable/packageable.

### Final production source tree

The source that is actually packaged is:

```text
code/business_entity_resolution/
├── src/
│   ├── data/
│   │   └── loaders.py
│   ├── preprocessing/
│   │   ├── normalize.py
│   │   └── corpus_stats.py
│   ├── blocking/
│   │   ├── exact_name.py
│   │   ├── tfidf_name.py
│   │   ├── tfidf_address.py
│   │   ├── rare_token.py
│   │   ├── numeric.py
│   │   └── candidate_generation.py
│   ├── features/
│   │   ├── name_features.py
│   │   ├── address_features.py
│   │   └── build.py
│   ├── validation/
│   │   ├── metrics.py
│   │   └── splits.py
│   ├── models/
│   │   ├── logreg.py
│   │   └── lightgbm_model.py
│   ├── decision/
│   │   └── threshold.py
│   ├── inference/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── pipeline.py
│   └── utils/
│       ├── io.py
│       └── submission.py
├── README.md
└── requirements.txt
```

This is a target architecture, not a demand to keep unused files. If two production modules are naturally combined, combine them. Do not create empty modules merely to satisfy a tree diagram.

### Ownership boundaries

- **P1:** `src/blocking/`
- **P2:** `src/preprocessing/` and `src/features/`
- **P3:** `src/validation/`, `src/models/`, and `src/decision/`
- **P4:** `src/data/`, `src/inference/`, and `src/utils/`
- P4 also owns the development-only packaging/build scripts, submission/version tracking, logs coordination, and final README/methodology assembly.
- Nobody rewrites another person's ML logic. Cross-owner changes require a short PR/review and an experiment-tracker entry when behavior changes.

### Development-only vs submission-only distinction

The final ZIP does **not** need the team's tests, logs, configs, experiment tracker, raw Parquet artifacts, notebooks, or temporary subsets unless a specific artifact is required for reproducibility. Those can remain in the development repository/shared storage.

Conversely, `README.md`, `requirements.txt`, and `src/` are mandatory in the final `code/business_entity_resolution/` folder.

The final package builder must construct the ZIP from a clean staging directory rather than zipping the whole Git repository.

### Production execution contract

The production pipeline must expose one reproducible path equivalent to:

```text
training data
    ↓
loaders
    ↓
normalization / corpus statistics
    ↓
candidate generation
    ↓
candidate recall evaluation / training labels
    ↓
**freeze final candidate DataFrame**
    ↓
pairwise features (computed for exactly the frozen candidates)
    ↓
OOF training + threshold selection
    ↓
final retrain
    ↓
test candidate generation
    ↓
test features
    ↓
test scoring + decision
    ↓
output/matching_results.tsv
output/candidate_pairs.tsv
```

The exact CLI may be chosen by P4, but it must be documented verbatim in the final README and must not depend on undocumented notebook state.

## 12. Team Responsibilities

| Person | Owns | Deliverables | Depends on | Doesn't touch |
|---|---|---|---|---|
| **1 — Candidate Generation** | All 7 routes, union, reverse retrieval, recall/oracle-ceiling harness | `candidates.parquet` schema, recall table (Section 4) | Normalized text columns from Person 2 (works against a stub normalizer first) | Feature code, model code |
| **2 — Normalization / Features** | Conservative + aggressive normalization, corpus-derived suffix/DBA mining, all Section 6 features | `records.parquet` (normalized columns), `features.parquet` | Candidate pairs from Person 1 | Blocking logic, model training |
| **3 — ML / Validation / Thresholding** | Folds, metric, LR/LightGBM, OOF, calibration, decision layer, LOCO | `scores.parquet`, tuned thresholds, honest CV report | Features from Person 2 | Blocking, submission format |
| **4 — Pipeline / Inference / Submission / Docs** | `src/data/`, `src/inference/`, `src/utils/`; loaders, schemas (frozen early), inference entrypoint, TSV writer, validator integration, experiment tracker, submission-version log, README/docs, licensing check, exact ZIP packaging | Runnable end-to-end script, frozen-candidate handoff, candidate-volume release report, clean-environment reproduction, valid submission ZIP | Everyone's interfaces | Feature/model logic itself |

**Shared production repo structure:**

```text
src/
├── data/          # P4: loaders.py
├── preprocessing/ # P2: normalize.py, corpus_stats.py
├── blocking/      # P1: one module per retrieval route + candidate_generation.py
├── features/      # P2: name/address/retrieval/country/missingness features
├── validation/    # P3: metrics.py, splits.py
├── models/        # P3: logreg.py, lightgbm_model.py
├── decision/      # P3: threshold.py and any adopted decision rules
├── inference/     # P4: train.py, predict.py, pipeline.py
└── utils/         # P4: io.py, submission.py

tests/             # development only
configs/           # development only
docs/              # development only
logs/              # development only
experiments/       # development only
scripts/           # development/packaging only
```

Nobody edits another person's production directory without coordination. Everyone codes against the frozen `records.parquet` / `candidates.parquet` / `features.parquet` / `scores.parquet` schemas agreed in Hour 0–4.

---

## 13. 72-Hour Timeline

**First 4 hours:** Confirm assumptions from the audits (already largely done). Freeze folds. Metric implemented and unit-tested. Repo skeleton + schemas frozen. Team reads the actual example file together — 20 minutes, pick 10 examples, discuss what feature would catch each one.

**First 12 hours:** Candidate Gen V1 (Routes 1–4 minimum) running end-to-end on a train subset, with recall numbers. Features V1 (name/address char-cosine + containment + missingness) computed. LR baseline trained. A complete, ugly, valid `matching_results.tsv`/`candidate_pairs.tsv` exists.

**First 24 hours:** Full 7-route candidate generation with recall table filled with real numbers (Section 4). LightGBM trained with OOF. Global-threshold decision layer. First honest macro-F0.5 number. E0–E3 from Section 11 done.

**First 36 hours:** Two-threshold decision layer, hard negatives properly wired (E4–E5), reverse-consistency rule (E6), LOCO run (E7) — the most important checkpoint: know the France risk by hour 36, not hour 60.

**First 48 hours:** Error analysis using misses from LOCO and from the multilingual/multi-match validation slices; targeted fixes only (E8, DBA/alias features if warranted by the team's own error review). Freeze candidate generation.

**Final 24 hours:** No new techniques after roughly hour 60. Final CV run, retrain on full train, generate test predictions, run the validator, write README/methodology/licensing docs, package the ZIP, dry-run from a clean environment, submit with buffer remaining.

---

## 13A. Submission Version Tracking

P4 maintains `logs/submissions.md` for every actual submission attempt. Each entry should record:

```text
submission_id
timestamp
git_commit
candidate_generation_version
model_version
decision_thresholds
total_candidate_pairs
mean/median/p95/p99/max candidates per S1
validator_status
leaderboard_result (when available)
notes
```

The purpose is traceability, not leaderboard-driven tuning. Public leaderboard feedback may be recorded, but model, threshold, and candidate-generation decisions must remain grounded in the leakage-safe validation protocol.

---

## 14. AWS Usage

Local development covers essentially everything — this is a batch tabular-features + LightGBM pipeline on CPU, not something that needs managed infrastructure. Where AWS genuinely helps: (1) a single SageMaker notebook instance with more RAM/CPU than a laptop, used for the full 10M+-row candidate generation and feature-building runs once the pipeline is correct on a subsample locally; (2) S3 as shared storage for the frozen `records.parquet`/`candidates.parquet`/`features.parquet` artifacts so all four people work from the same generated files instead of regenerating them differently. No Lambda, API Gateway, Bedrock, or Kubernetes — none of those change whether the team can produce a correct, validated submission.

---

## 14A. Final Packaging and Clean-Room Reproduction

### Staging directory

P4 must create a temporary staging directory such as:

```text
submission_staging/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

Do **not** zip the Git repository directly.

### README requirements

`code/business_entity_resolution/README.md` must state:

1. expected Python/environment version;
2. exact installation command;
3. exact training command;
4. exact inference/test command;
5. expected input directory layout;
6. where intermediate artifacts are written, if any;
7. where the two official TSV outputs are written;
8. how to run the official validator;
9. any required command-line arguments;
10. the exact dependency file used.

The README must describe the actual final pipeline, not an aspirational architecture. The execution location must also be explicit so the required root-level `output/` directory is never confused with `code/business_entity_resolution/output/`.

**Team clean-room convention:** from `submission_staging/`, enter `code/business_entity_resolution/` and run the production module with the root-level output path: `--output-dir ../../output/`. The same convention must be used in the README and clean-room harness.

The methodology write-up should follow the stricter **1–2 page team submission target**, with dense technical content covering the approach, models, experiments, and conclusion.

### Requirements requirements

`code/business_entity_resolution/requirements.txt` must:

- pin exact versions (`==`);
- contain only dependencies actually needed by the packaged pipeline;
- be tested in a clean environment;
- have license compatibility checked;
- not depend on an untracked local package or developer machine path.

### Clean-room test

Before the final ZIP:

1. Create a fresh virtual environment on a clean checkout/staging directory.
2. Install only the pinned requirements.
3. Provide the official training/test data at the documented paths.
4. Enter `submission_staging/code/business_entity_resolution/` and run the documented production command, writing outputs to the sibling root `../../output/`.
5. Regenerate both TSV outputs.
6. Run the official validator from the staging root using the root-level `output/` files.
7. Compare the regenerated files against the intended final outputs and investigate any unexplained difference.
8. Only then create the ZIP.

Canonical clean-room invocation:

```bash
cd submission_staging/code/business_entity_resolution
python -m src.inference.pipeline \
    --train-dir /path/to/train \
    --test-dir /path/to/test \
    --output-dir ../../output/

cd ../..
python code/business_entity_resolution/src/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir /path/to/test
```

### Official validator vs team validator

The challenge provides an official `utils/validate_submission.py` helper. Use that official validator for the final pre-submission check. Our own `src/utils/validate_submission.py` may provide additional pipeline assertions, but it must not be treated as a replacement for the official validator unless its behavior is proven equivalent.

## 15. Submission Checklist

### Output correctness
- [ ] `output/matching_results.tsv` exists.
- [ ] `output/candidate_pairs.tsv` exists.
- [ ] Both are TSVs with exact official column names.
- [ ] Exactly one matching row per test S1 entity.
- [ ] Exactly one candidate row per test S1 entity.
- [ ] Empty match/candidate lists are empty strings, not NaN/None/text placeholders.
- [ ] No duplicate IDs inside any ID list.
- [ ] No duplicate `source1_entity_id` rows.
- [ ] Every predicted ID is an S2/S3 test ID.
- [ ] No S1 ID is ever predicted.
- [ ] Every predicted ID is present in that S1's candidate list.
- [ ] `candidate_pairs.tsv` is the exact final candidate set actually scored.
- [ ] Candidate-volume release report is recorded: total pairs, mean/median/p95/p99/max per S1, zero-candidate S1 count, and reduction ratio.

### Pipeline correctness
- [ ] Candidate generation has documented recall/oracle-ceiling results.
- [ ] The final candidate DataFrame is explicitly frozen before feature computation/model inference.
- [ ] Features are computed for exactly the final candidate set.
- [ ] OOF flags are correct.
- [ ] Threshold/decision rules were tuned only on OOF training predictions.
- [ ] No ground-truth-derived leakage exists in blocking/features.
- [ ] Final model is retrained according to the frozen plan.
- [ ] Test inference uses the frozen pipeline/configuration.
- [ ] France is not excluded by any hard-coded country logic.

### Submission structure
- [ ] ZIP name is `<team_name>_submission.zip`.
- [ ] `output/` contains exactly the two required TSVs.
- [ ] `code/business_entity_resolution/src/` contains the production source.
- [ ] `code/business_entity_resolution/README.md` contains exact reproduction instructions.
- [ ] `code/business_entity_resolution/requirements.txt` exists and pins exact versions.
- [ ] `Documentation_template.md` is at the ZIP root and is filled in.
- [ ] No accidental dataset dump, notebook state, secrets, API keys, or unrelated files are packaged.

### Rules / compliance
- [ ] Official validator passes with no warnings.
- [ ] Every dependency/model license is compatible with the challenge.
- [ ] Final model is within the ≤8B parameter constraint.
- [ ] No external data lookup or external data augmentation is used.
- [ ] No external translation/geocoding/entity-resolution service is used unless the organizers explicitly confirmed that use is permitted.
- [ ] Any local/offline language model or preprocessing component used in the final pipeline has a compatible license and complies with the challenge's model constraints.
- [ ] Clean-environment reproduction succeeds from only the packaged code plus challenge data.
- [ ] Submission version is recorded (timestamp, commit/hash, candidate-generation version, model/decision version, candidate count, validator status, and leaderboard result if a submission was made).


---

## 16. Do-Not-Do-Yet List

- Transformers, cross-encoders, contrastive fine-tuning — not evidenced as necessary; the hard examples (NEXGILD, Kalyani) are solved by which routes retrieve candidates, not by a fancier scorer.
- Multilingual embeddings — hold for E9 only if the lexical system plateaus with hours to spare.
- Phonetic retrieval — no evidence in the examples motivates it yet.
- Per-country thresholds — premature without France data to calibrate against.
- Massive hyperparameter sweeps on LightGBM — a handful of configs is the right scale for 72 hours.
- Any AWS service beyond one notebook instance + S3.
- Any external translation/transliteration service or resource unless the organizers explicitly confirm it is permitted.
- Do not add local/offline translation/transliteration to V1; revisit only if error analysis proves it is needed and the rules/license permit it.
- Second-stage/cross-encoder entity model — only after the simple two-threshold decision layer plateaus.

---

## 16A. Source-of-Truth Hierarchy for the Team

When two documents or notes appear to disagree, use this order:

1. **Official challenge problem statement / submission rules** — highest authority.
2. **This Master Plan** — team strategy and evidence-driven design, provided it does not conflict with the official rules.
3. **Final 4-Person Execution Plan** — hour-by-hour implementation schedule, handoffs, and freeze points.
4. **Experiment tracker / logs** — record what was actually tested and which changes were adopted.
5. Informal chat messages / temporary notes — lowest authority.

If an experiment discovers that the current plan is wrong, do not silently change code. Log the hypothesis/result, get team agreement, update the relevant plan/config, and then implement the change.

## 17. Immediate Next 3 Actions

1. Confirm the schema contracts (`records.parquet`, `candidates.parquet`, `features.parquet`, `scores.parquet`) as a team, in writing, before anyone writes pipeline code.
2. Person 3 implements and unit-tests the macro-F0.5 metric (including singleton and empty-prediction edge cases) and freezes the GroupKFold split — this blocks everyone else's honest evaluation.
3. Person 1 starts Candidate Gen V1 with Routes 1, 2, and 4 (exact name key, name TF-IDF, address TF-IDF) against a train subsample, and reports **recall plus candidate-volume statistics** — this validates both the address-retrieval requirement and the new candidate-size objective.
4. Person 4 defines the exact final-candidate handoff: one frozen candidate DataFrame must feed both P2 features/model inference and `candidate_pairs.tsv`; add assertions before production runs.

---

# IF WE WERE STARTING RIGHT NOW

1. **All four, together (30 min):** Reread 8–10 examples from `training_match_examples.md` as a group, specifically the NEXGILD (Ex. 69), Kalyani (Ex. 66), and blank-address (Ex. 71) cases, and agree out loud on why each one requires a specific route or feature. This is the shared mental model the rest of the plan depends on.
2. **Person 4:** Create the repo and production `src/` tree, freeze the four schema contracts, write `src/data/loaders.py` with `dtype=str, keep_default_na=False, quoting=QUOTE_NONE`.
3. **Person 3:** Implement and unit-test the F0.5 metric; generate and save the frozen GroupKFold split.
4. **Person 2:** Write conservative normalization; start corpus-frequency suffix/DBA-marker mining (don't hard-code a list).
5. **Person 1:** Implement Route 1 (exact name) and Route 4 (address TF-IDF) first in `src/blocking/` — deliberately address before the "obvious" name routes, since the evidence shows address recall is the harder-won, more critical piece — then report first recall numbers against a train subsample.
6. **Reconvene once Person 1 has recall numbers and Person 3 has the metric working:** that's the trigger to start Person 2's features and Person 3's LR baseline in parallel.
