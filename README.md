# Short-Text Classification with Naive Bayes

## 1) Project Overview

This project builds a complete, reproducible experiment pipeline for multiclass short-text classification using Naive Bayes models on **three separate datasets**:

- `banking`
- `snips`
- `stackoverflow`

Each dataset is treated as an independent task (no cross-dataset merging).

For each dataset, the pipeline:
1. Loads `train/dev/test` TSV files.
2. Validates schema and rows.
3. Applies light preprocessing.
4. Trains and compares vectorizer + Naive Bayes configurations.
5. Selects the best configuration on **dev Macro-F1**.
6. Uses **Option B** model selection strategy:
   - Choose best config from train→dev search.
   - Retrain best config on **train+dev**.
   - Evaluate once on **test**.
7. Saves reports, metrics, and plots into `outputs/`.

---

## 2) Dataset Folder Structure

Expected structure:

```text
project_root/
  data/
    banking/
      train.tsv
      dev.tsv
      test.tsv
    snips/
      train.tsv
      dev.tsv
      test.tsv
    stackoverflow/
      train.tsv
      dev.tsv
      test.tsv
```

Each TSV must contain at least these columns:

- `text`
- `label`

> Note: For convenience, the code also supports a legacy layout where dataset folders are directly under project root (`./banking`, `./snips`, `./stackoverflow`).

---

## 3) Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies are intentionally minimal:

- pandas
- numpy
- scikit-learn
- matplotlib

---

## 4) How to Run

From the project root:

```bash
python main.py
```

The script will run all combinations on each dataset and write outputs under `outputs/`.

---

## 5) Compared Models and Vectorizers

### Vectorizers
- `CountVectorizer` with `ngram_range=(1,1)`
- `CountVectorizer` with `ngram_range=(1,2)`
- `TfidfVectorizer` with `ngram_range=(1,1)`
- `TfidfVectorizer` with `ngram_range=(1,2)`

### Classifiers
- `MultinomialNB`
- `BernoulliNB`
- `ComplementNB`

### Hyperparameters
- `alpha` in `[0.1, 0.5, 1.0, 2.0]`
- `min_df` in `[1, 2, 3]`

Invalid combinations are caught safely and logged in the experiments table with `status=failed`.

---

## 6) Preprocessing

Default preprocessing (light and reproducible):
- lowercase
- strip leading/trailing spaces
- collapse repeated whitespace

Optional switches (available in `PreprocessConfig`):
- remove punctuation
- remove digits
- remove English stopwords

No stemming/lemmatization is used by default.

---

## 7) Model Selection Logic

Primary selection metric: **dev Macro-F1**.

Tie-breaking order:
1. Higher dev Accuracy
2. Simpler n-gram setting (`(1,1)` preferred over `(1,2)`)
3. Simpler vectorizer (`Count` preferred over `TF-IDF`)

Final evaluation policy: **Option B**.
- Search config on train/dev.
- Retrain best config on **train+dev**.
- Evaluate once on test.

---

## 8) Output Artifacts

For each dataset (`outputs/<dataset>/`):

1. `all_experiments.csv`
   - One row per configuration with dev metrics.
2. `best_config.json`
   - Chosen best config and dev metrics.
3. `test_metrics.json`
   - Final test metrics.
4. `classification_report.txt`
   - Full sklearn classification report.
5. `confusion_matrix.png`
   - Confusion matrix plot for best model.

Global outputs (`outputs/`):

- `summary_results.csv` (best settings + key metrics per dataset)
- `best_test_macro_f1_barplot.png` (optional comparative plot)
- `analysis.txt` (short easiest/hardest dataset note)

---

## 9) Why Naive Bayes for Short Text

Naive Bayes is a strong baseline for short-text classification because:

- It works well on sparse bag-of-words features.
- It is fast to train and predict.
- It is robust with limited data and many classes.
- It often provides competitive performance with simple preprocessing.

This makes it practical for intent classification tasks like banking queries, SNIPS intents, and StackOverflow question tags.

---

## 10) Project Code Structure

```text
src/
  data_loader.py   # Loading TSV files + validation
  preprocess.py    # Text preprocessing
  features.py      # Vectorizer creation
  models.py        # Naive Bayes model creation
  evaluate.py      # Metrics, reports, confusion matrix, bar plot
  experiment.py    # End-to-end experiment loop + selection + outputs
  utils.py         # IO and logging helpers
main.py            # Entry point
```
