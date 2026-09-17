# Detection of AI-Generated Indonesian News Using Multi-Feature Fusion

[![Research status](https://img.shields.io/badge/status-research%20archive-2f6f9f)](#project-status)
[![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)](#reproducibility)
[![Task](https://img.shields.io/badge/task-AI--generated%20text%20detection-6f42c1)](#research-overview)
[![Domain](https://img.shields.io/badge/domain-Indonesian%20news-b65f00)](#datasets)

This repository contains the research code, selected data, trained artifacts, evaluation outputs, and demonstration application developed for detecting **AI-generated Indonesian news text**. The study combines stylometric, probabilistic, lexical, and contextual-semantic signals and evaluates them with decision-tree ensemble classifiers.

The repository is intended as a research archive that supports inspection of the experimental workflow and limited reproducibility of the reported results.

## Research overview

Recent language models can produce Indonesian news articles that closely resemble human writing. Detectors based on only one signal may perform well in-distribution but become less reliable when the publisher, writing style, or text generator changes.

This study investigates whether complementary feature families improve detection across both internal and external evaluation settings. The main research questions are:

1. How accurately can multi-feature fusion distinguish human-written and AI-generated Indonesian news?
2. How does each feature family contribute across Random Forest, XGBoost, and CatBoost?
3. Does the final model generalize to news publishers and text generators that differ from those represented during training?
4. Which errors can be recovered by alternative feature combinations or classifiers?

## Main contributions

- A **469-dimensional multi-feature representation** combining four complementary feature families.
- A controlled comparison of **all 15 non-empty feature combinations** across three ensemble classifiers, resulting in 45 evaluated configurations.
- Leakage-aware evaluation using **5-fold GroupKFold** with `source_id` as the grouping key.
- External validation across three publishers—Kompas, Liputan6, and Detik—and three generators—GPT, Gemini, and Claude.
- Out-of-fold error analysis covering feature-configuration recovery, cross-model recovery, and persistent misclassifications.

## Method summary

The final representation combines:

| Feature family | Purpose | Initial dimension | Processing | Final dimension |
| --- | --- | ---: | --- | ---: |
| Stylometry | Writing style and document structure | 18 | Standardization | 18 |
| Perplexity | Token-sequence predictability from Indonesian GPT-2 | 1 | Standardization | 1 |
| TF-IDF | Lexical and short-phrase patterns | Up to 1,000 | Truncated SVD | 300 |
| IndoBERT | Contextual-semantic information | 768 | PCA | 150 |
| **Full fusion** | Concatenation of all representations | — | Feature concatenation | **469** |

The general workflow is:

```text
Paired human–AI documents
        ↓
Text preprocessing and minimum-length filtering
        ↓
Stylometry + Perplexity + TF-IDF + IndoBERT
        ↓
Truncated SVD and PCA for high-dimensional features
        ↓
Feature concatenation and fold-specific standardization
        ↓
Random Forest / XGBoost / CatBoost
        ↓
Internal GroupKFold evaluation
        ↓
External cross-source and cross-generator validation
        ↓
Ablation and error analysis
```

## Datasets

| Dataset | Role | Size used in evaluation | Composition |
| --- | --- | ---: | --- |
| M4 Indonesian news partition | Training and internal cross-validation | 5,979 documents | 2,979 human; 3,000 AI |
| External paired news dataset | Cross-source and cross-generator evaluation | 1,640 documents | 820 human; 820 AI |

### Internal dataset

The internal corpus is derived from the Indonesian news partition of the M4 benchmark. Each human-written article is paired with an AI-generated article describing the same event. After minimum-length filtering, 5,979 documents are used for modeling.

`source_id` identifies documents that describe the same event. It is also used as the GroupKFold key so that paired human and AI documents never appear simultaneously in the training and validation subsets of a fold.

### External dataset

The external dataset contains paired news texts from Kompas, Liputan6, and Detik. Its AI-generated documents were produced using GPT, Gemini, and Claude. Gemini and Claude were not represented in the internal training corpus.

The dataset record is available on Zenodo:

- **DOI:** [10.5281/zenodo.22006290](https://doi.org/10.5281/zenodo.22006290)
- **License:** CC BY 4.0, as specified in the Zenodo record

### Data files in this repository

| Path | Description |
| --- | --- |
| `data/dataset_berita_clean.csv` | Main internal dataset used by the notebook |
| `data/berita_eksternal_clean_long.csv` | External evaluation dataset |
| `data/id-newspaper_chatGPT (1).jsonl` | Supporting provenance data for the internal corpus |

> [!IMPORTANT]
> Dataset availability in this repository does not override the license or redistribution terms of the original source. Verify the provenance and license of each file before redistribution. When a dataset cannot be redistributed, provide retrieval instructions or metadata instead of uploading the source data.

## Evaluation protocol

- **Validation:** 5-fold GroupKFold
- **Grouping key:** `source_id`
- **Random seed:** `42`
- **Primary metric:** F1-macro
- **Additional metrics:** accuracy, ROC-AUC, precision, recall, training F1-macro, and train–validation gap
- **Reported variation:** mean ± sample standard deviation across folds
- **Model comparison:** Random Forest, XGBoost, and CatBoost
- **Feature analysis:** 15 non-empty feature combinations × 3 classifiers
- **Final external test:** 1,640 independently collected paired documents

Data-dependent transformations—including TF-IDF fitting, Truncated SVD, PCA, imputation, and scaling—must be fitted only on the training portion of each fold to prevent leakage.

## Key results

### Internal full-fusion evaluation

| Model | F1-macro | ROC-AUC | Train–validation gap | OOF errors |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.9722 ± 0.0027 | 0.9967 ± 0.0006 | 0.0245 ± 0.0025 | 166 |
| XGBoost | 0.9828 ± 0.0017 | **0.9986 ± 0.0004** | 0.0172 ± 0.0017 | 103 |
| **CatBoost** | **0.9838 ± 0.0015** | 0.9983 ± 0.0006 | **0.0162 ± 0.0015** | **97** |

The reference zero-shot-style perplexity-threshold baseline reaches an out-of-fold F1-macro of **0.8947** and a ROC-AUC of **0.9460**.

### External evaluation of the final model

| Metric | Result |
| --- | ---: |
| Documents | 1,640 |
| F1-macro | 0.9866 |
| Accuracy | 0.9866 |
| Balanced accuracy | 0.9866 |
| ROC-AUC | 0.9992 |
| Precision, AI class | 0.9819 |
| Recall, AI class | 0.9915 |
| Errors | 22 |

### Feature and error-analysis findings

- Full fusion provides the highest F1-macro for CatBoost and XGBoost.
- Random Forest performs best with stylometry and perplexity rather than full fusion.
- Perplexity is the weakest individual feature but causes the largest decrease when removed from the CatBoost fusion, indicating a strong marginal contribution.
- TF-IDF contributes the smallest marginal gain within full fusion because part of its lexical information is already represented by IndoBERT.
- Among the 97 full-fusion CatBoost errors, 68, 82, and 90 documents are recovered by at least one three-feature, two-feature, and single-feature configuration, respectively.
- Considering all alternative feature configurations jointly, 92 errors are recovered and five documents remain misclassified under every configuration.
- Sixty-six CatBoost errors are shared by all three classifiers, while 31 are corrected by Random Forest, XGBoost, or both.

## Repository structure

```text
AI-Text-Detection/
├── notebooks/       # Main research notebook and experimental workflow
├── data/            # Selected internal and external data
├── models/          # Trained model artifacts
├── results/         # Metrics, checkpoints, predictions, and error analyses
├── figures/         # Figures generated from the experiments
├── scripts/         # Supporting and model-export scripts
└── apps/            # Web-based research demonstration
```

## Main notebook

The main experimental workflow is provided in:

```text
notebooks/AI_Detection_revisi_bab4_terurut_github.ipynb
```

The notebook covers:

- dataset loading and validation;
- text preprocessing;
- extraction of stylometry, perplexity, TF-IDF, and IndoBERT features;
- dimensionality reduction with Truncated SVD and PCA;
- construction of all non-empty feature combinations;
- GroupKFold evaluation of Random Forest, XGBoost, and CatBoost;
- external validation;
- out-of-fold prediction and error analysis; and
- model export for the demonstration application.

Run the notebook cells in order. IndoBERT and perplexity extraction can require substantial compute time and an internet connection when the required Hugging Face models are not already cached locally.

## Reproducibility

### Recommended environment

The trained artifacts are sensitive to library versions. The following versions are known to be important for compatibility:

```text
scikit-learn==1.9.0
catboost==1.2.10
```

Use the environment specification included with the release, when available, and record the versions of Python, NumPy, pandas, XGBoost, PyTorch, Transformers, and other dependencies used for reproduction.

### Reproduction checklist

1. Confirm that the expected dataset files are available under `data/`.
2. Verify the label mapping, pairing key, and minimum-length filtering.
3. Run the main notebook from the first cell to the final evaluation cell.
4. Keep `source_id` as the GroupKFold grouping key.
5. Fit all learned preprocessing components within each training fold.
6. Compare the generated fold metrics and out-of-fold predictions with the files under `results/`.
7. Export model artifacts only after confirming that the feature order and dimensions match the training pipeline.

> [!WARNING]
> Changes to feature order, feature definitions, dimensionality-reduction settings, preprocessing, or model hyperparameters require retraining. A serialized model is not valid if the inference pipeline differs from the pipeline used during training.

## Demonstration application

The research demonstration is located at:

```text
apps/detektor-ai-teks/
```

It consists of a FastAPI backend and a React/Vite frontend. Application-specific setup instructions are available in `apps/detektor-ai-teks/README.md`.

The backend expects the final model artifact at:

```text
apps/detektor-ai-teks/backend/artifacts/model_detektor_catboost_469.joblib
```

Transformer models are downloaded from Hugging Face during the first backend warm-up when they are not present in the local cache. Model caches are not stored in this repository.

## Limitations and responsible use

- The detector was developed for Indonesian news text with a minimum length of approximately 20 words.
- The internal training corpus contains AI-generated text from GPT-3.5-turbo; broader generator coverage is evaluated only through the external dataset.
- Performance outside the Indonesian news domain has not been established.
- Paraphrased, adversarially edited, or deliberately obfuscated text may behave differently from the evaluated samples.
- A prediction is not proof of authorship and should not be used as the sole basis for academic, legal, employment, or disciplinary decisions.
- Model outputs should be interpreted together with provenance information, human review, and the context in which the text was produced.

## Project status

This repository is maintained as a **research archive and reproducibility package**. It is not presented as a production-ready authorship-verification service.

Before creating a public GitHub release or Zenodo deposit, verify that the release includes:

- the final notebook and supporting scripts;
- an environment or dependency specification;
- only data that may legally be redistributed;
- trained models that are compatible with the documented pipeline;
- generated metrics and figures required to verify the reported results;
- a repository license;
- citation metadata, preferably in `CITATION.cff`; and
- a versioned release tag matching the Zenodo deposit.

## Citation

A permanent software citation should be added after the GitHub release has been archived on Zenodo. Replace this section with the final creator list, release title, version, year, and DOI from the Zenodo record.

Suggested citation format:

```text
Author(s). (2026). Detection of AI-Generated Indonesian News Using
Multi-Feature Fusion (Version X.Y.Z) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.REPLACE_WITH_RECORD_ID
```

Suggested BibTeX template:

```bibtex
@software{ai_generated_indonesian_news_detection_2026,
  author    = {REPLACE WITH CREATOR NAMES},
  title     = {Detection of AI-Generated Indonesian News Using Multi-Feature Fusion},
  year      = {2026},
  version   = {X.Y.Z},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.REPLACE_WITH_RECORD_ID},
  url       = {https://doi.org/10.5281/zenodo.REPLACE_WITH_RECORD_ID}
}
```

## License

The repository license has not yet been specified. Add a `LICENSE` file before publication and clearly distinguish between:

1. the license for source code;
2. the license for trained model artifacts; and
3. the licenses and redistribution conditions of the datasets.

Unless a license explicitly grants permission, the presence of a file in this repository should not be interpreted as permission to reuse, modify, or redistribute it.

## Acknowledgments

This repository uses data and pretrained language models developed by their respective authors and communities. Cite the M4 benchmark, the external Zenodo dataset, and each pretrained model according to their original documentation and license requirements.
