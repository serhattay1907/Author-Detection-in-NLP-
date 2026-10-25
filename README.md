# ✍️ Authorship Attribution with NLP — "Who wrote this?"

Identifies which of **30 newspaper columnists** wrote a given article, using TF-IDF features and a linear classifier, then explores each author's language with n-gram language models.

## 📊 Results
| Setting | Value |
|---|---|
| Dataset | 30Columnists — 1,500 articles, 30 authors |
| Split | 1,200 train / 300 test |
| Features | Word TF-IDF (uni + bigrams, ~470K features) · Character TF-IDF (3–5 grams) |
| Model | Logistic Regression |
| **Accuracy** | **76.7%** (random baseline: 3.3%) |
| **Macro F1** | **0.757** |

**Insight:** the most predictive features are largely *topical* (team names for sports writers, medical vocabulary for health writers, candidate names for political columnists) — on this dataset topic is a strong proxy for authorship, which is discussed as a limitation in the report.

## 🔍 Pipeline
1. **Load** — articles grouped by author directory
2. **Preprocess** — tokenisation, lowercasing, stop-word handling (NLTK)
3. **Features** — word and character n-gram TF-IDF
4. **Classify** — Logistic Regression, evaluated with accuracy, macro F1 and per-author precision/recall
5. **Interpret** — top-weighted terms per author
6. **Language modelling** — per-author unigram/bigram/trigram frequencies and probabilities, n-gram sentence generation, compared with Hugging Face text-generation models (GPT-2 default, Phi-1.5)

## 📁 Files
| File | Content |
|---|---|
| `SERHAT_TAY_Author_Detection_using_NLP.ipynb` | Full notebook with outputs |
| `SERHAT_TAY_Author_Detection_using_NLP.py` | Script export |
| `Author_Detection_clean.ipynb` | Lighter notebook version |
| `Report_Author_Detection.pdf` | Project report |

## ▶️ Run
Designed for Google Colab: upload `30Columnists.zip` to `/content`, then run the notebook top to bottom.
```bash
pip install -r requirements.txt
```

## 🛠️ Stack
Python · NLTK · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Hugging Face Transformers

---
**Author:** Serhat Tay
