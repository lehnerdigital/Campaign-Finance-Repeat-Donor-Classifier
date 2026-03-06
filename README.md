# 🔄 Repeat Donor Classifier
### NYC Campaign Finance Board — 2013–2021 Elections

> **Who will donate again?** A Gradient Boosting classifier that predicts repeat donor probability from first-touch data alone — scoreable within 24 hours of a donor's first contribution.

---

## Overview

After consulting in campaign finance, I kept coming back to the same question: *what if campaigns could predict donor loyalty before picking up the phone?*

This model is the answer. Trained on **760,955 transactions** across three NYC election cycles, it identifies which first-time donors are likely to give again — and which aren't — using only information available at the moment of their first gift.

| | Value | 
|---|---|
| **Dataset** | NYC Campaign Finance Board, 2013–2021 |
| **Donors** | 423,576 unique donors |
| **Target** | Binary — will donor give again? (1 = repeat, 0 = single-gift) |
| **Base rate** | 27.4% of donors give more than once |
| **Best model** | Gradient Boosting Classifier |
| **AUC** | 0.796 |
| **Accuracy** | 85% |
| **Precision (repeat donors)** | 100% |

---

## The Two Models in This Project

This classifier is the second of two complementary models built on the same dataset:

| | Regression Model | **This Classifier** |
|---|---|---|
| **Question** | How much will they give? | Will they give again? |
| **Output** | Predicted lifetime dollar value | Repeat probability (0–1) |
| **Top feature** | Median past gift amount | Size of first donation |
| **Use case** | Set ask amounts | Prioritize outreach |

**Used together:** classifier decides *who to call*, regression decides *how much to ask for*.

---

## What Predicts Repeat Giving

| Rank | Feature | Key Finding |
|---|---|---|
| 1 | **First donation amount** | $500–$1K first gifts return at **33.6%** vs 24% for under $25 |
| 2 | **Election year of first gift** | 2013 cohort returns at **32.6%** vs 24.1% for 2021 |
| 3 | **Offices donated to** | Multi-office donors signal broader civic engagement |
| 4 | **Borough** | Manhattan **33.4%** return rate vs Bronx at 22.5% |

---

## Donor Archetypes

**🔥 HIGH Repeat Probability**
- Manhattan-based Executive Director or Chief of Staff
- First gift of $500+ in the 2013 election cycle
- Donated to multiple offices or candidates
- → **Priority outreach within 24 hours of first gift**

**❄️ LOW Repeat Probability**
- Bronx-based retiree or unemployed individual
- First gift under $25 in the 2021 election cycle
- Gave to a single candidate only
- → **Light-touch automated nurture sequence**

---

## Project Structure

```
├── Repeat_Donor_Classifier.ipynb     # Full analysis notebook
├── repeat_donor_classifier.pkl       # Trained Gradient Boosting model
├── repeat_donor_features.pkl         # Feature list for inference
├── contributions.csv                 # Source data (not included — see Data section)
├── charts/
│   ├── classifier_roc_cm.png         # ROC curves + confusion matrix
│   ├── classifier_importance.png     # Feature importance
│   ├── classifier_repeat_rates.png   # Repeat rates by amount & year
│   ├── classifier_occ_borough.png    # Repeat rates by occupation & borough
│   └── classifier_prob_dist.png      # Probability distribution + precision-recall
└── Repeat_Donor_Classifier.pptx      # 11-slide presentation deck
```

---

## Notebook Walkthrough

The notebook (`Repeat_Donor_Classifier.ipynb`) covers 14 sections end to end:

1. **Setup & Imports** — libraries, color palette, configuration
2. **Load & Explore Data** — raw transaction overview
3. **Feature Engineering** — donor-level aggregation using first-touch data only
4. **EDA** — repeat rates by donation amount, election year, occupation, borough
5. **Train / Test Split** — stratified 80/20 split
6. **Model Training** — Logistic Regression, Random Forest, Gradient Boosting
7. **Evaluation** — ROC curves + confusion matrix
8. **Feature Importance** — built-in + permutation importance
9. **Probability Distribution** — predicted score distributions + precision-recall curve
10. **Score New Donors** — ready-to-use `score_donor()` inference function
11. **Threshold Analysis** — precision/recall tradeoff across decision cutoffs
12. **Cross-Validation** — 5-fold stratified CV
13. **Save Model** — exports `.pkl` files
14. **Key Findings Summary** — results table and strategic takeaways

---

## Quickstart

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Run the Notebook

```bash
jupyter notebook Repeat_Donor_Classifier.ipynb
```

> ⚠️ `contributions.csv` must be in the same directory. See **Data** section below.

### Score a New Donor (Inference)

```python
import joblib
import numpy as np

model    = joblib.load('repeat_donor_classifier.pkl')
features = joblib.load('repeat_donor_features.pkl')

def score_donor(first_amt, first_year, n_offices,
                borough_enc=0, c_code_enc=0, pay_enc=0, occ_enc=1):
    X = np.array([[np.log1p(first_amt), first_year, n_offices,
                   borough_enc, c_code_enc, pay_enc, occ_enc]])
    prob = model.predict_proba(X)[0][1]
    tier = 'HIGH' if prob >= 0.45 else ('MEDIUM' if prob >= 0.30 else 'LOW')
    return {'repeat_probability': round(float(prob), 4), 'tier': tier}

# High-probability donor: Manhattan exec, $750 first gift, 2013
score_donor(first_amt=750, first_year=2013, n_offices=3)
# → {'repeat_probability': 0.58, 'tier': 'HIGH'}

# Low-probability donor: Bronx retiree, $10 first gift, 2021
score_donor(first_amt=10, first_year=2021, n_offices=1)
# → {'repeat_probability': 0.18, 'tier': 'LOW'}
```

---

## Model Performance

### All Three Models Compared

| Model | AUC | Accuracy |
|---|---|---|
| Logistic Regression | 0.768 | 85% |
| Random Forest | 0.793 | 85% |
| **Gradient Boosting** | **0.796** | **85%** |

### What AUC = 0.796 Means in Plain English

If you randomly pick one repeat donor and one single-gift donor from the test set, the model ranks the repeat donor higher **79.6% of the time** — vs 50% for a random guess.

### Precision vs Recall Tradeoff

The model at the default 0.5 threshold is **conservative** — when it predicts a repeat donor, it is right nearly 100% of the time, but it only captures ~45% of all actual repeat donors. The threshold can be tuned depending on campaign resources:

- **Lower threshold (0.30)** → more donors flagged, higher recall, lower precision
- **Higher threshold (0.55)** → fewer but more confident repeat predictions

---

## Data

**Source:** [NYC Campaign Finance Board](https://www.nyc.gov/site/campaign-finance/index.page)  
**Years:** 2013, 2017, 2021 election cycles  
**Records:** 760,955 transactions  
**File:** `contributions.csv` — not included in this repo due to file size. Download directly from the NYC CFB open data portal.

### Key Columns Used

| Column | Description |
|---|---|
| `NAME` | Donor name (used with ZIP to create donor ID) |
| `ZIP` | Donor ZIP code |
| `AMNT` | Contribution amount |
| `DATE` | Contribution date |
| `ELECTION` | Election year |
| `OFFICECD` | Office the candidate is running for |
| `RECIPID` | Recipient (candidate) ID |
| `BOROUGHCD` | NYC borough code |
| `OCCUPATION` | Donor occupation |
| `C_CODE` | Contributor type (Individual, LLC, Committee, etc.) |
| `PAY_METHOD` | Payment method |

---

## Strategic Applications

1. **Real-time donor scoring** — Score every new donor within 24 hours of their first gift. Route high-probability donors to senior fundraisers immediately.

2. **Tiered outreach** — Use the probability score to build three contact tiers (HIGH / MEDIUM / LOW) with different outreach intensity and budget.

3. **2013 cohort re-activation** — Donors whose first gift was in 2013 return at 32.6% — the highest of any cohort. Prioritize this segment for re-engagement.

4. **Manhattan professional targeting** — Combined with occupation data, Executive Directors and Chiefs of Staff in Manhattan represent the single highest-ROI outreach segment.

5. **Combined scoring engine** — Pair with the [Donor Value Regression model](../campaingn-finance-donor-value-prediction-ml) to score both *who to contact* and *how much to ask for* simultaneously.

---

## Author

**Bryan Lehner**  
Founder, [Lehner Digital](https://lehnerdigital.com)  
Machine Learning · AI Automation · Workflow Engineering  

Built out of curiosity after years of consulting in campaign finance — wondering what machine learning could tell us about donor behavior that intuition couldn't.

---

## License

MIT License — free to use, modify, and distribute with attribution.
