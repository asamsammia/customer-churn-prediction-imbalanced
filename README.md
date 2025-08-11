# Customer Churn Prediction (Imbalanced Learning)

Predict churn with logistic regression and class imbalance strategies (class weights).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Structure
- `src/model.py` – logistic regression pipeline
- `src/metrics.py` – ROC-AUC, PR-AUC helpers
- `notebooks/01_exploration.ipynb` – EDA and feature prep
- `tests/` – minimal unit tests
