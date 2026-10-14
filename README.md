# Telecom Customer Churn Analysis (Python)

Exploratory data analysis of a telecom customer base using **Pandas, NumPy and Matplotlib** in a Jupyter Notebook.
The goal is to find out who is leaving, why, and how much monthly revenue is at risk, then recommend how to reduce churn.

> The dataset is **simulated** (created with `generate_dataset.py`) with the same columns as the public
> *Telco Customer Churn* data. To use the real file, replace `data/telco_churn_raw.csv` and re-run the notebook.

## Files

| File | Purpose |
|---|---|
| `churn_analysis.ipynb` | The full analysis with outputs (open this first) |
| `generate_dataset.py` | Creates the raw dataset (seeded and reproducible) |
| `data/telco_churn_raw.csv` | Raw data with deliberate quality problems |
| `data/telco_churn_clean.csv` | Cleaned data produced by the notebook |
| `data/churn_summary_by_group.csv` | Churn-rate summary tables |
| `images/` | Charts saved by the notebook |
| `requirements.txt` | Python packages |

## How to run

```bash
pip install -r requirements.txt
python generate_dataset.py        # optional - the raw CSV is already included
jupyter notebook churn_analysis.ipynb
```

## What the notebook does

1. **Audit and clean** - 25 duplicate rows, 35 missing `MonthlyCharges`, blank `TotalCharges`, inconsistent text (`male`, `FEMALE`, `electronic check`).
2. **Feature engineering** - tenure groups (`pd.cut`), spend bands (`pd.qcut`), number of services, high-risk flag (`np.where`).
3. **Analysis** - `groupby` and `pivot_table` churn rates, percentiles, correlation matrix, high-risk segments.
4. **Visualisation** - 4 Matplotlib charts saved in `images/`.
5. **Insights** - findings printed from the data, plus recommendations.

## Key findings (7,043 customers after cleaning)

| Finding | Result |
|---|---|
| Overall churn rate | 26.2% (1,847 customers); churned customers held 28.8% of monthly revenue |
| Contract type | Month-to-month churn 41.7% vs 2.6% on two-year contracts (16x higher) |
| Internet service | Fiber optic 35.5% vs DSL 23.1% |
| Payment method | Electronic check 31.9% vs 23.3% for all other methods |
| Tenure | First-year customers churn 47.0%; customers with 49+ months churn 5.2% |
| Highest-risk segment | Month-to-month, fiber optic, first year: 63.5% churn |
| Concentration | Month-to-month first-year customers are 22% of the base but 42% of all churn |

## Recommendations

1. Move month-to-month customers to longer contracts with a small discount or free add-on.
2. Investigate fiber-optic service quality and billing complaints.
3. Bundle tech support and online security for new customers.
4. Encourage electronic-check customers to switch to automatic payment.
5. Build an early-warning call list from the high-risk profile before month 6.

Author: Aswinkumar K C
