"""Create a SIMULATED telecom customer-churn dataset (Telco-style columns) with realistic data-quality problems.

Run:  python generate_dataset.py      ->  data/telco_churn_raw.csv
The data is simulated for learning and portfolio use. It is not real customer data.
You can replace it with the public IBM/Kaggle "Telco Customer Churn" CSV (same column names) and re-run the notebook.
"""
import numpy as np, pandas as pd, string
from pathlib import Path

rng = np.random.default_rng(2026)
N = 7043
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

def pick(options, probs, size=N):
    return rng.choice(options, size=size, p=probs)

contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
internet = pick(["Fiber optic", "DSL", "No"], [0.44, 0.34, 0.22])

# tenure depends on contract type
tenure = np.where(contract == "Month-to-month", rng.gamma(1.6, 12, N),
         np.where(contract == "One year", rng.normal(35, 16, N), rng.normal(52, 15, N)))
tenure = np.clip(np.round(tenure), 0, 72).astype(int)
tenure[rng.choice(N, 11, replace=False)] = 0           # brand-new customers (TotalCharges will be blank)

gender = pick(["Male", "Female"], [0.505, 0.495])
senior = pick([0, 1], [0.84, 0.16])
partner = pick(["Yes", "No"], [0.48, 0.52])
dependents = np.where(partner == "Yes", pick(["Yes", "No"], [0.45, 0.55]), pick(["Yes", "No"], [0.1, 0.9]))
phone = pick(["Yes", "No"], [0.90, 0.10])
lines = np.where(phone == "No", "No phone service", pick(["Yes", "No"], [0.47, 0.53]))

def addon(p_yes):
    return np.where(internet == "No", "No internet service", pick(["Yes", "No"], [p_yes, 1 - p_yes]))
security, backup, protection = addon(0.29), addon(0.34), addon(0.34)
tech = addon(0.29)
tv, movies = addon(0.38), addon(0.39)

paperless = pick(["Yes", "No"], [0.59, 0.41])
payment = pick(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
               [0.34, 0.23, 0.22, 0.21])

# monthly charges built from the services a customer holds
monthly = (np.where(phone == "Yes", 20, 0) + np.where(lines == "Yes", 10, 0)
           + np.where(internet == "Fiber optic", 45, np.where(internet == "DSL", 25, 0)))
for col, price in [(security, 5), (backup, 5), (protection, 5), (tech, 5), (tv, 9), (movies, 9)]:
    monthly = monthly + np.where(col == "Yes", price, 0)
monthly = np.round(monthly + rng.normal(0, 3, N) + 18.5, 2).clip(18.25, 119)

# churn probability from a logistic model
logit = (-0.55 + 1.15 * (contract == "Month-to-month") - 1.45 * (contract == "Two year")
         + 0.65 * (internet == "Fiber optic") - 0.55 * (internet == "No")
         - 0.032 * tenure + 0.012 * (monthly - 65)
         + 0.55 * (payment == "Electronic check") - 0.55 * (tech == "Yes") - 0.45 * (security == "Yes")
         + 0.30 * senior + 0.20 * (paperless == "Yes") - 0.20 * (partner == "Yes"))
prob = 1 / (1 + np.exp(-(logit - 0.70)))
churn = np.where(rng.random(N) < prob, "Yes", "No")

total = np.round(tenure * monthly * rng.uniform(0.96, 1.04, N), 2)

ids = ["%04d-%s" % (rng.integers(0, 10000), "".join(rng.choice(list(string.ascii_uppercase), 5))) for _ in range(N)]
df = pd.DataFrame(dict(customerID=ids, gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents,
                       tenure=tenure, PhoneService=phone, MultipleLines=lines, InternetService=internet,
                       OnlineSecurity=security, OnlineBackup=backup, DeviceProtection=protection, TechSupport=tech,
                       StreamingTV=tv, StreamingMovies=movies, Contract=contract, PaperlessBilling=paperless,
                       PaymentMethod=payment, MonthlyCharges=monthly, TotalCharges=total, Churn=churn))

# ---------- inject realistic data-quality problems (to be cleaned in the notebook) ----------
df["TotalCharges"] = df["TotalCharges"].astype(object)
df.loc[df["tenure"] == 0, "TotalCharges"] = " "                       # blank strings, as in the public Telco file
idx = rng.choice(df.index[df["tenure"] > 0], 35, replace=False)
df.loc[idx, "MonthlyCharges"] = np.nan                                 # missing monthly charges
idx = rng.choice(df.index, 40, replace=False)
df.loc[idx, "PaymentMethod"] = df.loc[idx, "PaymentMethod"].str.lower()          # inconsistent casing
idx = rng.choice(df.index, 30, replace=False)
df.loc[idx, "gender"] = df.loc[idx, "gender"].map({"Male": "male", "Female": "FEMALE"})
dups = df.sample(25, random_state=1)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

df.to_csv(OUT / "telco_churn_raw.csv", index=False)
print(df.shape, "churn rate (raw):", round((df.Churn == "Yes").mean() * 100, 1), "%")
