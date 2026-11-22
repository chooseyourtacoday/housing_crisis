# North DFW Housing Affordability Crisis
### A Predictive Data Analysis for Concerned Residents

**Author:** Alejandro Molina  
**GitHub:** [chooseyourtacoday](https://github.com/chooseyourtacoday)  
**Tools:** Python · Pandas · Jupyter · Claude AI · Tableau

**Tableau:** https://public.tableau.com/app/profile/alejandro.molina3014/vizzes

---

## Key findings

These numbers come from running the five notebooks on the files in this repo. The home-value series is Zillow's ZHVI for 13 North DFW zip codes, January 2015 through March 2026, downloaded on 2026-11-22. Zillow revises the index, so dollar levels differ from an earlier draft that used an April 2026 download. Year-over-year percent changes on the paths that draft printed are the same, including the 2022 affordability gap of **-23.53** percentage points.

**Pay did not keep up with prices.** Collin County average annual pay rose from **$61,631 (2015)** to **$88,091 (2024)**, **+42.9%**. Over those same years, average home values rose **86.1% to 105.7%** depending on the city.

| City | Avg home value, 2015 | Avg home value, 2024 | Change |
|------|---------------------:|---------------------:|-------:|
| Celina | $289,210 | $594,837 | +105.7% |
| Prosper | $408,244 | $793,507 | +94.4% |
| McKinney | $259,387 | $495,239 | +90.9% |
| Frisco | $365,392 | $685,021 | +87.5% |
| Allen | $312,389 | $583,161 | +86.7% |
| Plano | $298,121 | $554,679 | +86.1% |

**2022 was the worst gap.** Affordability gap is wage growth minus home-value growth, in percentage points. It was **-23.53 in 2022**. Prices outran wages that year by about 23.5 points. The gap was also negative in 2016, 2017, and 2021, and positive in 2018–2020 and 2023–2026.

**Prices are already off the peak.** City-average values peaked in **July 2022** (Plano peaked in **May 2024**). By **March 2026** they were lower:

| City | Versus peak, March 2026 |
|------|------------------------:|
| Celina | -14.4% |
| McKinney | -11.6% |
| Prosper | -10.8% |
| Allen | -8.1% |
| Frisco | -7.7% |
| Plano | -6.8% |

**Every city is above this project's affordability lines in March 2026.**

- Price-to-income ratio (home value / that city's median household income). Benchmark in this project: about 3x is healthy, above 5x is severely unaffordable. All six cities are above 3x. **Plano is the only one above 5x, at 5.23x.** The income figure is the latest ACS period in the file (2017–2021 for Plano, Allen, Frisco, and McKinney; 2019–2023 for Celina and Prosper), joined as one number per city.
- Mortgage burden (estimated payment / Collin County average monthly pay). The same county pay is used for every city, so this ranking follows home prices. Benchmark: above 30% is cost-burdened, above 50% is severely cost-burdened. All six cities are above 30%. **None is above 50%. Prosper is highest, at 49.3%.** Frisco is 43.1%.

### Honest model results

The model is a Random Forest **regression** of the same month's home value. Train on months before January 2024 (1,248 rows). Test on January 2024 through March 2026 (351 rows). Quote **R², MAE, and RMSE**. R² is the share of test-set variance explained. It is not classification accuracy.

| Model | R² | MAE | RMSE | MAE vs average test price ($584,513) |
|-------|---:|----:|-----:|--------------------------------------:|
| Earlier published run (leaked features, April 2026 Zillow vintage) | 0.9330 | $18,488 | $28,415 | 3.1% |
| Same leaked features, refit on this repo's data | 0.9422 | $18,220 | $25,350 | 3.1% |
| **Leak-free model (the one this repo saves)** | **0.6695** | **$47,210** | **$60,591** | **8.1%** |

On the leak-free test set the average prediction ($581,433) is close to the average actual price, but the typical miss is **$47,210**. Errors are uneven. MAE as a share of that city's average test price: Frisco 2.58%, Prosper 2.75%, Celina 3.22%, McKinney 6.52%, Plano 14.56%, Allen 16.57%.

**Why the old score was so high.** The published model put about **58%** of its importance on `affordability_index` and about **28%** on `mortgage_burden_pct`. Both are calculated from home value, which is the target. On this repo's file, `price_to_income_ratio * median_income` rebuilds `home_value` with a maximum absolute error of `1.164e-10` (floating-point noise). The refit of that leaked specification still leans on `affordability_index` (58.1%) and `mortgage_burden_pct` (27.9%) and still scores R² 0.9422. Dropping the six leaked columns, and changing nothing else about the split or the hyperparameters, drops R² from **0.9422 to 0.6695** and raises MAE from **$18,220 to $47,210**.

The leak-free model mostly uses the calendar year (**35.1%**), Collin County annual pay (**30.9%**), and median household income (**20.4%**). The mortgage rate is **2.0%**. Building permits are **0.7%**. That is association inside this model, not a ranking of causes.

**24-month scenario, not a correction call.** From April 2026 through March 2028 the saved scenario assumes wages grow 3% a year, population grows 0.5% a month, and the mortgage rate and permit count stay at the last observed values. No scenario input is computed from the predicted price. City-average changes under that scenario:

| City | March 2026 | March 2028 scenario | Change |
|------|-----------:|--------------------:|-------:|
| McKinney | $453,018 | $505,520 | +11.59% |
| Celina | $531,592 | $574,826 | +8.13% |
| Prosper | $739,696 | $761,568 | +2.96% |
| Allen | $544,690 | $558,376 | +2.51% |
| Plano | $521,356 | $532,689 | +2.17% |
| Frisco | $647,318 | $657,753 | +1.61% |

McKinney's dollar move ($52,502) is a bit larger than the test MAE. The other moves are smaller than a typical miss. The history already shows a pullback from the 2022 peak. This scenario does not add a reliable forecast of another drop or of a new surge.

---

## What target leakage means here

The question the model is graded on is: what is this month's home value? A feature leaks the target when it is computed from that same home value, so the model can see a rearranged copy of the answer.

| Feature | How it is built from home value | In the model? |
|---------|----------------------------------|---------------|
| `price_to_income_ratio` | `home_value / median_income` | No. Descriptive and Tableau only. |
| `mortgage_burden_pct` | Mortgage payment on `home_value`, divided by income | No. |
| `home_value_mom`, `home_value_yoy` | Percent change that includes this month's price | No. |
| `affordability_gap` | Wage growth minus `home_value_yoy` | No. |
| `affordability_index` | Blend of the ratio, the burden, and the gap | No. |
| City, year, month, mortgage rate, permits, wages, population, median income, wage growth, supply pressure | Not computed from this month's price | Yes. |

Notebook 03 still builds the affordability columns. They are the right tool for "who is stretched?" They are the wrong tool for predicting the price they were built from.

---

## Business questions

1. **Are we heading toward a housing market correction?** City averages are already **6.8% to 14.4%** below their peaks. The leak-free 24-month scenario does not project a further sharp drop. Its test MAE is **$47,210**, so small future moves are inside the noise.
2. **Who is most financially stretched?** On the March 2026 snapshot, **Plano** has the highest price-to-income ratio (**5.23x**). **Prosper** has the highest mortgage burden against Collin County average pay (**49.3%**).
3. **Is the wage gap a primary driver of the crisis?** From 2015 to 2024, pay rose **42.9%** and home values rose **86% to 106%**. The gap was worst in **2022 (-23.53 points)**.
4. **What does the next 2–3 years look like if nothing changes?** Under the scenario above, city averages change about **+1.6% to +11.6%** by March 2028. Treat that as a scenario with a wide error, not as a point forecast.
5. **What interventions could realistically help?** The ratios show prices ahead of pay, and every city above a 30% burden line. The model does not single out mortgage rates or permits. Those two inputs together are under 3% of leak-free importance.

---

## How to run

Install:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

Run the notebooks from the project folder, in this order. Each one reads the previous folder and writes the next.

| Order | Notebook | Reads | Writes |
|-------|----------|-------|--------|
| 1 | `01_Data_Ingestion.ipynb` | `data/raw/` | `staging/` (6 files) |
| 2 | `02_Data_Cleaning.ipynb` | `staging/` | `data/clean/master_dfw_housing.csv` (1,755 rows, 17 columns) |
| 3 | `03_Feature_Engineering.ipynb` | `data/clean/` | `data/features/dfw_housing_features.csv` (1,755 rows, 28 columns) |
| 4 | `04_Predictive_Model.ipynb` | `data/features/` | `data/predictions/` |
| 5 | `05_Model_Evaluation.ipynb` | `data/features/` and `data/predictions/` | `data/tableau/` (4 files) |

Those intermediate files are already in the repo, so a fresh clone can run a later notebook without rerunning the earlier ones. Rerunning from Notebook 01 regenerates them. If an input file is missing, the notebook stops and names the file and the notebook that creates it.

`data/raw/` contains:

- `zillow_north_dfw_zhvi.csv` — 13 zip codes, months 2015-01 through 2026-03, cut from [Zillow's national ZHVI](https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv) on 2026-11-22. The national file is about 120 MB and is not committed.
- `MORTGAGE30US.csv`, `indicator_data_download_20260427.csv`, `2024_txpopest_place.csv`, `Building Permits.csv`
- BLS QCEW annual files for Collin County (FIPS 48085) and Denton County (FIPS 48121), 2015–2024. The county-wide all-industry row in each file matches the wage series used throughout (Collin County **$61,631** in 2015 through **$88,091** in 2024).

---

## Project structure

```
housing_crisis/
├── data/
│   ├── raw/            ← source files, including the North DFW Zillow extract
│   ├── clean/          ← master dataframe from Notebook 02
│   ├── features/       ← engineered table from Notebook 03
│   ├── predictions/    ← leak-free scores, scenario, charts from Notebook 04
│   └── tableau/        ← four Tableau extracts from Notebook 05
├── staging/            ← six staged files from Notebook 01
├── 01_Data_Ingestion.ipynb
├── 02_Data_Cleaning.ipynb
├── 03_Feature_Engineering.ipynb
├── 04_Predictive_Model.ipynb
├── 05_Model_Evaluation.ipynb
└── README.md
```

---

## Notebooks

### Notebook 01 — Data Ingestion
Loads the six sources, keeps the 13 North DFW zip codes, reshapes Zillow from wide to long, and resamples weekly mortgage rates to month-end. Writes six files to `staging/`. The home-value file has **1,755** rows, **2015-01-31** through **2026-03-31**, and no null prices.

### Notebook 02 — Data Cleaning & Wrangling
Joins mortgage rates, permits, county wages, population, and city median income onto the monthly Zillow spine. Annual wages and population are carried across months with forward fill; population before 2020 is back filled from the 2020 count. Writes `data/clean/master_dfw_housing.csv`.

### Notebook 03 — Feature Engineering
Builds price-to-income, mortgage burden, home-value percent changes, wage growth, the affordability gap, population growth, supply pressure (permits per 1,000 residents), and a composite affordability index. These columns are for the written findings and for Tableau. Notebook 04 does not use the ones derived from home value as inputs.

### Notebook 04 — Predictive Model
Random Forest regressor (`n_estimators=200`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`). Time split at 2024-01-01. Fits the leaked specification once as a diagnostic, then fits and saves the leak-free model. The 24-month path is a scenario: wages **+3% per year** (an earlier draft multiplied by 1.03 inside the monthly loop, which is about 3% per month), population **+0.5% per month**, mortgage rate and permits held fixed.

### Notebook 05 — Model Evaluation & Tableau Export
Rebuilds the leak-free model for residuals, prints error by city, writes the findings from the computed tables, and exports:

- `data/tableau/tab_home_values.csv` (1,911 rows: history plus scenario)
- `data/tableau/tab_affordability.csv` (1,755 rows)
- `data/tableau/tab_forecast_summary.csv` (6 cities)
- `data/tableau/tab_feature_importance.csv` (10 leak-free inputs)

---

## Geography

| City | Zip Codes | County |
|------|-----------|--------|
| Frisco | 75033, 75034, 75035 | Collin / Denton |
| McKinney | 75069, 75070, 75071 | Collin |
| Allen | 75002, 75013 | Collin |
| Prosper | 75078 | Collin |
| Celina | 75009 | Collin |
| Plano | 75023, 75024, 75025 | Collin |

---

## Data sources

| Dataset | Source | File in `data/raw/` |
|---------|--------|---------------------|
| Home Values (ZHVI) | [Zillow Research](https://www.zillow.com/research/data/) | `zillow_north_dfw_zhvi.csv` |
| Mortgage Rates | [FRED](https://fred.stlouisfed.org/series/MORTGAGE30US) | `MORTGAGE30US.csv` |
| Median Household Income | [Texas DSHS](https://www.dshs.texas.gov/) | `indicator_data_download_20260427.csv` |
| Population Estimates | [Texas Demographic Center](https://demographics.texas.gov/) | `2024_txpopest_place.csv` |
| Building Permits | [Census Bureau](https://www.census.gov/construction/bps/) | `Building Permits.csv` |
| Wages by County | [BLS QCEW](https://www.bls.gov/cew/downloadable-data-files.htm) | `20XX.annual 48085/48121*.csv` |

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
jupyter
```

---

## Related Portfolio Projects

- [LoneStar Wireless Churn Analysis](https://github.com/chooseyourtacoday/LoneStar_Churn_Analysis)
- [Steam RPG Market Analysis](https://github.com/chooseyourtacoday)
- [Coffee Price Forecast](https://github.com/chooseyourtacoday)
- [Northwind SQL + Tableau](https://github.com/chooseyourtacoday)
