# Data sources and provenance

**Status of this build.** The build sandbox had *no access to Kaggle, UCI, Open Food Facts, MOSPI, World Bank or any other public data host* (only package registries and GitHub were reachable). Therefore:

* **No real-world dataset was downloaded or analysed in this build.**
* All results shipped with the project were produced on a **clearly labelled SIMULATED demo dataset**.
* A real-data path (download script + adapter) is provided for Online Retail II. The adapter was tested on a *format fixture* only; the live download was **not** tested. Licences and URLs below were **not verified from the sandbox** - check them at the source before relying on them.

Nothing in the repository presents simulated values as observations. `data/processed/dataset_meta.json` records `mode: demo_synthetic | real`, and the UI shows a DEMO banner whenever the mode is not `real`.

---

## 1. Synthetic FMCG demo dataset (ACTIVE in demo mode)

| Field | Value |
|---|---|
| Name | Synthetic FMCG demo dataset |
| Source / publisher | Generated locally by `src/demo_data.py` (seeded; `scripts/make_demo_data.py`) |
| URL | none |
| Access date | generation timestamp in `dataset_meta.json` |
| Licence | Part of this project; no third-party restrictions |
| Observation period | 2023-01-02 to 2024-12-23, weekly (104 weeks) |
| Geography | 4 fictional regions x 3 store types = 12 stores (INR-denominated) |
| Records (default) | ~49.5k product-store-week rows (72 products, 12 of them "established" with unobserved launch, 60 with observed launch); ~95.6k transaction lines for 2,500 customers |
| Important columns | date, product_id, store_id, units, price, promo_flag, discount_pct, holiday_flag, weeks_since_intro, category, subcategory, brand (fictional), pack_size, pack_unit, positioning, region, store_type |
| Nature | **SIMULATED** from a documented data-generating process (see below) |
| Known limitations | Real-world heterogeneity, competitor reactions, stock-outs, cannibalisation and real campaign effects are absent. Holiday weeks are an *illustrative* calendar (ISO weeks 10, 11, 42-44, 52). Brand names are fictional. Basket lifts are structureless by construction. |

**Data-generating process** (so that estimators can be checked against known truth): per product-store-week mean = category base x store-type x region x category-region noise x product effect (SD 0.35) x tier volume effect x pack-size effect x trend x seasonality x holiday x `price_ratio^elasticity` (planted category elasticities -1.2 to -2.1) x promo uplift x launch ramp-up `1-exp(-(age+1)/tau)`; observed units are gamma-Poisson draws.

Contribution to models: A1 lag forecaster (training/validation/test), A2 cold-start launch model, C price response (the test-suite verifies planted elasticities are recovered), D segmentation (transactions), E promotion association, F opportunity score, market positioning (historical reference prices).

## 2. Online Retail II (real; supported by `scripts/download_data.py`, NOT run in this build)

| Field | Value |
|---|---|
| Publisher | UCI Machine Learning Repository (dataset 502; data originally from a UK online retailer) |
| URL | https://archive.ics.uci.edu/dataset/502/online+retail+ii (script uses the `static/public/502/online+retail+ii.zip` link - **unverified**) |
| Licence | Listed by UCI as CC BY 4.0 at the time of writing - **verify** |
| Observation period | Dec 2009 - Dec 2011 (per UCI description) |
| Geography | Mostly United Kingdom; prices in GBP |
| Records | ~1.07M invoice lines (per UCI description) |
| Columns | Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country |
| Limitations | **Not an FMCG manufacturer dataset** (mostly giftware wholesaler). No category field (categories are keyword-derived from descriptions), no promotions, no regions beyond country, no pack size, no launch outcomes; launch dates are proxied by first-sale week (left-censored). Cancellations/non-product codes are removed by the adapter. |
| Contribution | If imported: lag forecaster, cold-start model (only products whose first sale is >8 weeks after dataset start), elasticity (no promo variation -> weaker identification), RFM segmentation and basket analysis |

Because of these limitations, the cold-start model on this data answers "how do new *giftware* products ramp up" - not FMCG launch success. Use it to validate the pipeline on real transactions, then replace it with an FMCG dataset that has the fields you need.

## 3. Candidate sources evaluated but NOT integrated

Not accessed (sandbox network restrictions); nothing below was verified. Suitability notes are from general knowledge of these datasets and must be confirmed.

| Source | Potential use | Main caveat |
|---|---|---|
| Corporacion Favorita Grocery Sales Forecasting (Kaggle) | Store-item-day grocery demand, promotions, oil price, holidays - good fit for the lag forecaster | Ecuador; no launch labels; Kaggle account/terms required |
| M5 Forecasting (Walmart, Kaggle) | Item-store-day sales, prices, events | US retail; competition terms apply |
| Rossmann Store Sales (Kaggle) | Store-level promotions/holidays | Store totals, no product dimension; drugstore retail |
| Open Food Facts | Real packaged-product attributes (pack size, category, nutrition) for competitor attribute comparison | Crowd-sourced; no prices/sales; ODbL attribution/share-alike terms |
| MOSPI / Department of Consumer Affairs | CPI and retail price observations for India | Manual download; upload via Pricing page > Macro indicators |
| World Bank indicators | Inflation, urbanisation, household expenditure | Country-level only |
| Company annual reports | Category context, brand portfolios | Unstructured; manual use |

Macro indicators are never fetched automatically. If you download a series, upload it on the Pricing page with `source_name`, `source_url` and `retrieved_at`; stale series are flagged.

## 4. Current market prices

No live prices are retrieved and none are fabricated. The competitor view combines (a) *dataset reference prices* - labelled HISTORICAL with the dataset end date - and (b) *user-provided* competitor prices (CSV or manual entry) that must carry source, URL and retrieval timestamp and are flagged fresh / ageing / stale. Prices are normalised to price per 100 g/ml.

## 5. Launch-outcome labels

**None exist** in any source used here. Consequently no launch-success classifier is trained; the platform reports a *commercial-hurdle attainment probability* (see README). To train a real classifier you need >=200 launches with verified outcomes (>=30 per class) and features known at launch time; `src/success_model.py` documents the requirement and refuses to fabricate labels.
