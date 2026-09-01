# NYC Taxi Fare Prediction

Predicting the fare amount of New York City taxi rides from trip metadata
(pickup/dropoff coordinates, timestamps, and passenger count), based on the
Kaggle [New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)
competition.

## Approach

The notebook (`new-york-taxi-fare-prediction.ipynb`) walks through:

1. **Data cleaning & datetime features** — parse `pickup_datetime`, adjust
   from UTC to local (EST) time, and extract `Year`, `Month`, `Day`, `Hours`,
   `Minutes`, plus a binary `MorningNight` flag.
2. **Distance feature** — compute the great-circle (haversine) distance
   between pickup and dropoff coordinates as `total_distance`, then drop the
   raw latitude/longitude columns.
3. **Modeling** — train and compare several regressors on the engineered
   features:
   - Linear Regression
   - Ridge Regression (L2, tuned via `GridSearchCV`)
   - Lasso Regression (L1, tuned via `GridSearchCV`)
   - XGBoost Regressor
4. **Evaluation** — feature importance (via `ExtraTreesRegressor`), residual
   plots, and R² / MAE / MSE / RMSE metrics on a held-out validation set.

## Results

| Model              | Notes                                    |
|--------------------|------------------------------------------|
| Linear Regression  | R² ≈ 0.02 — coordinates/time alone are weak linear predictors |
| Ridge Regression   | Best CV RMSE ≈ 9.59 (alpha = 100)        |
| Lasso Regression   | Best CV RMSE ≈ 9.59 (alpha = 0.01)       |
| **XGBoost**        | **MAE ≈ 2.37, RMSE ≈ 5.38** — best performer |

Adding the haversine `total_distance` feature is the single biggest driver
of model performance, and XGBoost substantially outperforms the linear
models on this data.

> **Note:** the notebook's XGBoost R² was previously computed as
> `sqrt(r2_score(...))`, a bug that inflated the true R² (≈0.69) to ≈0.83.
> This has been fixed in the notebook, and the XGBoost train/test split was
> also aligned with the one used for the linear models (`test_size=0.2,
> random_state=42`) so all four models are now compared on the same
> held-out data. Re-run the notebook to regenerate the corrected metrics.

## Project Structure

```
.
├── new-york-taxi-fare-prediction.ipynb  # main analysis & modeling notebook
├── NYCTaxiFares.csv                     # local sample dataset
├── requirements.txt                     # Python dependencies
└── LICENSE
```

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook new-york-taxi-fare-prediction.ipynb
```

> **Note:** the notebook was originally developed on Kaggle and reads
> `train.csv`/`test.csv` from `/kaggle/input/new-york-city-taxi-fare-prediction/`.
> The bundled `NYCTaxiFares.csv` has a similar schema (pickup/dropoff
> coordinates, `fare_amount`, `passenger_count`, `pickup_datetime`) but uses
> a `fare_class` column instead of `key`. To run fully offline, point the
> `read_csv` calls at `NYCTaxiFares.csv` and adjust the `key`/`fare_class`
> references accordingly.

## License

Released under the [MIT License](LICENSE).
