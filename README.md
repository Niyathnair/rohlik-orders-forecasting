# Rohlik Orders Forecasting

A Kaggle notebook for the Rohlik Orders Forecasting Challenge. It predicts the daily number of orders per warehouse from calendar and holiday features, using a voting ensemble of six tree-based regressors, and writes a submission file for the competition test set.

## Approach

1. Load `train.csv`, `test.csv`, `train_calendar.csv` and `test_calendar.csv` from the competition input folder. The two calendar files are only inspected; they are not used in the model.
2. Keep only the columns that exist in both train and test: `warehouse`, `date`, `holiday_name`, `holiday`, `shops_closed`, `winter_school_holidays`, `school_holidays`. Train-only columns (shutdown flags, weather, user activity) are dropped. Train and test are concatenated for joint encoding.
3. Split `date` into `date_year`, `date_month`, `date_day` and `date_day_of_week`, then drop the original column.
4. Fill missing `holiday_name` with `"None"` and one-hot encode it; label-encode `warehouse`.
5. Split the labelled rows 80/20 with `train_test_split(test_size=0.2, random_state=290)`.
6. Baseline: fit a `VotingRegressor` over `XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`, `HistGradientBoostingRegressor`, `GradientBoostingRegressor` and `RandomForestRegressor`, all with default parameters, and score it with MAPE on the 20% hold-out.
7. Tuning: run `RandomizedSearchCV` (15 iterations, 3-fold CV, scoring `neg_mean_absolute_percentage_error`) for XGBoost, LightGBM and CatBoost over `n_estimators`/`iterations`, `learning_rate`, `max_depth`/`num_leaves`/`depth` and `subsample`. The best three estimators replace the defaults in the voting ensemble.
8. Evaluate the tuned ensemble with 5-fold shuffled `KFold` cross-validation on all labelled rows, refit on all labelled rows, and predict the test set.

## Results

All numbers are MAPE as printed in the saved notebook outputs.

| Model | Evaluation | MAPE |
|---|---|---|
| Voting ensemble, default parameters | 20% hold-out split | 0.0437 |
| Voting ensemble, tuned XGB/LGBM/CatBoost | 5-fold CV on all labelled rows (mean) | 0.0398 |
| Voting ensemble, tuned, refit on all labelled rows | same 20% split | 0.0308 |

The 0.0308 figure is not a hold-out score: the ensemble had already been refit on the full labelled set, which includes that 20% split. The cross-validated 0.0398 is the out-of-sample estimate for the tuned ensemble.

## Repository contents

- `rohlik-submission.ipynb` - the full pipeline: loading, feature engineering, ensemble training, tuning, and submission export.

## Running it

The notebook was written on Kaggle (Python 3.10) and reads data from fixed Kaggle paths.

```
pip install numpy pandas seaborn matplotlib plotly scikit-learn xgboost lightgbm catboost
```

- On Kaggle: attach the competition data to the notebook and run all cells.
- Locally: download the competition files and either place them under `/kaggle/input/rohlik-orders-forecasting-challenge/` or edit the four `pd.read_csv` paths in the fourth code cell. Then open the notebook with `jupyter notebook rohlik-submission.ipynb`.
- The notebook calls `OneHotEncoder(sparse=False)`. That argument was removed in scikit-learn 1.4, so use an older scikit-learn or change it to `sparse_output=False`.

Outputs: `submission.csv` and `submission_Rohlik_again.csv`, each with columns `id` and `Target`. As the cells are ordered, `submission.csv` holds the predictions of the default-parameter ensemble and `submission_Rohlik_again.csv` holds those of the tuned ensemble.

## Data

Kaggle competition `rohlik-orders-forecasting-challenge` (the input folder name the notebook reads from): https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge

The training file has 7,340 rows and 18 columns. Train and test together cover seven warehouses (Prague_1, Prague_2, Prague_3, Brno_1, Budapest_1, Munich_1, Frankfurt_1). The data is not included in this repository.
