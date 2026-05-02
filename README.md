# hw7 -- streamflow forecasting
verde river near camp verde, az (usgs 09506000)

---

## setup

```bash
conda env create -f environment.yml
conda activate hw7_streamflow
```

## how to run

change the variables at the top of `run_forecast.py` then:

```bash
python run_forecast.py
```

or `bash run.sh`

figures go to `outputs/figures/`, saved model params go to `outputs/models/`

## models

**persistence** -- just predicts Q tomorrow = Q today. no fitting.

**linear_ar** -- ols regression with only Q at lag-1 as predictor

**seasonal_regression** -- ols with lags at 1, 7, 14 days + sin/cos of day of year to capture the seasonal cycle. fitted in log(Q+1) space since streamflow is skewed.

## options at top of script

- `MODEL` -- which model to use
- `REFIT` -- True to retrain, False loads saved params from json
- `VALIDATE` -- True to print metrics and show validation plots
- `LOG_TRANSFORM` -- True to fit in log space
- `FORECAST_START` -- first day of the 5-day forecast
- `FIT_START/FIT_END` -- training window
- `TEST_START/TEST_END` -- held out test window
