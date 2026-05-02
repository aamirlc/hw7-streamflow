import os
import sys
import json
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# verde river near camp verde
STATION_ID     = '09506000'
FIT_START      = '2000-01-01'
FIT_END        = '2018-12-31'
TEST_START     = '2019-01-01'
TEST_END       = '2023-12-31'
FORECAST_START = '2025-04-28'
N_DAYS         = 5

REFIT         = True
VALIDATE      = True
LOG_TRANSFORM = True  # fit on log(Q+1), streamflow is right skewed

# model options: 'persistence', 'linear_ar', 'seasonal_regression'
MODEL = 'seasonal_regression'

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)


def fetch_streamflow(station, start_date, end_date):
    url = 'https://waterservices.usgs.gov/nwis/dv/'
    params = {
        'format': 'json',
        'sites': station,
        'parameterCd': '00060',
        'startDT': start_date,
        'endDT': end_date,
        'siteStatus': 'all',
    }
    print(f'fetching {station} {start_date} to {end_date}...')
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    records = data['value']['timeSeries'][0]['values'][0]['value']
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['dateTime']).dt.normalize()
    df['Q'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[['date', 'Q']].set_index('date').sort_index()
    nmiss = df['Q'].isna().sum()
    if nmiss > 0:
        print(f'  {nmiss} missing values, forward filling')
        df['Q'] = df['Q'].ffill()
    print(f'  {len(df)} records')
    return df


def build_features(df, model_type, log_transform=False):
    df = df.copy()
    if log_transform:
        df['Q'] = np.log1p(df['Q'])
    df['Q_lag1']  = df['Q'].shift(1)
    df['Q_lag7']  = df['Q'].shift(7)
    df['Q_lag14'] = df['Q'].shift(14)
    doy = df.index.dayofyear.astype(float)
    df['sin1'] = np.sin(2 * np.pi * doy / 365.25)
    df['cos1'] = np.cos(2 * np.pi * doy / 365.25)
    df['sin2'] = np.sin(4 * np.pi * doy / 365.25)
    df['cos2'] = np.cos(4 * np.pi * doy / 365.25)
    df = df.dropna()
    if model_type in ('persistence', 'linear_ar'):
        feat_cols = ['Q_lag1']
    elif model_type == 'seasonal_regression':
        feat_cols = ['Q_lag1', 'Q_lag7', 'Q_lag14', 'sin1', 'cos1', 'sin2', 'cos2']
    else:
        sys.exit(f'unknown model: {model_type}')
    return df[feat_cols].values, df['Q'].values, df, feat_cols


def fit_model(X, y, model_type):
    if model_type == 'persistence':
        return None
    m = LinearRegression()
    m.fit(X, y)
    return m


def save_model(model, model_type, feat_cols):
    if model_type == 'persistence':
        return
    params = {
        'coef': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'feat_cols': feat_cols
    }
    with open(f'outputs/models/{model_type}.json', 'w') as f:
        json.dump(params, f)
    print(f'model saved to outputs/models/{model_type}.json')


def load_model(model_type):
    path = f'outputs/models/{model_type}.json'
    if not os.path.exists(path):
        sys.exit(f'no saved model at {path}, set REFIT=True first')
    with open(path) as f:
        params = json.load(f)
    m = LinearRegression()
    m.coef_ = np.array(params['coef'])
    m.intercept_ = params['intercept']
    print(f'model loaded from {path}')
    return m


def predict_model(model, X, model_type):
    if model_type == 'persistence':
        return X[:, 0].copy()
    return model.predict(X)


def compute_metrics(obs, pred):
    rmse  = np.sqrt(mean_squared_error(obs, pred))
    mae   = mean_absolute_error(obs, pred)
    nse   = 1 - np.sum((obs - pred)**2) / np.sum((obs - obs.mean())**2)
    pbias = 100 * np.sum(pred - obs) / np.sum(obs)
    return {'RMSE': rmse, 'MAE': mae, 'NSE': nse, 'PBIAS%': pbias}


def make_forecast(df_all, model, model_type, feat_cols, start, n_days, log_transform=False):
    Q = np.log1p(df_all['Q']) if log_transform else df_all['Q'].copy()
    dates = [pd.Timestamp(start) + timedelta(days=i) for i in range(n_days)]
    preds = []
    for i, d in enumerate(dates):
        lag1  = preds[i-1] if i > 0 else Q.get(d - timedelta(days=1), np.nan)
        lag7  = Q.get(d - timedelta(days=7),  np.nan)
        lag14 = Q.get(d - timedelta(days=14), np.nan)
        doy   = float(d.dayofyear)
        fmap  = {
            'Q_lag1': lag1, 'Q_lag7': lag7, 'Q_lag14': lag14,
            'sin1': np.sin(2*np.pi*doy/365.25), 'cos1': np.cos(2*np.pi*doy/365.25),
            'sin2': np.sin(4*np.pi*doy/365.25), 'cos2': np.cos(4*np.pi*doy/365.25),
        }
        x = np.array([[fmap[c] for c in feat_cols]])
        p = lag1 if model_type == 'persistence' else model.predict(x)[0]
        preds.append(p)
    preds = np.expm1(np.array(preds)) if log_transform else np.array(preds)
    return pd.Series(np.maximum(preds, 0), index=dates, name='Q_cfs')


def make_workflow_diagram():
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#d4e6f1'):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle='round,pad=0.15',
                           facecolor=fc, edgecolor='#2c3e50', linewidth=1.4)
        ax.add_patch(p)
        ax.text(x, y, text, ha='center', va='center', fontsize=9.5,
                multialignment='center')

    def arrow(x, y_from, y_to):
        ax.annotate('', xy=(x, y_to + 0.32), xytext=(x, y_from - 0.32),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.4))

    steps = [
        (5, 12.2, 'USGS NWIS API', '#aed6f1'),
        (5, 10.9, 'Fetch Daily Streamflow\n(station 09506000)', '#d4e6f1'),
        (5,  9.6, 'Split: Train (2000-2018)\nTest (2019-2023)', '#d4e6f1'),
        (5,  8.3, 'Feature Engineering\nlag 1 / 7 / 14 days  +  sin/cos DOY', '#d4e6f1'),
        (5,  7.0, 'Fit Model\npersistence  |  linear AR  |  seasonal regression', '#d5f5e3'),
        (5,  5.7, 'Validate on Test Period\nRMSE   MAE   NSE   PBIAS', '#d4e6f1'),
        (5,  4.4, '5-Day Recursive Forecast', '#d4e6f1'),
        (5,  3.1, 'Output Figures\ntime series  |  scatter  |  residuals  |  forecast', '#fdebd0'),
    ]

    for x, y, text, fc in steps:
        box(x, y, 7.2, 0.85, text, fc)

    ys = [s[1] for s in steps]
    for i in range(len(ys) - 1):
        arrow(5, ys[i], ys[i+1])

    ax.set_title('Workflow -- Verde River Streamflow Forecasting', fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig('outputs/figures/workflow_diagram.png', dpi=150, bbox_inches='tight')
    print('workflow diagram saved')
    plt.show()


# fetch data
fetch_start = str(pd.Timestamp(FIT_START) - timedelta(days=30))[:10]
df_all  = fetch_streamflow(STATION_ID, fetch_start, FORECAST_START)
df_fit  = df_all.loc[FIT_START:FIT_END]
df_test = df_all.loc[TEST_START:TEST_END]
print(f'train: {len(df_fit)}  test: {len(df_test)}')

# workflow diagram
make_workflow_diagram()

# full record plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_all.index, df_all['Q'], color='steelblue', linewidth=0.6)
ax.set_title('Verde River near Camp Verde (09506000)')
ax.set_xlabel('Date')
ax.set_ylabel('Discharge (ft^3/s)')
plt.tight_layout()
plt.savefig('outputs/figures/full_record.png', dpi=150, bbox_inches='tight')
plt.show()

# seasonal cycle
df_all['month'] = df_all.index.month
monthly = df_all.groupby('month')['Q'].agg(['mean', 'std'])
df_all.drop(columns=['month'], inplace=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(monthly.index, monthly['mean'], color='steelblue', alpha=0.7)
ax.errorbar(monthly.index, monthly['mean'], yerr=monthly['std'],
            fmt='none', color='black', linewidth=1, capsize=4)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_xlabel('Month')
ax.set_ylabel('Discharge (ft^3/s)')
ax.set_title('Seasonal Cycle -- Verde River')
plt.tight_layout()
plt.savefig('outputs/figures/seasonal_cycle.png', dpi=150, bbox_inches='tight')
plt.show()

# log transform check
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(df_all['Q'], bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
axes[0].set_xlabel('Q (ft^3/s)')
axes[0].set_title('raw Q')
axes[1].hist(np.log1p(df_all['Q']), bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
axes[1].set_xlabel('log(Q+1)')
axes[1].set_title('log(Q+1)')
plt.tight_layout()
plt.savefig('outputs/figures/log_transform.png', dpi=150, bbox_inches='tight')
plt.show()

# build features
X_fit,  y_fit,  df_feat_fit,  feat_cols = build_features(df_fit,  MODEL, LOG_TRANSFORM)
X_test, y_test, df_feat_test, _         = build_features(df_test, MODEL, LOG_TRANSFORM)
print(f'features: {feat_cols}')
print(f'train samples: {len(X_fit)}  test samples: {len(X_test)}')

# fit or load
if REFIT:
    print(f'fitting {MODEL}...')
    model = fit_model(X_fit, y_fit, MODEL)
    if model is not None:
        print('coefs:', dict(zip(feat_cols, model.coef_)))
        print('intercept:', round(model.intercept_, 4))
    save_model(model, MODEL, feat_cols)
else:
    model = load_model(MODEL)

# validation
if VALIDATE:
    y_train_pred = predict_model(model, X_fit,  MODEL)
    y_test_pred  = predict_model(model, X_test, MODEL)

    if LOG_TRANSFORM:
        tr_obs, tr_pred = np.expm1(y_fit),  np.expm1(y_train_pred)
        te_obs, te_pred = np.expm1(y_test), np.expm1(y_test_pred)
    else:
        tr_obs, tr_pred = y_fit, y_train_pred
        te_obs, te_pred = y_test, y_test_pred

    print('\ntrain metrics:')
    for k, v in compute_metrics(tr_obs, tr_pred).items():
        print(f'  {k}: {v:.3f}')
    print('test metrics:')
    for k, v in compute_metrics(te_obs, te_pred).items():
        print(f'  {k}: {v:.3f}')

    df_comb   = pd.concat([df_feat_fit, df_feat_test])
    Q_plot    = np.expm1(df_comb['Q'])  if LOG_TRANSFORM else df_comb['Q']
    obs_plot  = np.expm1(y_test)        if LOG_TRANSFORM else y_test
    pred_plot = np.expm1(y_test_pred)   if LOG_TRANSFORM else y_test_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df_comb.index, Q_plot, color='steelblue', linewidth=0.6, label='observed')
    axes[0].plot(df_feat_test.index, pred_plot, color='tomato', linewidth=1.2, label='predicted')
    axes[0].axvline(pd.Timestamp(TEST_START), color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_title(MODEL)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Q (ft^3/s)')
    axes[0].legend(fontsize=8)

    axes[1].scatter(obs_plot, pred_plot, alpha=0.25, s=6, color='steelblue')
    lim = max(obs_plot.max(), pred_plot.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], 'k--', linewidth=0.9)
    axes[1].set_xlabel('Observed (ft^3/s)')
    axes[1].set_ylabel('Predicted (ft^3/s)')
    axes[1].set_title('predicted vs observed')

    resid = pred_plot - obs_plot
    axes[2].hist(resid, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
    axes[2].axvline(0, color='tomato', linestyle='--')
    axes[2].set_xlabel('residual (ft^3/s)')
    axes[2].set_title('residuals')

    plt.tight_layout()
    plt.savefig(f'outputs/figures/validation_{MODEL}.png', dpi=150, bbox_inches='tight')
    plt.show()

# forecast
print(f'\n5-day forecast starting {FORECAST_START}:')
forecast = make_forecast(df_all, model, MODEL, feat_cols, FORECAST_START, N_DAYS, LOG_TRANSFORM)
print(forecast.round(1).to_frame().to_string())

last_obs = df_all['Q'].iloc[-60:]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(last_obs.index, last_obs.values, color='steelblue', linewidth=1.2, label='observed')
ax.plot(forecast.index, forecast.values, 'o-', color='tomato',
        linewidth=1.8, markersize=6, label='5-day forecast')
ax.axvline(forecast.index[0], color='gray', linestyle='--', linewidth=0.9)
ax.set_title(f'5-day forecast -- Verde River  |  {MODEL}')
ax.set_xlabel('Date')
ax.set_ylabel('Q (ft^3/s)')
ax.legend()
plt.tight_layout()
plt.savefig(f'outputs/figures/forecast_{MODEL}.png', dpi=150, bbox_inches='tight')
plt.show()

print('done')
