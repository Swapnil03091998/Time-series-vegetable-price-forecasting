!pip install skforecast

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  HistGradientBoostingRegressor

from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast.model_selection_multiseries import grid_search_forecaster_multiseries

data = pd.read_csv('/kaggle/input/agriculture-vegetables-fruits-time-series-prices/kalimati_tarkari_dataset.csv')
data.head()

frequency_df = pd.DataFrame(data['Commodity'].value_counts()).reset_index().rename(columns={'index':'Commodity', 'Commodity':'Frequency'})
frequency_df.head(20)

# Data preprocessing
# ======================================================================================
selected_items = list(frequency_df['Commodity'][:20]) # Top 20 items
#selected_items = [1, 2, 3, 4 , 5] # Selection of items to reduce computation time

data = data[data['Commodity'].isin(selected_items)].copy()
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = pd.pivot_table(
           data    = data,
           values  = 'Average',
           index   = 'Date',
           columns = 'Commodity'
       )
data.columns.name = None
data.columns = [col for col in data.columns]
data = data.asfreq('1D')
data = data.sort_index()
data.head(4)

# Split data into train-validation-test
# ======================================================================================
end_train = '2019-10-31'
end_val = '2020-07-31'

data_train = data.loc[:end_train, :].copy()
data_val   = data.loc[end_train:end_val, :].copy()
data_test  = data.loc[end_val:, :].copy()

print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Validation dates : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})")
print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

# Plot time series
# ======================================================================================
fig, ax = plt.subplots(figsize=(9, 6))
data.iloc[:, :4].plot(
    legend   = True,
    subplots = True, 
    sharex   = True,
    title    = 'Average prices',
    ax       = ax, 
)
fig.tight_layout();

# Autocorrelation plot
# ======================================================================================
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(9, 7), sharex=True)
axes = axes.flat
for i, col in enumerate(data.columns[:4]):
    plot_acf(data[col], ax=axes[i], lags=7*5)
    axes[i].set_ylim(-1, 1.1)
    axes[i].set_title(f'{col}')
fig.tight_layout()
plt.show()

data.isna().sum()

for col in data.columns:
    data['{}'.format(col)] = data['{}'.format(col)].fillna(data['{}'.format(col)].mean())

# Train and backtest a model for all items: ForecasterAutoregMultiSeries
# ======================================================================================
items = list(data.columns)

# Define forecaster
forecaster_ms = ForecasterAutoregMultiSeries(
                    regressor          = HistGradientBoostingRegressor(random_state=123),
                    lags               = 14,
                    transformer_series = StandardScaler(),
                )

# Backtesting forecaster for all items
multi_series_mae, predictions_ms = backtesting_forecaster_multiseries(
                                       forecaster         = forecaster_ms,
                                       series             = data,
                                       levels             = items,
                                       steps              = 7,
                                       metric             = 'mean_absolute_error',
                                       initial_train_size = len(data_train) + len(data_val),
                                       refit              = False,
                                       fixed_train_size   = False,
                                       verbose            = False
                                   )

# Results
display(multi_series_mae.head(3))
print('')
display(predictions_ms.head(3))

multi_series_mae['mean_absolute_error'].mean() #comes out to be 11

# Hide progress bar tqdm
# ======================================================================================
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Hyperparameter search for the multi-series model and backtesting for each item
# ======================================================================================
lags_grid = [3, 7, 14]
param_grid = {
    'max_iter': [100, 500],
    'max_depth': [3, 5, 10, None],
    'learning_rate': [0.01, 0.1]
}

forecaster_ms = ForecasterAutoregMultiSeries(
                    regressor          = HistGradientBoostingRegressor(random_state=123),
                    lags               = 14, # Placeholder, the value will be overwritten
                    transformer_series = StandardScaler(),
                )

results_grid_ms = grid_search_forecaster_multiseries(
                      forecaster         = forecaster_ms,
                      series             = data.loc[:end_val, :],
                      levels             = None, # If None all levels are selected
                      lags_grid          = lags_grid,
                      param_grid         = param_grid,
                      steps              = 7,
                      metric             = 'mean_absolute_error',
                      initial_train_size = len(data_train),
                      refit              = False,
                      fixed_train_size   = False,
                      return_best        = True,
                      verbose            = False
                  )      

multi_series_mae, predictions_ms = backtesting_forecaster_multiseries(
                                       forecaster         = forecaster_ms,
                                       series             = data,
                                       levels             = None, # If None all levels are selected
                                       steps              = 7,
                                       metric             = 'mean_absolute_error',
                                       initial_train_size = len(data_train) + len(data_val),
                                       refit              = False,
                                       fixed_train_size   = False,
                                       verbose            = False
                                   )

# Train and backtest a model for all items: ForecasterAutoregMultiSeries
# ======================================================================================
items = list(data.columns)

# Define forecaster
forecaster_ms = ForecasterAutoregMultiSeries(
                    regressor          = HistGradientBoostingRegressor(random_state=123,
                                                                      learning_rate=0.1, 
                                                                       max_depth=3, 
                                                                       max_iter=100),
                    lags               = 7,
                    transformer_series = StandardScaler(),
                )

# Backtesting forecaster for all items
multi_series_mae, predictions_ms = backtesting_forecaster_multiseries(
                                       forecaster         = forecaster_ms,
                                       series             = data,
                                       levels             = items,
                                       steps              = 7,
                                       metric             = 'mean_absolute_error',
                                       initial_train_size = len(data_train) + len(data_val),
                                       refit              = False,
                                       fixed_train_size   = False,
                                       verbose            = False
                                   )

# Results
display(multi_series_mae.head(3))
print('')
display(predictions_ms.head(3))

multi_series_mae['mean_absolute_error'].mean() #comes out to be 10.87
