import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/kaggle/input/agriculture-vegetables-fruits-time-series-prices/kalimati_tarkari_dataset.csv')

df.head()
df.info()

df.drop('SN', axis=1, inplace=True)

print("data starting from date", df['Date'].min())
print("data ending at date", df['Date'].max())

frequency_df = pd.DataFrame(df['Commodity'].value_counts()).reset_index().rename(columns={'index':'Commodity', 'Commodity':'Frequency'})

df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year

sns.lineplot(y=df[df['Commodity']=='Ginger']['Average'], x=df[df['Commodity']=='Ginger']['year'])

#getting values for Ginger
dfg = df[df['Commodity']=='Ginger'].reset_index(drop=True)

dfg['month'] = dfg['Date'].dt.month
dfg['day'] = dfg['Date'].dt.day
dfg.head()

sns.lineplot(x=dfg[(dfg['year']==2013)&(dfg['month']==7)]['Date'], 
             y=dfg[(dfg['year']==2013)&(dfg['month']==7)]['Average'])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

target = dfg['Average']
# Plot ACF and PACF
fig, ax = plt.subplots(nrows=2, figsize=(10, 8))

# ACF plot
plot_acf(target, ax=ax[0], lags=10)
ax[0].set_title('Autocorrelation Function (ACF)')

# PACF plot
plot_pacf(target, ax=ax[1], lags=10)
ax[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# Select the 'Average' column as the target variable
target = dfg['Average']

# Create lagged variables
lag_order = 3
lagged_data = pd.DataFrame()
for lag in range(1, lag_order + 1):
    lagged_data[f'Lag{lag}'] = target.shift(lag)

lagged_data['Average'] = target  # Include the original 'Average' column

# Remove rows with missing values introduced by shifting
lagged_data = lagged_data.dropna()

# Split data into train, validation, and test sets
train_size = int(0.8 * len(lagged_data))
val_size = int(0.1 * len(lagged_data))
test_size = len(lagged_data) - train_size - val_size

train_data = lagged_data[:train_size]
val_data = lagged_data[train_size:train_size+val_size]
test_data = lagged_data[train_size+val_size:]

# Split features and target variables
X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_val, y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# Scale the input features and target variable separately
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

X_val_scaled = X_scaler.transform(X_val)
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))

X_test_scaled = X_test  # Test set remains unscaled for now

# Reshape features for input to GRU model
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))

print(X_train_reshaped.shape)
print(X_val_reshaped.shape)
print(X_test.shape)

# Build the GRU model
model = Sequential()
model.add(GRU(64, input_shape=(lag_order, 1)))
model.add(Dense(1))


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Train the model with EarlyStopping
model.fit(X_train_reshaped, y_train_scaled, epochs=50, batch_size=16, verbose=1, callbacks=[early_stopping], validation_data=(X_val_reshaped, y_val_scaled))

# Evaluate the model
train_loss = model.evaluate(X_train_reshaped, y_train_scaled, verbose=0)
test_loss = model.evaluate(X_scaler.fit_transform(X_test).reshape((X_test.shape[0], X_test.shape[1], 1)), 
                           y_scaler.fit_transform(y_test.values.reshape(-1, 1)), verbose=0)

print(f"Train Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Train Loss: 0.0009
# Test Loss: 0.0055

#mean absolute percentage error
# Evaluate the model
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_scaler.fit_transform(X_test))


#Inverse transform the scaled predictions and actual values
y_train_pred = y_scaler.inverse_transform(y_train_pred)
y_test_pred = y_scaler.inverse_transform(y_test_pred)
y_train_actual = y_scaler.inverse_transform(np.array(y_train_scaled).reshape(-1, 1))
y_test_actual = (np.array(y_test).reshape(-1, 1))



# Calculate MAPE
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

train_mape = calculate_mape(y_train_actual, y_train_pred)
test_mape = calculate_mape(y_test_actual, y_test_pred)

print(f"Train MAPE: {train_mape:.2f}%")
print(f"Test MAPE: {test_mape:.2f}%")

# Train MAPE: 1.30%
# Test MAPE: 3.76%

#An MAPE (Mean Absolute Percentage Error) value of 4% suggests that, on average, the model's predictions deviate from the actual values by 4% of the actual values.




