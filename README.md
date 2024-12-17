# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import folium
from folium.plugins import HeatMap

# Load Data
file_path = '/content/drive/My Drive/Global Missing Migrants Dataset.csv'
data = pd.read_csv(file_path)

# Ensure Clean Column Names
data.columns = data.columns.str.strip()

# Extract and Clean Coordinates
data['Coordinates'] = data['Coordinates'].str.split(',')
data['Latitude'] = data['Coordinates'].apply(lambda x: float(x[0]) if isinstance(x, list) and len(x) > 0 else np.nan)
data['Longitude'] = data['Coordinates'].apply(lambda x: float(x[1]) if isinstance(x, list) and len(x) > 1 else np.nan)
data = data.dropna(subset=['Latitude', 'Longitude'])

# Convert Date to Standard Format
data['Reported Month'] = data['Reported Month'].fillna('01').str.zfill(2)
data['Date'] = pd.to_datetime(data['Incident year'].astype(str) + '-' + data['Reported Month'], errors='coerce')

# Group Data by Time and Location
time_series = data.groupby(['Date', 'Latitude', 'Longitude']).agg({
    'Total Number of Dead and Missing': 'sum',
    'Number of Survivors': 'sum'
}).reset_index()

# --- Feature Engineering ---
# Aggregate over months for seasonality
time_series['Year'] = time_series['Date'].dt.year
time_series['Month'] = time_series['Date'].dt.month

# Create Sequential Dataset for LSTM
aggregated = time_series.groupby(['Date']).agg({
    'Total Number of Dead and Missing': 'sum'
}).reset_index()

# Scaling Data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(aggregated[['Total Number of Dead and Missing']])

# Create Sequences for LSTM
sequence_length = 12  # Use past 12 months to predict next month
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i + sequence_length])
    y.append(scaled_data[i + sequence_length])

X, y = np.array(X), np.array(y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and Train LSTM Model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Predict Future Values
future_predictions = []
last_sequence = X_test[-1]  # Starting point for future prediction

for _ in range(12):  # Predict 12 months into the future
    prediction = lstm_model.predict(last_sequence.reshape(1, sequence_length, 1))
    future_predictions.append(prediction[0, 0])
    last_sequence = np.vstack([last_sequence[1:], prediction])

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# --- Random Forest for Risk Factors ---
features = ['Latitude', 'Longitude', 'Year', 'Month']
target = 'Total Number of Dead and Missing'

# Prepare Data
time_series['Total Number of Dead and Missing'] = time_series['Total Number of Dead and Missing'].fillna(0)
X_rf = time_series[features]
y_rf = time_series[target]

# Split Data
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_rf_train, y_rf_train)
rf_preds = rf_model.predict(X_rf_test)

# Evaluate Random Forest Model
rf_rmse = np.sqrt(mean_squared_error(y_rf_test, rf_preds))
rf_r2 = r2_score(y_rf_test, rf_preds)
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Random Forest R²: {rf_r2}")

# --- Visualizations ---
# LSTM: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(aggregated['Date'][-len(y_test):], scaler.inverse_transform(y_test), label='Actual')
plt.plot(aggregated['Date'][-len(y_test):], scaler.inverse_transform(lstm_model.predict(X_test)), label='Predicted')
plt.title("LSTM: Actual vs Predicted Migration Risks")
plt.legend()
plt.show()

# Heatmap of Risk
heatmap_data = time_series[['Latitude', 'Longitude', 'Total Number of Dead and Missing']].dropna()
heatmap_values = heatmap_data.values.tolist()

m = folium.Map(location=[20, 0], zoom_start=2)
HeatMap(heatmap_values, radius=10, blur=15).add_to(m)
m
# project
