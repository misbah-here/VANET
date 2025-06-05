
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Prepare synthetic data
# -------------------------------
# Example: generate simulated input data (speed, acceleration, lane_change)
# Shape: (samples, time_steps, features)
num_samples = 10000
time_steps = 10  # past 10 seconds of data
features = 3  # speed, acceleration, lane_change

# Simulated normalized inputs
X = np.random.rand(num_samples, time_steps, features)

# Simulated trajectory output: [x, y] positions after 10s
y = np.random.rand(num_samples, 2)

# Normalize inputs
scaler_X = MinMaxScaler()
X_flat = X.reshape(-1, features)
X_scaled = scaler_X.fit_transform(X_flat).reshape(num_samples, time_steps, features)

# -------------------------------
# Step 2: Build the LSTM Model
# -------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, features)),
    LSTM(64),
    Dense(2)  # Output: predicted (x, y) position
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# -------------------------------
# Step 3: Train the Model
# -------------------------------
history = model.fit(X_scaled, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# -------------------------------
# Step 4: Evaluate the Model
# -------------------------------
predictions = model.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, predictions))
print(f"RMSE on training data: {rmse:.2f} meters")

# -------------------------------
# Optional: Plot Training Loss
# -------------------------------
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Training Curve')
plt.legend()
plt.show()
