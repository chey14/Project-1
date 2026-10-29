import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.makedirs("results_dashboard", exist_ok=True)
np.random.seed(42); tf.random.set_seed(42)

df = pd.read_csv("data/House_1_daily.csv", parse_dates=["Time"]).dropna().sort_values("Time").reset_index(drop=True)
data = df["Aggregate"].values.reshape(-1, 1)
n = len(data)
train_end, val_end = int(n * 0.70), int(n * 0.85)

scaler = MinMaxScaler().fit(data[:train_end])
scaled = scaler.transform(data)

def create_sequences(vals, start, end, L=7):
    X, y = [], []
    for i in range(start, end):
        X.append(vals[i - L : i]); y.append(vals[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(scaled, 7, train_end)
X_val, y_val = create_sequences(scaled, train_end, val_end)
X_test, y_test = create_sequences(scaled, val_end, n)

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=100, d_model=64, **kw):
        super().__init__(**kw)
        self.max_len, self.d_model = max_len, d_model
    def build(self, input_shape):
        pos = np.arange(self.max_len)[:, None]
        div = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(pos * div); pe[:, 1::2] = np.cos(pos * div)
        self.pe = tf.constant(pe, dtype=tf.float32)
        super().build(input_shape)
    def call(self, x): return x + self.pe[: tf.shape(x)[1], :]

inp = layers.Input(shape=(7, 1))
x = layers.Dense(64)(inp)
x = PositionalEncoding(7, 64)(x)
x = layers.Dropout(0.15)(x)
attn = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.15)(x, x)
x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attn]))
ffn = layers.Dense(128, activation="relu")(x)
ffn = layers.Dense(64)(ffn)
x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ffn]))
x = layers.GlobalAveragePooling1D()(x)
out = layers.Dense(1)(layers.Dense(32, activation="relu")(x))

model = models.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

cb = [EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True), ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5)]
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150, batch_size=16, callbacks=cb, verbose=1)

pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
actual = scaler.inverse_transform(y_test).flatten()
res = pd.DataFrame({"Time": df["Time"].iloc[val_end:n].reset_index(drop=True), "Actual": actual, "Predicted": pred})
res.to_csv("results_dashboard/forecast_results.csv", index=False)

mae, rmse = mean_absolute_error(actual, pred), np.sqrt(mean_squared_error(actual, pred))
mape = np.nanmean(np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))) * 100
with open("results_dashboard/forecast_metrics.json", "w") as f:
    json.dump({"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}, f)
print(f"[OK] Transformer trained. MAPE: {mape:.2f}%")