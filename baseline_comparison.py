import math, random
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

random.seed(42)

def cpu_load(dt):
    hour, weekday = dt.hour, dt.weekday()
    if   0 <= hour <  6: base = 5.0
    elif 6 <= hour <  9: base = 25.0 + (hour - 6) * 10
    elif 9 <= hour < 12: base = 55.0
    elif 12 <= hour < 14: base = 45.0
    elif 14 <= hour < 18: base = 65.0
    elif 18 <= hour < 20: base = 40.0
    else:                 base = 15.0
    if weekday >= 5: base *= 0.35
    if weekday == 4 and 16 <= hour < 18: base += 20.0
    spike = 30.0 if random.random() < 0.05 else 0.0
    noise = random.gauss(0, 3.0)
    return round(max(1.0, min(99.0, base + spike + noise)), 2)

now   = datetime(2024, 4, 1, tzinfo=timezone.utc)
start = now - timedelta(days=14)
timestamps, cpu_values = [], []
t = start
while t <= now:
    timestamps.append(t)
    cpu_values.append(cpu_load(t))
    t += timedelta(minutes=1)

y = np.array(cpu_values)
print(f"Сгенерировано точек: {len(y)}")

split = int(len(y) * 0.8)
y_train, y_test = y[:split], y[split:]

def metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mae_pct  = mae  / np.mean(y_true) * 100
    rmse_pct = rmse / np.mean(y_true) * 100
    print(f"{name:35s}  MAE={mae_pct:.2f}%  RMSE={rmse_pct:.2f}%  R²={r2:.3f}")
    return {"Метод": name, "MAE, %": round(mae_pct,2), "RMSE, %": round(rmse_pct,2), "R²": round(r2,3)}

y_naive = y[split-1 : split+len(y_test)-1]
r1 = metrics(y_test, y_naive, "Naïve (y(t+1) = y(t))")

history = list(y_train)
ma_pred = []
for i in range(len(y_test)):
    ma_pred.append(np.mean(history[-6:]))
    history.append(y_test[i])
r2_ = metrics(y_test, np.array(ma_pred), "Moving Average (MA-6)")

LAG = 6
def make_features(series, lag=6):
    X, yy = [], []
    for i in range(lag, len(series)):
        X.append(list(series[i-lag:i]) + [i])
        yy.append(series[i])
    return np.array(X), np.array(yy)

X_all, y_all = make_features(y, LAG)
split_lr = split - LAG
lr = LinearRegression()
lr.fit(X_all[:split_lr], y_all[:split_lr])
y_lr = lr.predict(X_all[split_lr:])
r3 = metrics(y_all[split_lr:], y_lr, "Linear Regression (LR)")

print("\n── Итоговая таблица ──")
df_res = pd.DataFrame([r1, r2_, r3])
print(df_res.to_string(index=False))
