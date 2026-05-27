import pickle, psycopg2, numpy as np
import matplotlib.pyplot as plt

conn = psycopg2.connect(host="localhost", user="thesis",
                        password="thesis_pass", dbname="metrics_db")
cur = conn.cursor()
cur.execute("SELECT cpu_usage FROM metrics ORDER BY collected_at ASC LIMIT 200")
series = [r[0] for r in cur.fetchall()]

with open('ml/model/model.pkl', 'rb') as f:
    data = pickle.load(f)

model, scaler, lags = data['model'], data['scaler'], data['lags']
X, y = [], []
for i in range(lags, len(series)-1):
    arr = series[i-lags:i]
    feat = list(arr) + [np.mean(arr), np.std(arr),
                        np.max(arr), np.min(arr), 0, 0, 0, 0]
    X.append(feat); y.append(series[i+1])

preds = model.predict(scaler.transform(X))
residuals = np.array(y) - preds

plt.figure(figsize=(10, 4))
plt.scatter(preds, residuals, alpha=0.4, s=10)
plt.axhline(0, color='red', linewidth=1)
plt.xlabel('Прогнозное значение, %')
plt.ylabel('Остаток (факт − прогноз), %')
plt.title('График остатков прогнозирования')
plt.tight_layout()
plt.savefig('residuals.png', dpi=150)
