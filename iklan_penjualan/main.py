# Import library yang dibutuhkan
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# 1. Membaca dataset dari CSV
df = pd.read_csv("Advertising.csv")

# Tampilkan 5 data awal
print("Contoh data:")
print(df.head())

# 2. Pilih fitur (X) dan target (Y) — sesuai kolom asli
X = df[['TV', 'radio', 'newspaper']]  # variabel input
Y = df[['sales']]                     # target output

# 3. Split data untuk training dan testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Buat model Linear Regression dan latih
model = LinearRegression()
model.fit(X_train, Y_train)

# 5. Evaluasi model
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\nEvaluasi Model:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")

# 6. Visualisasi hasil prediksi vs aktual
plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

# 7. Simpan model ke file
os.makedirs("models", exist_ok=True)
with open("models/linear_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model berhasil disimpan ke folder 'models/linear_model.pkl'")
