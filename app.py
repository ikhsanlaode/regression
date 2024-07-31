import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Membuat model regresi linier
model = LinearRegression()

# Melatih model
model.fit(X, y)

# Prediksi
y_pred = model.predict(X)

# Menampilkan hasil
print(f"Koefisien: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Visualisasi
plt.scatter(X, y, color='blue', label='Data Aktual')
plt.plot(X, y_pred, color='red', label='Garis Regresi')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()