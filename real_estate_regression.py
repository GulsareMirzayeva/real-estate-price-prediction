"""
🏠 Simple Linear Regression - Real Estate Price Prediction
----------------------------------------------------------

This script predicts apartment prices based on their size (in square feet)
using a linear regression model from scikit-learn.

Steps:
- Load the dataset
- Train a linear regression model (price ~ size)
- Print model parameters (intercept, coefficient, R² score)
- Predict the price of a 750 sq.ft apartment
- Plot and save the regression line vs actual data
"""
# Öyrəndiklərim
# -----------------------------------------------------------
# CSV faylından məlumatları oxumaq
# Sadə xətti regresiya modelini qurmaq (x 2D olmalıdır)
# Intercept, coefficient və R² score nə deməkdir, necə təhlil olunur
# Yeni sahə üçün qiymət proqnozu çıxarmaq
# Modelin vizual təhlilini çəkərək nəticələri qrafiklə göstərmək


# 📦 Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 📂 Load the dataset
data = pd.read_csv("real_estate_price_size.csv")

# Separate the input (size) and target (price)
X = data["size"].values.reshape(-1, 1)  # Input must be 2D for sklearn
y = data["price"]                       # Output: apartment price

# 🔁 Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# 🎯 Get model parameters
r2 = model.score(X, y)             # R-squared: how well the model explains the data
intercept = model.intercept_       # β₀: the starting price when size = 0
coefficient = model.coef_[0]       # β₁: how much price increases per sq.ft

# 📋 Print summary results
print("🔍 Model Summary")
print(f"Intercept (β₀): {intercept:.2f}")
print(f"Coefficient (β₁): {coefficient:.2f}")
print(f"R² Score: {r2:.4f}")

# 🔮 Predict the price for a 750 sq.ft apartment
predicted_price = model.predict([[750]])
print(f"Predicted price for 750 sq.ft: ${predicted_price[0]:,.2f}")

# 📊 Plot the actual data and the regression line
yhat = model.predict(X)
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, yhat, color="orange", label="Regression line")

plt.xlabel("Size (sq.ft)")
plt.ylabel("Price ($)")
plt.title("Linear Regression – Real Estate")
plt.legend()
plt.savefig("plot.png")  # Save the plot image
plt.show()
