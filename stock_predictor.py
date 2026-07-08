
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


print("Downloading stock data...")
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")


data = data.reset_index()


data["Day"] = range(len(data))


X = data[["Day"]]
y = data["Close"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


predicted = model.predict(X_test)


r2 = r2_score(y_test, predicted)
mse = mean_squared_error(y_test, predicted)

print("\n========== MODEL RESULTS ==========")
print("R² Score :", round(r2, 4))
print("Mean Squared Error :", round(mse, 2))


next_day = [[len(data)]]
next_price = model.predict(next_day)

print("\nPredicted Next Day Closing Price: $", round(float(next_price.ravel()[0]), 2))


plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(predicted, '--', label="Predicted Price", color="red")

plt.title("Stock Price Prediction using Linear Regression")
plt.xlabel("Test Samples")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)

plt.savefig("M:/StockPricePredictor/stock_prediction.png", dpi=300, bbox_inches="tight")
plt.show()