import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])


m, c = np.polyfit(x, y, 1)
print("Slope (m):", m)
print("Intercept (c):", c)
y_pred = m * x + c

# Calculate errors after y_pred is defined and before mse is calculated
errors = y - y_pred

mse = np.mean(errors**2)
print("Mean Squared Error:", mse)
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', label='Regression Line')

# Error lines
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'g--')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression with Errors')
plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

y_pred = model.predict(x.reshape(-1, 1))