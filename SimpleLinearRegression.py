import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

# y = ax + b

rng = np.random.RandomState(1)

x = 10 * rng.rand(50)
y = 2 * x -5 + rng.randn(50)

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, max(x), 1000)
yfit = model.predict(xfit[:, np.newaxis])


plt.scatter(x, y)
plt.plot(xfit, yfit, color='red')
plt.grid()
plt.show()

