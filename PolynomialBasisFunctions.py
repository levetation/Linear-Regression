import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.cos(x) + 0.1 * rng.randn(50)

# Machine learning pipeline
poly_model = make_pipeline(
    # Generates polynomial features up to a specified degree.
    # This could be anything, it could generate Gaussian features
    PolynomialFeatures(7), 
    LinearRegression() # Fits LR model to the polynomial features
)
poly_model.fit(x[:, np.newaxis], y)

# Predict y values
xfit = np.linspace(0, 10, 1000)[:, np.newaxis] # [:, newaxis] -> turns 1-D array into 2-D
yfit = poly_model.predict(xfit) 

# Plot data points
plt.scatter(x, y)
# Plot model
plt.plot(xfit, yfit, color='red', label='Model')
# Plot confidence intervals
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()