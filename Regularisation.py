"""
Basic functions with linear regression makes the model flexible.
But it can lkead to over-fitting.
i.e., if you choose too many Guassian basis functions, the results
don't end up looking as good: GuassianFeatures(30) instead of 
GuassianFeatures(20).

We can introduce 'regulisation' to penalise large values of model
parameters.

https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html#Ridge-regression-($L_2$-Regularization)
The most common form of regulisation is 'ridge regression' or L_2
regulisation or sometimes called Tikhonov regulisation.

Another type of regulisation: https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html#Lasso-regression-($L_1$-regularization)
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

# Machine learning pipeline

from sklearn.linear_model import Ridge
gauss_model = make_pipeline(
    GaussianFeatures(30),
    Ridge(alpha=0.1), # almost as efficient as the orginal regression model
    # LinearRegression()
)
gauss_model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)[:, np.newaxis] # [:, newaxis] -> turns 1-D array into 2-D
yfit = gauss_model.predict(xfit)


# Predict y values
xfit = np.linspace(0, 10, 1000)[:, np.newaxis] # [:, newaxis] -> turns 1-D array into 2-D
yfit = gauss_model.predict(xfit) 

# Plotting
plt.scatter(x, y)
plt.plot(xfit, yfit, color='red', label='Model')

# Plot settings.
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()