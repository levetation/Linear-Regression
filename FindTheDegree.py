import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.cos(x) + 0.1 * rng.randn(50)

from sklearn.model_selection import train_test_split, cross_val_score

# Split data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

degrees = [1, 2, 3, 4, 5, 6, 7]  # Degrees of polynomial to try
mse_scores = []

for degree in degrees:
    # Create polynomial features and fit model
    poly_model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    
    # Perform cross-validation
    scores = cross_val_score(poly_model, x_train[:, np.newaxis], y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Calculate mean MSE across folds
    max_mse = -scores.max()
    mse_scores.append(max_mse)

# Find degree with the best
best_degree = degrees[np.argmin(mse_scores)]
print("Best degree:", best_degree)

# Train final model using best degree on entire training set
final_poly_model = make_pipeline(
    PolynomialFeatures(best_degree),
    LinearRegression()
)
final_poly_model.fit(x_train[:, np.newaxis], y_train)

# Evaluate final model on validation set
val_score = final_poly_model.score(x_val[:, np.newaxis], y_val)
print("Validation R-squared with best degree:", val_score)
