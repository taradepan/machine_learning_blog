from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load the California Housing dataset
california_data, california_target = fetch_california_housing(return_X_y=True)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(california_data, california_target, test_size=0.2, random_state=42)

# create a Linear Regression object
regressor = LinearRegression()

# fit the model using training data
regressor.fit(X_train, y_train)

# predict the output using test data
y_pred = regressor.predict(X_test)

# calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)