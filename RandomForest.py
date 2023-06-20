from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the iris dataset
iris_data, iris_target = load_iris(return_X_y=True)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=42)

# create a Random Forest Classifier object
classifier = RandomForestClassifier()

# fit the model using training data
classifier.fit(X_train, y_train)

# predict the output using test data
y_pred = classifier.predict(X_test)

# calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)