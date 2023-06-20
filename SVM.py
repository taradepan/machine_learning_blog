from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()

# Split the dataset into training and testing sets
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a SVM classifier object
clf = svm.SVC(kernel='linear')

# Fit the model using the training data
clf.fit(X_train, y_train)

# Predict the output using the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)