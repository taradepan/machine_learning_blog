from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# load the breast cancer dataset
breast_cancer_data, breast_cancer_target = load_breast_cancer(return_X_y=True)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, breast_cancer_target, test_size=0.2, random_state=42)

# create a StandardScaler object
scaler = StandardScaler()

# fit the scaler using training data
scaler.fit(X_train)

# transform the training and testing data using the scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create a Logistic Regression object
classifier = LogisticRegression()

# fit the model using scaled training data
classifier.fit(X_train_scaled, y_train)

# predict the output using scaled test data
y_pred = classifier.predict(X_test_scaled)

# calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)