from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# load the diabetes dataset
diabetes_data, _ = load_diabetes(return_X_y=True)

# standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(diabetes_data)

# create a KMeans clustering model with 3 clusters
model = KMeans(n_clusters=3, random_state=42)

# fit the model to the data
model.fit(data)

# predict the cluster labels for the first 5 data points
labels = model.predict(data[:5])
print("Cluster labels:", labels)