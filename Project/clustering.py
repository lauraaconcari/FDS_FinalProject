import pandas as pd
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read training data
train = pd.read_csv('Project/data/train.csv')
X_train = train.drop('Tag', axis=1)
columns_to_drop = ['Track', 'Artist']
X_train_numerical = X_train.drop(columns=columns_to_drop)

# Read and preprocess test data
test = pd.read_csv('Project/data/test.csv')
X_test = test.drop('Tag', axis=1)
X_test_numerical = X_test.drop(columns=columns_to_drop)

# One-hot encoding for categorical variables
X_train_encoded = pd.get_dummies(X_train_numerical)
X_test_encoded = pd.get_dummies(X_test_numerical)

# Rebalance the test set
test_resampled = pd.DataFrame(columns=test.columns)
tag_count = test['Tag'].value_counts()
min_count = tag_count.min()

for tag in tag_count.index:
    tags = test[test['Tag'] == tag]
    tag_resampled = resample(tags, replace=True, n_samples=min_count, random_state=42)
    test_resampled = pd.concat([test_resampled, tag_resampled])

X_test_resampled = test_resampled.drop('Tag', axis=1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_resampled)

# Apply k-means clustering
number_of_clusters = 5
kmeans = KMeans(n_clusters=number_of_clusters, random_state=42)

# Fit on training data
kmeans.fit(X_train_scaled)

# Predict clusters on test data
test_clusters = kmeans.predict(X_test_scaled)
print(test_clusters[:10])

