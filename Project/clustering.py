import pandas as pd
from sklearn.utils import resample
from sklearn.cluster import KMeans

train = pd.read_csv('data/train.csv')
X_train = train.drop('Tag', axis=1)  # X_train contains only the features
columns_to_drop = ['Track', 'Artist'] #prendiamo solo i numerici
X_train_numerical = X_train.drop(columns=columns_to_drop)
y_train = train['Tag'] # y_train contains only the targets

# rebalance the element in the test set such that for each tag there will be the same amount of entries
test = pd.read_csv('data/test.csv')
X_test = test.drop('Tag', axis=1)
X_test_numerical = X_test.drop(columns=columns_to_drop)
y_test = test['Tag']  

tag_count = y_test.value_counts()
test_resampled = pd.DataFrame(columns=test.columns)
min_count = tag_count.min()

for tag in tag_count.index:
    tags = test[test['Tag'] == tag]
    tag_resampled = resample(tags, replace=True, n_samples=min_count, random_state=42)
    test_resampled = pd.concat([test_resampled, tag_resampled])

X_test = test_resampled.drop('Tag', axis=1)
y_test = test_resampled['Tag']

################# TO DO HERE ##############
number_of_clusters=5
kmeans = KMeans(n_clusters=number_of_clusters)
kmeans.fit(X_train)
test_clusters = kmeans.predict(X_test_numerical)
print(test_clusters[:10])


##########################################