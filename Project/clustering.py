import pandas as pd
from sklearn.utils import resample

train = pd.read_csv('data/test.csv')

X_train = train.drop('Tag', axis=1)  # X_train contains only the features
y_train = train['Tag'] # y_train contains only the targets

# rebalance the element in the test set such that for each tag there will be the same amount of entries
test = pd.read_csv('data/test.csv')
X_test = test.drop('Tag', axis=1)
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

################# TODO HERE ##############




##########################################