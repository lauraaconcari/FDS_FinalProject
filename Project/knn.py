import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train = pd.read_csv('Project/data/cleaned_train.csv')

# rebalance the element in the test set such that for each tag there will be the same amount of entries
test = pd.read_csv('Project/data/cleaned_test.csv')

X_test = test.drop(['Track', 'Artist', 'Tag'], axis=1)  # Exclude 'Track', 'Artist', and 'Tag' columns
y_test = test['Tag']

tag_count = y_test.value_counts()
test_resampled = pd.DataFrame(columns=test.columns)
min_count = tag_count.min()

for tag in tag_count.index:
    tags = test[test['Tag'] == tag]
    tag_resampled = resample(tags, replace=True, n_samples=min_count, random_state=42)
    test_resampled = pd.concat([test_resampled, tag_resampled])

# drop the features that are not useful for the classification
X_train = train.drop(['Track', 'Artist', 'Tag'], axis=1)  # Exclude 'Track', 'Artist', and 'Tag' columns
y_train = train['Tag'] # get the true labels for train

# drop the useless features
X_test = test_resampled.drop(['Track', 'Artist', 'Tag'], axis=1)  # Exclude 'Track', 'Artist', and 'Tag' columns
y_test = test_resampled['Tag']

# choose some values (odd) for K over which we are gonna iterate in order to find the best k
k_values = [3, 5, 7, 9]

param_grid = {'n_neighbors': k_values}
# use gridsearch in order to find the best k

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# find the best k
best_k = grid_search.best_params_['n_neighbors']

# once found the best k, we are gonna work on that dataset
knn = KNeighborsClassifier(n_neighbors=best_k)
# train model
knn.fit(X_train, y_train)
# test the model
y_pred = knn.predict(X_test)
# compute some useful metrics
precision = precision_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Precision: {precision}')
print(f"Accuracy: {accuracy:.4f}")
print(f'Recall: {recall}')
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# add the columns with the predictions, this will be usefull later when we will create the playlists
test_resampled['predicted_Tag'] = y_pred

playlists_data = test_resampled.drop_duplicates()
playlists_data = playlists_data.sort_values(['predicted_Tag', 'Track'])

# initialize an empty dataframe which will contains the top 20 songs
top_20 = pd.DataFrame()

# iterate over the tags, for each tag we are gonna add to the dataframe the first 20 songs
for tag in playlists_data['predicted_Tag'].unique():
    tag_playlist = playlists_data[playlists_data['predicted_Tag'] == tag].head(20)
    top_20 = pd.concat([top_20, tag_playlist])

# we don't need to display any features beyond the track name and the tag
top_20 = top_20[['Track', 'predicted_Tag']]

# create the csv which contains the playlist
top_20.to_csv('top20_knn.csv', index=False)