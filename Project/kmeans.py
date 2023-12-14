import pandas as pd
from sklearn.utils import resample
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import defaultdict

train = pd.read_csv('data/cleaned_train.csv')
# we select a range of values for k, we need to find the best k in order to organize the songs
k_values = range(2, 15)  
# we consider only numeri feature
X_train = train.drop(['Track', 'Artist', 'Tag'], axis=1)
# in order to find the best k we need to save for each k what score we do get
inertias = []
silhouette_scores = []

# iterate over the values for k
for k in k_values:
    # run the model and append the results
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_train, kmeans.labels_))
# below, we will show two plot regarding how 'inertia' and 'silhouette_scores' changes along with the k
plt.figure(figsize=(10, 5))

# intertia plot
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker='o')
plt.title('Inertia')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

# silhouette plot
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
# in order to find the best K we can analyze these plots
# where the inertia starts to decrease more slowly or where the silhouette score is maximized.
# so there is a tradeoff between the silhouette score and the inertia. 
# given the plot, we will chose k = 10
best_k = 10
best_kmeans = KMeans(n_clusters=best_k, random_state=42)
best_kmeans.fit(X_train)

# add cluster numbers
train['Cluster'] = best_kmeans.labels_
 # this dict will store for each cluster the tags with the higher counts
tags_dict = defaultdict(str)

# create dict that has as keys the cluster numbers and as values the tag with the highest count
for cluster in range(best_k):
    cluster_tags_count = train[train['Cluster'] == cluster]['Tag'].value_counts()
    most_present_tag = cluster_tags_count.idxmax()
    tags_dict[cluster] = most_present_tag

# we substitute for each cluster number the computed tag
train['predicted_Tag'] = train['Cluster'].map(tags_dict)

playlists_data = train.drop_duplicates()
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
top_20.to_csv('top20_kmeans.csv', index=False)
