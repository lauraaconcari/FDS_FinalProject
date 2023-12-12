import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carica i dati di addestramento
train_data = pd.read_csv('Project/data/train.csv')

# Carica i dati di test
test_data = pd.read_csv('Project/data/test.csv')

# Seleziona le caratteristiche pertinenti per il clustering
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']

# Normalizza le caratteristiche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data[features])
X_test_scaled = scaler.transform(test_data[features])

# Scegli il numero di cluster
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Aggiungi la colonna "cluster" al DataFrame di addestramento
train_data['cluster'] = kmeans.fit_predict(X_train_scaled)

# Aggiungi la colonna "cluster" al DataFrame di test
test_data['cluster'] = kmeans.predict(X_test_scaled)

# Assegna il cluster più comune a ciascuna persona nel dataset di test
mode_clusters_test = test_data.groupby('Tag')['cluster'].agg(lambda x: x.mode().iloc[0])

# Crea un dizionario che mappa ciascuna persona al cluster assegnato nel test set
cluster_assignment_test = dict(mode_clusters_test)

# Stampa il mapping persona-cluster
print(cluster_assignment_test)

# Assegna il cluster nel dataset di addestramento utilizzando il mapping del test set
train_data['cluster'] = train_data['Tag'].map(cluster_assignment_test)

# Seleziona solo le canzoni assegnate al cluster della persona specifica nel dataset di addestramento
selected_songs_train = test_data[['Tag', 'Track', 'cluster']]

# Esporta il nuovo dataset
selected_songs_train.to_csv('nuovo_dataset_train.csv', index=False)








