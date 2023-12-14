import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Leggi il dataset
dataset = pd.read_csv('playlists_RF.csv')  # Sostituisci con il tuo percorso e nome del file CSV

# Lista delle features
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']

# Lista dei valori unici nella colonna 'TAG'
tag_values = dataset['Tag'].unique()

# Inizializza lo scaler
scaler = MinMaxScaler()

# Normalizza i valori delle features
dataset[features] = scaler.fit_transform(dataset[features])

# Visualizza gli istogrammi
bar_width = 0.35
index = list(range(len(features)))

for tag_value in tag_values:
    plt.figure(figsize=(12, 6))

    # Seleziona solo le righe relative al valore corrente di 'TAG'
    tag_data = dataset[dataset['Tag'] == tag_value]

    # Calcola il valore medio per ciascuna feature
    means_original = tag_data[features].mean()

    # Seleziona solo le righe relative al valore corrente di 'predicted_TAG'
    predicted_tag_data = dataset[dataset['predicted_Tag'] == tag_value]

    # Calcola il valore medio per ciascuna feature anche per 'predicted_TAG'
    means_predicted = predicted_tag_data[features].mean()

    # Crea un istogramma con due colonne per ogni feature (una per 'TAG' e una per 'predicted_TAG')
    plt.bar(index, means_original, bar_width, label=f'TAG: {tag_value} - Original', alpha=0.7)
    plt.bar([i + bar_width for i in index], means_predicted, bar_width, label=f'TAG: {tag_value} - Predicted', alpha=0.7, color='red', align='edge')

    plt.xlabel('Features')
    plt.ylabel('Mean Value')
    plt.title(f'TAG: {tag_value}')
    plt.xticks([i + bar_width / 2 for i in index], features)
    plt.legend()
    plt.show()
