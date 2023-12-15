import pandas as pd

# Carica il file 'playlists.csv'
playlists_data = pd.read_csv('playlists_KNN.csv')  # Sostituisci 'path_to_playlists.csv' col percorso effettivo

# Ordina il DataFrame in base alle colonne 'Tag' e 'Song'
playlists_data.sort_values(['predicted_Tag_KNN', 'Track'], inplace=True)

# Crea un nuovo DataFrame per contenere le prime 20 canzoni per ogni tag
top20_songs_per_tag = pd.DataFrame()

# Estrai le prime 20 canzoni per ogni tag
for tag in playlists_data['predicted_Tag_KNN'].unique():
    tag_playlist = playlists_data[playlists_data['predicted_Tag_KNN'] == tag].head(20)
    top20_songs_per_tag = pd.concat([top20_songs_per_tag, tag_playlist])

# Seleziona solo le colonne 'Track' e 'Tag'
top20_songs_per_tag = top20_songs_per_tag[['Track', 'predicted_Tag_KNN']]

# Salva il nuovo DataFrame in un nuovo file CSV
top20_songs_per_tag.to_csv('top20_songs_per_tag_KNN.csv', index=False)  # Sostituisci 'path_to_top20_songs_per_tag.csv' col percorso desiderato