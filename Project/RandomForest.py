import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Carica i dati di addestramento
train_data = pd.read_csv('Project/data/cleaned_train.csv')

# Carica i dati di test
test_data = pd.read_csv('Project/data/cleaned_test.csv')

# Seleziona le caratteristiche pertinenti per il clustering
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
# 'Tag' è la colonna delle etichette
X_train = train_data[features]
y_train = train_data['Tag']

# Dividi il dataset di addestramento in training e validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Crea e addestra il modello Random Forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Valuta le prestazioni del modello sul set di validazione
predictions_val = random_forest_model.predict(X_val)
accuracy_val = accuracy_score(y_val, predictions_val)
print(f'Accuracy on Validation Set: {accuracy_val}')

precision = precision_score(y_val, predictions_val, average='weighted')
recall = recall_score(y_val, predictions_val, average='weighted')
f1 = f1_score(y_val, predictions_val, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')


# Supponiamo che 'X_test' sia la matrice delle caratteristiche del dataset di test
X_test = test_data[features]

# Predici le etichette per il dataset di test
test_predictions = random_forest_model.predict(X_test)

# Ora 'test_predictions' contiene le etichette predette per il dataset di test
# Puoi utilizzare queste etichette per creare le playlist personalizzate per ogni persona
# Ad esempio, puoi filtrare il dataset per ogni persona e creare le rispettive playlist
test_data['predicted_Tag'] = test_predictions

# Ora puoi utilizzare 'test_data' per creare le playlist personalizzate per ogni persona
# Ad esempio, puoi filtrare il dataset per ogni persona e creare le rispettive playlist
for tag in test_data['predicted_Tag'].unique():
    playlist = test_data[test_data['predicted_Tag'] == tag]
    # Ora 'playlist' contiene le canzoni predette per la persona specifica
    # Puoi fare ulteriori elaborazioni per creare le playlist nel formato desiderato
    # ad esempio, salvare le playlist in file o caricarle su una piattaforma di streaming musicale

# Per salvare le playlist in file CSV
test_data.to_csv('playlists_RF.csv', index=False)