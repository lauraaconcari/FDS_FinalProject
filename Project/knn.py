import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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

# Crea e addestra il modello KNN
k_values = [3, 5, 7, 9]
param_grid = {'n_neighbors': k_values}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)

# Valuta le prestazioni del modello KNN sul set di validazione
predictions_val_knn = knn_model.predict(X_val)
accuracy_val_knn = accuracy_score(y_val, predictions_val_knn)
print(f'Accuracy KNN: {accuracy_val_knn}')

precision_knn = precision_score(y_val, predictions_val_knn, average='weighted')
recall_knn = recall_score(y_val, predictions_val_knn, average='weighted')
f1_knn = f1_score(y_val, predictions_val_knn, average='weighted')

print(f'Precision KNN: {precision_knn}')
print(f'Recall KNN: {recall_knn}')
print(f'F1-Score KNN: {f1_knn}')

cm_knn = confusion_matrix(y_val, predictions_val_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels KNN')
plt.ylabel('True Labels')
plt.title('Confusion Matrix KNN')
plt.show()

# Supponiamo che 'X_test' sia la matrice delle caratteristiche del dataset di test
X_test_knn = test_data[features]

# Predici le etichette per il dataset di test utilizzando il modello KNN
test_predictions_knn = knn_model.predict(X_test_knn)

# Ora 'test_predictions_knn' contiene le etichette predette per il dataset di test con KNN
# Puoi utilizzare queste etichette per creare le playlist personalizzate per ogni persona
# Ad esempio, puoi filtrare il dataset per ogni persona e creare le rispettive playlist
test_data['predicted_Tag_KNN'] = test_predictions_knn

# Ora puoi utilizzare 'test_data' per creare le playlist personalizzate per ogni persona anche con KNN
# Ad esempio, puoi filtrare il dataset per ogni persona e creare le rispettive playlist
for tag_knn in test_data['predicted_Tag_KNN'].unique():
    playlist_knn = test_data[test_data['predicted_Tag_KNN'] == tag_knn]
    # Ora 'playlist_knn' contiene le canzoni predette per la persona specifica con KNN
    # Puoi fare ulteriori elaborazioni per creare le playlist nel formato desiderato
    # ad esempio, salvare le playlist in file o caricarle su una piattaforma di streaming musicale

# Per salvare le playlist in file CSV con KNN
test_data.to_csv('playlists_KNN.csv', index=False)
