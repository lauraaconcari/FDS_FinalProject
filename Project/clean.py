import pandas as pd

# Carica i dati di addestramento e test
train_data = pd.read_csv('Project/data/train.csv')  
test_data = pd.read_csv('Project/data/test.csv')   

# Rimuovi le ripetizioni di tracce da entrambi i dataset
train_data_unique = train_data.drop_duplicates(subset='Track')
test_data_unique = test_data.drop_duplicates(subset='Track')

# Rimuovi le tracce presenti sia in train che in test da test
test_data_unique = test_data_unique[~test_data_unique['Track'].isin(train_data_unique['Track'])]

# Salva i dataset puliti su nuovi file CSV
train_data_unique.to_csv('Project/data/cleaned_train.csv', index=False)  
test_data_unique.to_csv('Project/data/cleaned_test.csv', index=False)   
