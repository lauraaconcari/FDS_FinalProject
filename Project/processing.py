import pandas as pd
import numpy as np

def min_max_scaling(column):
    if column.min() < 0 or column.max() > 1:
        return (column - column.min()) / (column.max() - column.min())
    else:
        return column

train = pd.read_csv('data/our_songs.csv')
train = train.drop_duplicates()
train = train.dropna()
# normalize columns
to_normalize = ['danceability', 'energy', 'speechiness','acousticness','instrumentalness','liveness','valence']
train[to_normalize] = train[to_normalize].apply(min_max_scaling, axis=0)

train = train.drop(columns=['type','id','uri','track_href','analysis_url','duration_ms'])
train = train.reset_index(drop=True)

### preprocessing of test set
test = pd.read_excel('/Users/gianluca/Desktop/magistrale/primo anno/Primo semestre/Fundamentals of Data Science/Project/data/songs_df.xlsx')
test = test[['track_name','artists_names','artist_genres','acousticness',
             'danceability','energy','instrumentalness','key','liveness',
             'loudness','mode','speechiness','tempo','time_signature','valence']]
test = test.dropna()

# split the genres such that it becomes a list of strings
test['artist_genres'] = test['artist_genres'].str.split(';')

def categorize_genre(genre):
    rock_keywords = ['rock']
    hip_hop_keywords = ['rap', 'hip hop']
    classic_keywords = ['classical', 'orchestra', 'opera', 'symphony']
    techno_keywords = ['techno', 'electronic', 'house']
    reggaeton_keywords = ['reggaeton', 'dembow', 'latin', 'perreo','spanish','raggae']
    genre_lower = [g.lower() for g in genre]

    if any(keyword in genre_lower for keyword in rock_keywords):
        return 'Erika'
    elif any(keyword in genre_lower for keyword in hip_hop_keywords):
        return 'Giuliana'
    elif any(keyword in genre_lower for keyword in classic_keywords):
        return 'Laura'
    elif any(keyword in genre_lower for keyword in techno_keywords):
        return 'Andrea'
    elif any(keyword in genre_lower for keyword in reggaeton_keywords):
        return 'Gianluca'
    else:
        return 'other'
    
test['Tag'] = test['artist_genres'].apply(categorize_genre)
# drop the useless columns and rows
test = test[test['Tag'] != 'other']
test = test.drop(columns=['artist_genres'])
test = test.reset_index(drop=True)

# fix the artist name such that will appear only the first artist
test['artists_names'] = test['artists_names'].str.split(';').apply(lambda x: x[0])

## clean test: check if there are any rows in common
merge = pd.merge(train[['Track', 'Artist']], 
                 test[['track_name', 'artists_names']], 
                 how='left', left_on=['Track', 'Artist'], 
                 right_on=['track_name', 'artists_names'])
merge = merge[merge['Track'].isna()]
merge = merge.drop(['Track', 'Artist'], axis=1)

# print(merge)
# prepare target and test set
# rename the two column with different names
clmn_names = {
    'track_name': 'Track', 
    'artists_names': 'Artist'
    }
test.rename(columns=clmn_names, inplace=True)

# reorder the train set columns such that they'll be in the same order as the test set
train = train[['Track', 'Artist','acousticness', 'danceability', 'energy',
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
              'speechiness', 'tempo', 'time_signature', 'valence', 'Tag']]

# permute the rows because we extracted the playlist in order, the random seed is for reproducibility
train = train.sample(frac=1, random_state=42)
train.to_csv('data/train2.csv', index=False)
test.to_csv('data/test2.csv', index=False)