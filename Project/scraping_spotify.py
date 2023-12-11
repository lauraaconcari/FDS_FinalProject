import spotipy
from spotipy.oauth2 import SpotifyOAuth
import csv
from config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI

scope = 'user-library-read'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=scope))

# copy the url and extract the id, which is on the right of '/'
uri = 'spotify:playlist:37i9dQZF1DWWsq4e0rDzty'
results = sp.playlist_tracks(uri)

with open('songs_gl.csv', 'w', newline='', encoding='utf-8') as songs:
    features = sp.audio_features([results['items'][0]['track']['id']])[0]
    # list of columns names
    columns = ['Track','Artist','Tag']

    columns = columns + list(features.keys())
    # write on the csv
    writer = csv.DictWriter(songs, fieldnames=columns)
    writer.writeheader()
    for item in results['items']:
        track = item['track']
        audio_features = sp.audio_features([track['id']])[0]
        
        audio_features['Track'] = track['name']
        audio_features['Artist'] = track['artists'][0]['name']
        audio_features['Tag'] = 'Erika'

        writer.writerow(audio_features)
