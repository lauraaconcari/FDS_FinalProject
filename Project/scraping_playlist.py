import spotipy
from spotipy.oauth2 import SpotifyOAuth
import spotipy.exceptions
import csv
import time
from config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI
from collections import defaultdict

scope = 'playlist-read-private'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='bb5b99dd86fe486eb4bcf131015a465c', client_secret='97a12a3b2f914b63ac341d85a5f0cea8', redirect_uri=REDIRECT_URI, scope=scope))

# copy the url and extract the id, which is on the right of '/'
# i need a list of playlist's ids
raggae = [('5TBKHHPYImlo3JFvmlnfAC','Gianluca'),('03sDEv7FN58Mb9CJOs1Tgn','Gianluca'),('4X1UqwchhBFITHBLT9tkNJ','Gianluca'),('5m9CzTLsTQ9eZi2FycUpIo','Gianluca'),('3gHH9sIe79r7Y082jIV1Cu','Gianluca'),('37i9dQZF1DXco4ODkIraMD','Gianluca')] 
classic = [('1kGtBpJnR0bPWX4JXi5wUo','Laura'),('2JV0xEGVwIKCuFo5EbAica','Laura'),('37i9dQZF1EIcUYESbdArbm','Laura'),('37i9dQZF1DWWEJlAGA9gs0','Laura'),('37i9dQZF1DWVFeEut75IAL','Laura'),('37i9dQZF1DWSMJ6QxunHcJ','Laura')]
hiphop = [('37i9dQZF1EQnqst5TRi17F','Giuliana'),('5DZ93TH6ABgWYRgdBNJs8O','Giuliana'),('37i9dQZF1DX186v583rmzp','Giuliana'),('37i9dQZF1DX48TTZL62Yht','Giuliana'),('37i9dQZF1DXbkfWVLd8wE3','Giuliana'),('3OrM2V0EsEeJf10WqqA8NP','Giuliana'),('4Re8POjxevmWFk8fcT71sU','Giuliana')]
techno = [('37i9dQZF1DWTiY3QffFUdg','Andrea'),('01aBVibTe6ia4BbCFQm4iD','Andrea'),('5Eec5PtiELmFRBsNgjyA0v','Andrea'),('1doRrkfcLJKhSb3agBqCch','Andrea'),('2oC9jBlyBkV5DZZC4TNXWB','Andrea'),('5Kb9Xeuzni24ZWL4XuE7RI','Andrea'),('0Ijh3qcWYkfBYnwWDC9kxQ','Andrea')]
rock = [('37i9dQZF1EQpj7X7UK8OOF','Erika'),('37i9dQZF1DWZNFWEuVbQpD','Erika'),('4t8cpLBAeXiT4MXyS7SbV6','Erika'),('37i9dQZF1DWXRqgorJj26U','Erika'),('4jOqGKvV7iu0ojea2pt9Te','Erika'),('37i9dQZF1DX6rsDrBNGuWW','Erika'),('37i9dQZF1DWWsq4e0rDzty','Erika')]
uris = raggae + classic + hiphop + techno + rock

# create a dict such that for each user there is a set of genres extraced from the rispective column
user_genres = defaultdict(set)

with open('data/our_songs.csv', 'w', newline='', encoding='utf-8') as songs:

    for pl,tag in uris:
        uri = 'spotify:playlist:' + pl
        results = sp.playlist_tracks(uri)
        features = sp.audio_features([results['items'][0]['track']['id']])[0]
        # # list of columns names
        columns = ['Track','Artist','Tag']
        columns = columns + list(features.keys())
        # write on the csv
        writer = csv.DictWriter(songs,fieldnames=columns)
        writer.writeheader()
        for item in results['items']:
            track = item['track']
            audio_features = sp.audio_features([track['id']])[0]
            audio_features['Track'] = track['name']
            audio_features['Artist'] = track['artists'][0]['name']
            audio_features['Tag'] = tag

            writer.writerow(audio_features)
