# main.py

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from emotionDetect import predict_emotion
from dotenv import load_dotenv
load_dotenv()

# Verify that the credentials are loaded
if not all([os.getenv('SPOTIPY_CLIENT_ID'), os.getenv('SPOTIPY_CLIENT_SECRET'), os.getenv('SPOTIPY_REDIRECT_URI')]):
    print("Error: Missing Spotify credentials in .env file!")
    print("Please ensure SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, and SPOTIPY_REDIRECT_URI are set.")
    exit(1)

# Define Scope and Playlist
# Scopes required to read devices and control playback
SCOPE = "user-modify-playback-state user-read-playback-state user-read-currently-playing"
EMOTION_BASED_URI = {
    "happy": "spotify:playlist:37i9dQZF1EVJSvZp5AOML2",
    "sad": "spotify:playlist:37i9dQZF1EIg85EO6f7KwU",
    "angry": "spotify:playlist:37i9dQZF1EIhuCNl2WSFYd",
    "neutral": "spotify:playlist:37i9dQZF1EpnnJ84UBWHBI",
    "disgusted": "spotify:playlist:37i9dQZF1EVJSvZp5AOML2",
    "fearful": "spotify:playlist:37i9dQZF1EIdHmP6runabL",
    "surprised": "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M"
}
# PLAYLIST_URI = EMOTION_BASED_URI["happy"]

# Authentication 
try:
    print("🎵 Starting Spotify authentication...")
    print("📡 Make sure your callback server is running on localhost:8888")
    print("🌐 And your ngrok tunnel is active!")
    print("🔗 Callback URL: https://choreoid-caroline-nonelastically.ngrok-free.dev/callback")
    print("\n� Clearing cached tokens to ensure fresh authentication with new scopes...")
    
    # Clear any cached tokens to force re-authentication with new scopes
    cache_file = ".cache"
    # if os.path.exists(cache_file):
    #     os.remove(cache_file)
    #     print("🗑️ Cleared cached authentication")
    
    print("\n�🚀 Opening browser for authentication...")
    # SpotipyOAuth handles the full Authorization Code flow
    auth_manager = SpotifyOAuth(
        scope=SCOPE,
        show_dialog=True,  # Always show the auth dialog
        open_browser=True,  # Automatically open browser
        cache_path=cache_file  # Specify cache file path
    )
    
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    # Test the connection
    user_info = sp.current_user()
    print(f"✅ Authentication successful!")
    print(f"👤 Logged in as: {user_info['display_name']} ({user_info['id']})")

except Exception as e:
    print(f"❌ An error occurred during authentication: {e}")
    print("\n🔧 Troubleshooting checklist:")
    print("1. Is your callback server running on localhost:8888?")
    print("2. Is your ngrok tunnel active?")
    print("3. Have you added the redirect URI to your Spotify app settings?")
    print("4. Are your credentials correct in the .env file?")
    exit()

# Find Active Device (Required for Playback) 
def get_active_device(spotify_object):
    """Finds the ID of an active Spotify device."""
    devices = spotify_object.devices()
    active_device_id = None
    
    # Print available devices for user reference
    print("\nAvailable Devices:")
    if not devices['devices']:
        print("🚨 No active devices found. Please start playback on a Spotify app (phone, desktop, web) first.")
        return None
        
    for device in devices['devices']:
        print(f"- {device['name']} (ID: {device['id']}, Active: {device['is_active']})")
        if device['is_active']:
            active_device_id = device['id']
            
    if not active_device_id:
        print("⚠️ No device is currently marked as 'active'. Using the first available device.")
        active_device_id = devices['devices'][0]['id']

    return active_device_id

# Play the Playlist 
def play_playlist(spotify_object, device_id, uri):
    """Starts playback of a specific playlist on the given device."""
    try:
        spotify_object.start_playback(
            device_id=device_id,
            context_uri=uri
        )
        print(f"\n✅ Successfully started playing playlist: {uri} on device ID: {device_id}")
    except spotipy.SpotifyException as e:
        print(f"\n❌ Spotify API Error: {e}")
        print("Common fix: Make sure you have a Spotify application open and playing something, or try using the first device ID manually.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


def getPlaylistByEmotion(image_path):
    result = predict_emotion(image_path)
    emotion = result['emotion'].lower()
    if emotion in EMOTION_BASED_URI:
        print(f"Detected emotion: {emotion}. Selecting corresponding playlist.")
        return EMOTION_BASED_URI[emotion]
    else:
        print(f"No playlist found for emotion '{emotion}'. Defaulting to 'happy' playlist.")
        return EMOTION_BASED_URI["happy"]

def main(image_path):
    device_id = get_active_device(sp)
    if device_id:
        playlist_uri = getPlaylistByEmotion(image_path)
        play_playlist(sp, device_id, playlist_uri)    

if __name__ == "__main__":
    test_image_path = '../TestingImages/angry.jpeg'  
    result = main(test_image_path)