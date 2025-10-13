# main.py

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

# --- 1. Load Environment Variables ---
# NOTE: In a real environment, you'd use a library like 'python-dotenv' 
# to load these from the .env file, but for a simple script, Spotipy 
# automatically looks for them if they are set in your shell.

# --- 2. Define Scope and Playlist ---
# Scope required to control playback (play, pause, skip, etc.)
SCOPE = "user-modify-playback-state" 

# Replace this with the URI of the actual playlist you want to play!
PLAYLIST_URI = 'spotify:playlist:37i9dQZF1DXcBWIGoYBM5M' 

# --- 3. Authentication ---
try:
    # SpotipyOAuth handles the full Authorization Code flow
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPE))
    print("Authentication successful.")

except Exception as e:
    print(f"An error occurred during authentication: {e}")
    print("Please ensure your Client ID, Secret, and Redirect URI are correct.")
    exit()

# --- 4. Find Active Device (Required for Playback) ---
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

# --- 5. Play the Playlist ---
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


# --- Main Execution ---
device_id = get_active_device(sp)

if device_id:
    play_playlist(sp, device_id, PLAYLIST_URI)