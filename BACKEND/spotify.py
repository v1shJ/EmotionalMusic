# main.py

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import sys
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

# Get emotion from command line argument
if len(sys.argv) < 2:
    print("❌ Error: Please provide an emotion!")
    print("📖 Usage: python spotify.py <emotion>")
    print("🎭 Available emotions: happy, sad, rage")
    print("💡 Example: python spotify.py sad")
    exit(1)

emotion = sys.argv[1].lower()  # Get the emotion and convert to lowercase

# Validate the emotion
if emotion not in EMOTION_BASED_URI:
    print(f"❌ Error: '{emotion}' is not a valid emotion!")
    print("🎭 Available emotions: " + ", ".join(EMOTION_BASED_URI.keys()))
    exit(1)

PLAYLIST_URI = EMOTION_BASED_URI[emotion]
print(f"🎭 Selected emotion: {emotion}")
print(f"🎵 Playlist URI: {PLAYLIST_URI}")

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

# ========== PLAYBACK CONTROL FUNCTIONS ==========

def play_playlist(spotify_object, device_id, uri):
    """Starts playback of a specific playlist on the given device."""
    try:
        spotify_object.start_playback(
            device_id=device_id,
            context_uri=uri
        )
        print(f"\n✅ Successfully started playing playlist: {uri} on device ID: {device_id}")
        return True
    except spotipy.SpotifyException as e:
        print(f"\n❌ Spotify API Error: {e}")
        print("Common fix: Make sure you have a Spotify application open and playing something, or try using the first device ID manually.")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return False

def pause_playback(spotify_object, device_id=None):
    """Pause the currently playing track."""
    try:
        spotify_object.pause_playback(device_id=device_id)
        print("⏸️ Playback paused")
        return True
    except spotipy.SpotifyException as e:
        print(f"❌ Error pausing playback: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def resume_playback(spotify_object, device_id=None):
    """Resume the currently paused track."""
    try:
        spotify_object.start_playback(device_id=device_id)
        print("▶️ Playback resumed")
        return True
    except spotipy.SpotifyException as e:
        print(f"❌ Error resuming playback: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def next_track(spotify_object, device_id=None):
    """Skip to the next track."""
    try:
        spotify_object.next_track(device_id=device_id)
        print("⏭️ Skipped to next track")
        return True
    except spotipy.SpotifyException as e:
        print(f"❌ Error skipping to next track: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def previous_track(spotify_object, device_id=None):
    """Skip to the previous track."""
    try:
        spotify_object.previous_track(device_id=device_id)
        print("⏮️ Skipped to previous track")
        return True
    except spotipy.SpotifyException as e:
        print(f"❌ Error skipping to previous track: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def set_volume(spotify_object, volume_percent, device_id=None):
    """Set the playback volume (0-100)."""
    try:
        if not 0 <= volume_percent <= 100:
            print("❌ Volume must be between 0 and 100")
            return False
        
        spotify_object.volume(volume_percent, device_id=device_id)
        print(f"🔊 Volume set to {volume_percent}%")
        return True
    except spotipy.SpotifyException as e:
        print(f"❌ Error setting volume: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def toggle_shuffle(spotify_object, device_id=None):
    """Toggle shuffle mode on/off."""
    try:
        # Get current playback state to check if shuffle is on
        current = spotify_object.current_playback()
        if current and current.get('shuffle_state') is not None:
            new_shuffle_state = not current['shuffle_state']
            spotify_object.shuffle(new_shuffle_state, device_id=device_id)
            status = "ON" if new_shuffle_state else "OFF"
            print(f"🔀 Shuffle turned {status}")
            return True
        else:
            print("❌ Unable to get current playback state")
            return False
    except spotipy.SpotifyException as e:
        print(f"❌ Error toggling shuffle: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def set_repeat_mode(spotify_object, repeat_state="context", device_id=None):
    """Set repeat mode: 'track', 'context', or 'off'."""
    try:
        valid_states = ['track', 'context', 'off']
        if repeat_state not in valid_states:
            print(f"❌ Invalid repeat state. Use: {', '.join(valid_states)}")
            return False
            
        spotify_object.repeat(repeat_state, device_id=device_id)
        print(f"🔁 Repeat mode set to: {repeat_state}")
        return True
    except spotipy.SpotifyException as e:
        print(f"❌ Error setting repeat mode: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def seek_to_position(spotify_object, position_ms, device_id=None):
    """Seek to a specific position in the current track (in milliseconds)."""
    try:
        spotify_object.seek_track(position_ms, device_id=device_id)
        seconds = position_ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        print(f"⏩ Seeked to {minutes:02d}:{seconds:02d}")
        return True
    except spotipy.SpotifyException as e:
        print(f"❌ Error seeking: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def get_current_track_info(spotify_object):
    """Get information about the currently playing track."""
    try:
        current = spotify_object.current_playback()
        if current and current.get('item'):
            track = current['item']
            artists = ', '.join([artist['name'] for artist in track['artists']])
            album = track['album']['name']
            duration_ms = track['duration_ms']
            progress_ms = current.get('progress_ms', 0)
            
            # Convert to minutes:seconds format
            duration_sec = duration_ms // 1000
            duration_min = duration_sec // 60
            duration_sec = duration_sec % 60
            
            progress_sec = progress_ms // 1000
            progress_min = progress_sec // 60
            progress_sec = progress_sec % 60
            
            is_playing = current.get('is_playing', False)
            status = "▶️ Playing" if is_playing else "⏸️ Paused"
            
            print(f"\n🎵 Current Track Info:")
            print(f"📀 Track: {track['name']}")
            print(f"👨‍🎤 Artist(s): {artists}")
            print(f"💿 Album: {album}")
            print(f"⏱️ Progress: {progress_min:02d}:{progress_sec:02d} / {duration_min:02d}:{duration_sec:02d}")
            print(f"📊 Status: {status}")
            print(f"🔀 Shuffle: {'ON' if current.get('shuffle_state') else 'OFF'}")
            print(f"🔁 Repeat: {current.get('repeat_state', 'Unknown')}")
            
            return {
                'track_name': track['name'],
                'artists': artists,
                'album': album,
                'is_playing': is_playing,
                'progress_ms': progress_ms,
                'duration_ms': duration_ms
            }
        else:
            print("🔇 No track is currently playing")
            return None
    except spotipy.SpotifyException as e:
        print(f"❌ Error getting track info: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

def search_and_play_track(spotify_object, query, device_id=None):
    """Search for a track and play it."""
    try:
        # Search for tracks
        results = spotify_object.search(q=query, type='track', limit=1)
        tracks = results['tracks']['items']
        
        if not tracks:
            print(f"❌ No tracks found for: {query}")
            return False
            
        track = tracks[0]
        track_uri = track['uri']
        artists = ', '.join([artist['name'] for artist in track['artists']])
        
        # Play the track
        spotify_object.start_playback(device_id=device_id, uris=[track_uri])
        print(f"🎵 Now playing: {track['name']} by {artists}")
        return True
        
    except spotipy.SpotifyException as e:
        print(f"❌ Error searching/playing track: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False


# ========== INTERACTIVE MUSIC CONTROL LOOP ==========

def show_help():
    """Display available commands."""
    print("\n🎵 ===== SPOTIFY MUSIC CONTROLLER =====")
    print("📖 Available Commands:")
    print("  play          - ▶️  Start playing the selected playlist")
    print("  pause         - ⏸️  Pause playback")
    print("  resume        - ▶️  Resume playback") 
    print("  next          - ⏭️  Skip to next track")
    print("  prev          - ⏮️  Go to previous track")
    print("  info          - 🎵  Show current track info")
    print("  volume <0-100> - 🔊  Set volume (e.g., 'volume 75')")
    print("  shuffle       - 🔀  Toggle shuffle mode")
    print("  repeat <mode> - 🔁  Set repeat ('track', 'context', 'off')")
    print("  seek <mm:ss>  - ⏩  Seek to position (e.g., 'seek 1:30')")
    print("  search <query> - 🔍  Search and play track")
    print("  help          - ❓  Show this help message")
    print("  exit          - 🚪  Exit the program")
    print("=" * 40)

def parse_time_to_ms(time_str):
    """Convert mm:ss format to milliseconds."""
    try:
        if ':' in time_str:
            minutes, seconds = map(int, time_str.split(':'))
            return (minutes * 60 + seconds) * 1000
        else:
            # If no colon, treat as seconds
            return int(time_str) * 1000
    except ValueError:
        return None

# Main Execution 
device_id = get_active_device(sp)

if not device_id:
    print("❌ No active device found. Exiting...")
    exit(1)

# Start with playing the emotion-based playlist
print(f"\n🎭 Starting with {emotion} playlist...")
play_playlist(sp, device_id, PLAYLIST_URI)

# Show help on startup
show_help()

print("\n🎮 Music Controller Ready! Type 'help' for commands or 'exit' to quit.")

# Interactive control loop
while True:
    try:
        user_input = input("\n🎵 Enter command: ").strip().lower()
        
        if not user_input:
            continue
            
        # Parse command and arguments
        parts = user_input.split(' ', 1)
        command = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        # Execute commands
        if command == "exit" or command == "quit":
            print("👋 Goodbye! Thanks for using Spotify Music Controller!")
            break
            
        elif command == "help" or command == "?":
            show_help()
            
        elif command == "play":
            play_playlist(sp, device_id, PLAYLIST_URI)
            
        elif command == "pause":
            pause_playback(sp, device_id)
            
        elif command == "resume":
            resume_playback(sp, device_id)
            
        elif command == "next":
            next_track(sp, device_id)
            
        elif command == "prev" or command == "previous":
            previous_track(sp, device_id)
            
        elif command == "info":
            get_current_track_info(sp)
            
        elif command == "volume":
            if args:
                try:
                    volume = int(args)
                    set_volume(sp, volume, device_id)
                except ValueError:
                    print("❌ Invalid volume. Use: volume <0-100>")
            else:
                print("❌ Please specify volume level. Use: volume <0-100>")
                
        elif command == "shuffle":
            toggle_shuffle(sp, device_id)
            
        elif command == "repeat":
            if args and args in ['track', 'context', 'off']:
                set_repeat_mode(sp, args, device_id)
            else:
                print("❌ Invalid repeat mode. Use: repeat <track|context|off>")
                
        elif command == "seek":
            if args:
                position_ms = parse_time_to_ms(args)
                if position_ms is not None:
                    seek_to_position(sp, position_ms, device_id)
                else:
                    print("❌ Invalid time format. Use: seek <mm:ss> or seek <seconds>")
            else:
                print("❌ Please specify time. Use: seek <mm:ss> or seek <seconds>")
                
        elif command == "search":
            if args:
                search_and_play_track(sp, args, device_id)
            else:
                print("❌ Please specify search query. Use: search <song name artist>")
                
        else:
            print(f"❌ Unknown command: '{command}'. Type 'help' for available commands.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using Spotify Music Controller!")
        break
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Type 'help' for available commands or 'exit' to quit.")