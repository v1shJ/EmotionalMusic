#!/usr/bin/env python3
"""
FastAPI server to handle Spotify OAuth callback
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Spotify OAuth Callback Server")

@app.get("/")
async def root():
    return {"message": "Spotify OAuth Callback Server is running!"}

@app.get("/callback")
async def spotify_callback(request: Request, code: str = None, error: str = None):
    """Handle Spotify OAuth callback"""
    if error:
        return HTMLResponse(f"""
        <html>
            <head><title>Spotify Auth Error</title></head>
            <body>
                <h1>Authentication Error</h1>
                <p>Error: {error}</p>
                <p>Please try again.</p>
            </body>
        </html>
        """)
    
    if code:
        # Success! We got the authorization code
        return HTMLResponse(f"""
        <html>
            <head>
                <title>Spotify Auth Success</title>
                <style>
                    body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                    .success {{ color: #1db954; }}
                    .code {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1 class="success">✅ Authentication Successful!</h1>
                <p>Authorization code received successfully.</p>
                <p>You can now close this tab and return to your application.</p>
                <div class="code">
                    <small>Auth Code: {code[:20]}...</small>
                </div>
            </body>
        </html>
        """)
    
    return HTMLResponse("""
    <html>
        <head><title>Spotify Auth</title></head>
        <body>
            <h1>Waiting for Spotify authorization...</h1>
            <p>Please complete the authorization in Spotify.</p>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "spotify-callback-server"}

if __name__ == "__main__":
    print("🎵 Starting Spotify OAuth Callback Server...")
    print("🌐 Server will be available at: http://localhost:8888")
    print("🔗 Callback endpoint: http://localhost:8888/callback")
    print("📡 Through ngrok: https://choreoid-caroline-nonelastically.ngrok-free.dev/callback")
    print("\n🚀 Starting server...")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8888,
        log_level="info"
    )