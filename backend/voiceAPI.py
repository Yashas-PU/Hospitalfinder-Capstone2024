from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
import os
import uuid
import pygame
import uvicorn
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store temporary audio files
TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Initialize pygame mixer (for audio playback)
pygame.mixer.init()

# Pydantic model for the request body
class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "en"  # Default language is 'en'

@app.post("/text-to-speech/")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech and play the audio locally using pygame.

    Args:
        text (str): The text to convert to speech.
        language (str): Language code for speech synthesis.
    
    Returns:
        Message indicating the result of the speech generation.
    """
    # Validate input
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Generate speech
        tts = gTTS(text=request.text, lang=request.language)
        audio_file = f"{TEMP_AUDIO_DIR}/{uuid.uuid4().hex}.mp3"
        tts.save(audio_file)

        # Play the generated audio file using pygame
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Wait until the music finishes playing before continuing
        while pygame.mixer.music.get_busy():  # Check if music is still playing
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed

        return {"message": "Audio is playing locally."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid language code: {request.language}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app if the script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
