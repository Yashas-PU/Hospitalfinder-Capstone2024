from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gtts import gTTS
import os
import uuid
import pygame
import uvicorn
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from datetime import datetime
from flask import Flask, jsonify

# Initialize FastAPI app
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for storing temporary audio files
TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Initialize pygame mixer (for audio playback)
pygame.mixer.init()

# Text-to-Speech functionality
class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "en"  # Default language is English

@app.post("/text-to-speech/")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech and play it locally using pygame.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        tts = gTTS(text=request.text, lang=request.language)
        audio_file = f"{TEMP_AUDIO_DIR}/{uuid.uuid4().hex}.mp3"
        tts.save(audio_file)

        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        return {"message": "Audio is playing locally."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid language code: {request.language}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Medical Chatbot functionality
preLoadData = """You are a medical diagnostic assistant. Provide a diagnosis and recommendations based on the following input.
Format your response as follows:
- Name: [Patient's Name]
- Age: [Patient's Age]
- Assumed Issues: [List of possible issues]
- Most Common Solution: [Suggested solution]
- If Required, Recommended Medication: [List of medications]
- Contact: [Which doctor to meet: Cardiology/Orthopedics/...etc.]"""

class MedicalChatbot:
    def __init__(self):
        load_dotenv()  # Load environment variables
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Invalid/No API Key in .env file")
            return
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        self.chat = self.model.start_chat(history=[])

    def check_internet_connection(self):
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except requests.exceptions.ReadTimeout:
            return False
        except requests.exceptions.RequestException:
            return False

    def get_medical_response(self, member):
        name = member.get('name')
        age = member.get('age')
        gender = member.get('gender')
        diseases = member.get('diseases')
        message = member.get('message')

        question = f"where the given data of Patient are> Name: {name}, Age: {age}, Gender: {gender}, Diseases: {diseases}, Message: {message}"
        full_prompt = preLoadData + "\n" + question

        if not self.check_internet_connection():
            return {"message": "Sorry, can't connect to the internet. Please check your connection."}

        response = self.chat.send_message(full_prompt, stream=True)
        for chunk in response:
            if hasattr(chunk, 'text'):
                return {"message": " ".join(chunk.text for chunk in response)}

        return {"message": "No valid response received from the AI model."}

# Initialize chatbot
chatbot = MedicalChatbot()

# Model for medical chatbot input
class MedicalChatRequest(BaseModel):
    memberName: str
    dob: str
    gender: str
    diseases: str
    message: str

@app.post("/chat/")
async def chat(request: MedicalChatRequest):
    """
    Handle medical chat queries and get diagnosis from the chatbot.
    """
    name = request.memberName
    dob = request.dob
    gender = request.gender
    diseases = request.diseases
    message = request.message

    age = calculate_age(dob) if dob else None

    member = {
        'name': name,
        'age': age,
        'gender': gender,
        'diseases': diseases,
        'message': message
    }

    response = chatbot.get_medical_response(member)
    return response

def calculate_age(dob):
    try:
        dob_date = datetime.strptime(dob, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        # Handle the case where time info is not provided, just date
        dob_date = datetime.strptime(dob, '%Y-%m-%d')  # Adjust format if no time info
    today = datetime.today()
    age = today.year - dob_date.year
    if today.month < dob_date.month or (today.month == dob_date.month and today.day < dob_date.day):
        age -= 1
    return age



# Run the app with uvicorn when this script is executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
