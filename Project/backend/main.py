from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def health():
    return {"message": "Server Running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"Received message: {request.message}")
    return ChatResponse(response="Ok")

@app.post("/api/voice", response_model=ChatResponse)
async def voice(audio: UploadFile = File(...)):
    print(f"Received audio file: {audio.filename}")
    print(f"Content type: {audio.content_type}")
    
    # audio_data = await audio.read()

    # write code to process audio and get response from model
    
    return ChatResponse(response="Ok")