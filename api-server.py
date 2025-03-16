import torch
import torchaudio
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
from typing import Optional, List, Dict, Any, Union
import time
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 

# Load the model
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class AudioRequest(BaseModel):
    text: str
    language: str = "en-us"
    emotion: list[float] = Field(default_factory=lambda: [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077])
    fmax: float = 22050.0
    pitch_std: float = 20.0
    speaking_rate: float = 15.0
    speaker_audio: str | None = None  # Base64-encoded audio for speaker embedding
    
# Models for OpenAI compatibility
class AudioContent(BaseModel):
    data: str  # Base64 encoded audio data
    format: str = "wav"

class InputAudio(BaseModel):
    input_audio: AudioContent

class ChatCompletionContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    input_audio: Optional[AudioContent] = None

class ChatCompletionMessage(BaseModel):
    role: str
    content: Union[str, List[ChatCompletionContentItem]]

class AudioOptions(BaseModel):
    voice: str  # Required field that will be used for gender or voice clone indicator
    format: str = "wav"
    fmax: Optional[float] = 22050.0
    pitch_std: Optional[float] = 20.0
    speaking_rate: Optional[float] = 15.0
    emotion: Optional[str] = "neutral"  # New field for emotion selection

# Mapping of emotions to emotion vectors
EMOTION_MAP = {
    "neutral": [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
    "happy": [0.6, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05],
    "sad": [0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    "angry": [0.05, 0.05, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05],
    "surprised": [0.05, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05]
}


class ChatCompletionRequest(BaseModel):
    model: str
    modalities: List[str] = ["text", "audio"]
    audio: AudioOptions  # Required field
    messages: List[ChatCompletionMessage]

class AudioData(BaseModel):
    data: str  # Base64 encoded audio

class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str = ""
    audio: Optional[AudioData] = None

class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        text_content = ""
        speaker = None
        prompt_speech = None
        cond_dict = None
        last_message = request.messages[-1]
        if isinstance(last_message.content, str):
            text_content = last_message.content
        else:
            for item in last_message.content:
                if item.type == "text":
                    text_content = item.text
                elif item.type == "input_audio" and item.input_audio:
                    audio_data = base64.b64decode(item.input_audio.data)
                    with open("temp_speaker.wav", "wb") as f:
                        f.write(audio_data)
                    wav, sampling_rate = torchaudio.load("temp_speaker.wav")
                    speaker = model.make_speaker_embedding(wav, sampling_rate)
                    
        emotion_vector = EMOTION_MAP.get(request.audio.emotion, EMOTION_MAP["neutral"])
        
        if speaker is None:
            cond_dict = make_cond_dict(
                text=text_content,
                language="en-us",
                fmax=request.audio.fmax,
                pitch_std=request.audio.pitch_std,
                speaking_rate=request.audio.speaking_rate,
                emotion=emotion_vector
            )
        else:
            cond_dict = make_cond_dict(
                text=text_content,
                speaker=speaker,
                language="en-us",
                fmax=request.audio.fmax,
                pitch_std=request.audio.pitch_std,
                speaking_rate=request.audio.speaking_rate,
            )
        conditioning = model.prepare_conditioning(cond_dict)
        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()
        
        output_path = "output.wav"
        torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
        with open(output_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(
                message=ResponseMessage(
                    content=text_content,
                    audio=AudioData(data=audio_b64)
                )
            )]
        )
        return JSONResponse(content=response.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)