from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

# Read reference audio
with open("assets/output.mp3", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# Define the text input
text_input = "This is the text to be converted using the cloned voice. I have a dream... I need to be the best ai voice clone out there."

# Voice cloning request with required text field
response = client.chat.completions.create(
    model="zonos",
    modalities=["text", "audio"],
    audio={
        "voice": "clone",
        "format": "wav",
        "pitch_std": 60,
        "speaking_rate": 30,
        "emotion": "angry"
    },
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_input},  # Ensure text is also in messages
                {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}}
            ]
        }
    ]
)

# Save the generated audio
with open("cloned_output.wav", "wb") as f:
    f.write(base64.b64decode(response.choices[0].message.audio.data))

print("Audio saved as cloned_output.wav")
