import os
import uuid
import asyncio
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "Auralis AI"
OUTPUT_DIR = Path("generated")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Small is much easier to run locally.
# Change to "medium" if you have a capable GPU.
MODEL_SIZE = "small"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description="Generative AI music creation API",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# MODEL
# ============================================================

print(f"Loading MusicGen on {DEVICE}...")

model = MusicGen.get_pretrained(MODEL_SIZE)
model.set_generation_params(
    duration=10
)

print("Music model loaded.")


# ============================================================
# REQUEST MODELS
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 10
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0


# ============================================================
# HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


async def generate_music(request: GenerateRequest):

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="A music prompt is required."
        )

    duration = clamp(
        request.duration,
        1,
        30
    )

    temperature = clamp(
        request.temperature,
        0.1,
        2.0
    )

    filename = f"{uuid.uuid4().hex}.wav"

    output_path = OUTPUT_DIR / filename

    # MusicGen's generation is CPU/GPU intensive,
    # so run it outside the main async event loop.
    def generate():

        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )

        wav = model.generate(
            [request.prompt],
            progress=True
        )

        # MusicGen returns:
        # [batch, channels, samples]

        audio_write(
            str(output_path.with_suffix("")),
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )

    await asyncio.to_thread(generate)

    return filename


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    index_file = Path("static/index.html")

    if not index_file.exists():
        return HTMLResponse(
            "<h1>Auralis AI</h1><p>Frontend missing.</p>"
        )

    return index_file.read_text(
        encoding="utf-8"
    )


@app.post("/api/generate")
async def create_music(request: GenerateRequest):

    filename = await generate_music(request)

    return {
        "success": True,
        "filename": filename,
        "url": f"/api/audio/{filename}"
    }


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):

    path = OUTPUT_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Audio not found."
        )

    return FileResponse(
        path,
        media_type="audio/wav",
        filename=filename
    )


@app.get("/api/health")
async def health():

    return {
        "status": "online",
        "device": DEVICE,
        "model": MODEL_SIZE
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
