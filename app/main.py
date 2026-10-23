from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os

from .models import GenerateRequest, GenerateResponse
from .generator import generate_dataset

app = FastAPI(
    title="Mechanical Dataset Generator API",
    description="Generate realistic synthetic mechanical fault datasets on demand",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "generated"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

@app.get("/")
def root():
    return {
        "message": "Mechanical Dataset Generator API",
        "docs": "/docs",
        "endpoints": ["/generate", "/download/{filename}"]
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    df, filepath = generate_dataset(
        n_samples=req.n_samples,
        classes=req.classes,
        noise_level=req.noise_level,
        normalize=req.normalize,
        seed=req.seed,
        output_dir=OUTPUT_DIR
    )
    
    class_counts = df["label"].value_counts().to_dict()
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    filename = Path(filepath).name
    
    return GenerateResponse(
        message="Dataset generated successfully",
        total_samples=len(df),
        classes=class_counts,
        download_url=f"/download/{filename}",
        file_size_mb=round(file_size, 2)
    )

@app.get("/download/{filename}")
def download(filename: str):
    filepath = Path(OUTPUT_DIR) / filename
    if not filepath.exists():
        return {"error": "File not found"}
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="text/csv"
    )