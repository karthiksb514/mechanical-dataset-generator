from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateRequest(BaseModel):
    n_samples: int = Field(10000, ge=1000, le=200000, description="Total samples")
    classes: List[str] = Field(
        default=["NORMAL", "BEARING_WEAR", "OVERLOAD", "MISALIGNMENT", "IMBALANCE"],
        description="Fault classes to include"
    )
    noise_level: float = Field(0.05, ge=0.0, le=0.3, description="Gaussian noise level")
    normalize: bool = Field(True, description="Apply Min-Max scaling [0-1]")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")

class GenerateResponse(BaseModel):
    message: str
    total_samples: int
    classes: dict
    download_url: str
    file_size_mb: float