import numpy as np
import pandas as pd
from pathlib import Path
import uuid

# Class-specific parameters (realistic mechanical behavior)
CLASS_PARAMS = {
    "NORMAL": {
        "temperature": (45, 8), "pressure": (100, 12), "vibration": (2.5, 0.8),
        "load": (500, 50), "stress": (80, 15), "strain": (0.002, 0.0004),
        "cycle_count": (50000, 8000)
    },
    "BEARING_WEAR": {
        "temperature": (65, 12), "pressure": (95, 15), "vibration": (8.5, 2.5),
        "load": (520, 60), "stress": (110, 20), "strain": (0.0035, 0.0008),
        "cycle_count": (120000, 15000)
    },
    "OVERLOAD": {
        "temperature": (85, 15), "pressure": (140, 20), "vibration": (6.0, 1.8),
        "load": (900, 100), "stress": (180, 30), "strain": (0.006, 0.001),
        "cycle_count": (30000, 5000)
    },
    "MISALIGNMENT": {
        "temperature": (55, 10), "pressure": (105, 14), "vibration": (12.0, 3.5),
        "load": (480, 55), "stress": (95, 18), "strain": (0.0028, 0.0006),
        "cycle_count": (70000, 10000)
    },
    "IMBALANCE": {
        "temperature": (50, 9), "pressure": (98, 13), "vibration": (15.0, 4.0),
        "load": (510, 50), "stress": (100, 16), "strain": (0.003, 0.0007),
        "cycle_count": (60000, 9000)
    },
}

FEATURE_NAMES = ["temperature", "pressure", "vibration", "load", "stress", "strain", "cycle_count"]

def generate_dataset(
    n_samples: int = 10000,
    classes: list = None,
    noise_level: float = 0.05,
    normalize: bool = True,
    seed: int = 42,
    output_dir: str = "generated"
) -> tuple:
    
    if classes is None:
        classes = list(CLASS_PARAMS.keys())
    
    np.random.seed(seed)
    samples_per_class = n_samples // len(classes)
    
    all_data = []
    
    for cls in classes:
        params = CLASS_PARAMS[cls]
        data = {}
        for feat, (mean, std) in params.items():
            values = np.random.normal(mean, std, samples_per_class)
            # Add sensor noise
            values += np.random.normal(0, std * noise_level, samples_per_class)
            data[feat] = values
        
        df_cls = pd.DataFrame(data)
        df_cls["label"] = cls
        all_data.append(df_cls)
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    
    if normalize:
        for col in FEATURE_NAMES:
            min_val, max_val = df[col].min(), df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
    
    # Save
    Path(output_dir).mkdir(exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    filename = f"mechanical_dataset_{n_samples}_{file_id}.csv"
    filepath = Path(output_dir) / filename
    df.to_csv(filepath, index=False)
    
    return df, str(filepath)