import streamlit as st
import numpy as np
import pandas as pd
import io
import uuid

st.set_page_config(
    page_title="Mechanical Dataset Generator",
    page_icon="⚙️",
    layout="wide"
)

# ---------- Data Generation Logic ----------
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

def generate_dataset(n_samples, classes, noise_level, normalize, seed):
    np.random.seed(seed)
    samples_per_class = n_samples // len(classes)
    all_data = []

    for cls in classes:
        params = CLASS_PARAMS[cls]
        data = {}
        for feat, (mean, std) in params.items():
            values = np.random.normal(mean, std, samples_per_class)
            values += np.random.normal(0, std * noise_level, samples_per_class)
            data[feat] = values
        df_cls = pd.DataFrame(data)
        df_cls["label"] = cls
        all_data.append(df_cls)

    df = pd.concat(all_data, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if normalize:
        for col in FEATURE_NAMES:
            min_val, max_val = df[col].min(), df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)

    return df

# ---------- UI ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a365d;
    }
    .stButton>button {
        background-color: #2b6cb0;
        color: white;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚙️ Synthetic Mechanical Dataset Generator</p>', unsafe_allow_html=True)
st.markdown("Generate realistic fault-detection datasets for predictive maintenance")

with st.sidebar:
    st.header("⚙️ Generation Settings")

    n_samples = st.slider("Total Samples", 1000, 100000, 10000, 1000)

    available_classes = ["NORMAL", "BEARING_WEAR", "OVERLOAD", "MISALIGNMENT", "IMBALANCE"]
    selected_classes = st.multiselect("Fault Classes", available_classes, default=available_classes)

    noise_level = st.slider("Sensor Noise Level", 0.0, 0.25, 0.05, 0.01)
    normalize = st.checkbox("Normalize features to [0-1]", value=True)
    seed = st.number_input("Random Seed", value=42, min_value=0)

    st.markdown("---")
    generate_btn = st.button("🚀 Generate Dataset", use_container_width=True)

if generate_btn:
    if not selected_classes:
        st.error("Please select at least one class.")
    else:
        with st.spinner("Generating realistic mechanical data..."):
            df = generate_dataset(
                n_samples=n_samples,
                classes=selected_classes,
                noise_level=noise_level,
                normalize=normalize,
                seed=seed
            )

            st.success("✅ Dataset generated successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", f"{len(df):,}")
            col2.metric("Features", "7 + Label")
            col3.metric("Classes", len(selected_classes))

            # Class distribution
            st.subheader("Class Distribution")
            dist = df["label"].value_counts().reset_index()
            dist.columns = ["Class", "Samples"]
            st.bar_chart(dist.set_index("Class"))

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            file_name = f"mechanical_dataset_{len(df)}_{str(uuid.uuid4())[:6]}.csv"

            st.download_button(
                label="📥 Download CSV Dataset",
                data=csv,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True
            )

            # Preview
            with st.expander("Preview first 10 rows"):
                st.dataframe(df.head(10), use_container_width=True)

else:
    st.info("👈 Configure settings in the sidebar and click **Generate Dataset**")

    st.markdown("### What you get")
    st.markdown("""
    - **7 realistic mechanical features**: Temperature, Pressure, Vibration, Load, Stress, Strain, Cycle Count  
    - **5 fault classes** with class-specific distributions  
    - **Sensor noise** simulation  
    - **Optional Min-Max normalization**  
    - Ready for machine learning (Random Forest, XGBoost, Neural Nets, etc.)
    """)