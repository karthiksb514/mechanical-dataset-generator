import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(
    page_title="Mechanical Dataset Generator",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
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

API_URL = "http://localhost:8000"

if generate_btn:
    if not selected_classes:
        st.error("Please select at least one class.")
    else:
        with st.spinner("Generating realistic mechanical data..."):
            payload = {
                "n_samples": n_samples,
                "classes": selected_classes,
                "noise_level": noise_level,
                "normalize": normalize,
                "seed": seed
            }
            try:
                resp = requests.post(f"{API_URL}/generate", json=payload, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                
                st.success(f"✅ {result['message']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Samples", f"{result['total_samples']:,}")
                col2.metric("File Size", f"{result['file_size_mb']} MB")
                col3.metric("Classes", len(result['classes']))
                
                st.subheader("Class Distribution")
                dist_df = pd.DataFrame(list(result["classes"].items()), columns=["Class", "Samples"])
                st.bar_chart(dist_df.set_index("Class"))
                
                download_url = f"{API_URL}{result['download_url']}"
                file_resp = requests.get(download_url)
                
                st.download_button(
                    label="📥 Download CSV Dataset",
                    data=file_resp.content,
                    file_name=result["download_url"].split("/")[-1],
                    mime="text/csv",
                    use_container_width=True
                )
                
                with st.expander("Preview first 10 rows"):
                    df_preview = pd.read_csv(io.BytesIO(file_resp.content))
                    st.dataframe(df_preview.head(10), use_container_width=True)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("👈 Configure settings in the sidebar and click **Generate Dataset**")