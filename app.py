"""
Aerial Object Detection — Streamlit App
Bird vs Drone Classifier + Optional YOLOv8 Detection
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import time

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aerial Object Detector",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #4C9BE8, #9B4CE8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 1rem;
    }
    .big-label { font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0; }
    .confidence { font-size: 1.1rem; color: #aaa; }
    .bird-color  { color: #4C9BE8; }
    .drone-color { color: #E8724C; }
    .metric-box {
        background: #1e1e2e; border-radius: 8px; padding: 1rem;
        text-align: center; border: 1px solid #333;
    }
    .stProgress > div > div { background: linear-gradient(90deg, #4C9BE8, #9B4CE8); }
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    import tensorflow as tf
    model_path = 'best_model.h5'
    if not Path(model_path).exists():
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
        return None

@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        weights = 'yolo_best.pt'
        if not Path(weights).exists():
            return None
        return YOLO(weights)
    except ImportError:
        return None
    except Exception:
        return None

@st.cache_data
def load_class_indices():
    if Path('class_indices.json').exists():
        with open('class_indices.json') as f:
            return json.load(f)
    return {'bird': 0, 'drone': 1}

# ── Preprocessing ──────────────────────────────────────────────────
def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert('RGB').resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Predict (classifier) ───────────────────────────────────────────
def predict_class(model, img, class_indices):
    arr   = preprocess_image(img)
    prob  = float(model.predict(arr, verbose=0)[0][0])
    idx2name = {v: k for k, v in class_indices.items()}
    pred_idx = int(prob > 0.5)
    pred_name = idx2name.get(pred_idx, 'unknown')
    conf = prob if pred_idx == class_indices.get('drone', 1) else 1 - prob
    return pred_name, conf, prob

# ── YOLOv8 Inference ───────────────────────────────────────────────
def run_yolo(yolo_model, img: Image.Image, conf_thresh=0.25):
    import numpy as np
    results = yolo_model.predict(np.array(img.convert('RGB')),
                                  conf=conf_thresh, verbose=False)
    return results[0]

def draw_detections(img: Image.Image, result, class_names=['bird', 'drone']):
    draw = ImageDraw.Draw(img)
    colors = {'bird': '#4C9BE8', 'drone': '#E8724C'}
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            color  = colors.get(label, '#FFFFFF')
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1, y1-22, x1+len(label)*9+50, y1], fill=color)
            draw.text((x1+4, y1-20), f"{label} {conf:.2f}", fill='white')
            detections.append({'label': label, 'conf': conf,
                                'bbox': [x1, y1, x2, y2]})
    return img, detections

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    use_yolo = st.toggle("Enable YOLOv8 Detection", value=False,
                          help="Draw bounding boxes around detected objects")

    if use_yolo:
        conf_thresh = st.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)
    else:
        conf_thresh = 0.25

    st.markdown("---")
    st.markdown("### 📊 Model Info")

    classifier = load_classifier()
    yolo_model = load_yolo()
    class_indices = load_class_indices()

    if classifier:
        st.success("✅ Classifier loaded")
        total_params = classifier.count_params()
        st.caption(f"Parameters: {total_params:,}")
    else:
        st.error("❌ best_model.h5 not found\nRun notebooks 02 & 03 first.")

    if use_yolo:
        if yolo_model:
            st.success("✅ YOLOv8 loaded")
        else:
            st.warning("⚠️ yolo_best.pt not found\nRun notebook 04 first.")

    st.markdown("---")
    st.markdown("### 📁 Classes")
    st.markdown("🐦 **Bird** — natural aerial object")
    st.markdown("🛸 **Drone** — unmanned aerial vehicle")

# ── Main UI ────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛸 Aerial Object Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Bird vs Drone classification with optional YOLOv8 detection</p>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🖼️ Single Image", "📂 Batch Analysis"])

# ── Tab 1: Single Image ────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Upload an aerial image",
                                  type=['jpg', 'jpeg', 'png', 'webp'],
                                  help="Upload a bird or drone image")

    if uploaded:
        img_original = Image.open(uploaded).convert('RGB')
        col_img, col_result = st.columns([1.2, 1])

        with col_img:
            st.subheader("Input Image")
            display_img = img_original.copy()

            if use_yolo and yolo_model:
                result   = run_yolo(yolo_model, img_original.copy(), conf_thresh)
                display_img, detections = draw_detections(
                    img_original.copy(), result)
            st.image(display_img, use_container_width=True)

        with col_result:
            st.subheader("Prediction")
            if classifier:
                t0 = time.time()
                pred_name, conf, raw_prob = predict_class(
                    classifier, img_original, class_indices)
                elapsed = (time.time() - t0) * 1000

                emoji = "🐦" if pred_name == "bird" else "🛸"
                color_cls = "bird-color" if pred_name == "bird" else "drone-color"

                st.markdown(f"""
                <div class="prediction-card">
                    <div style="font-size:3rem">{emoji}</div>
                    <div class="big-label {color_cls}">{pred_name.upper()}</div>
                    <div class="confidence">Confidence: {conf*100:.1f}%</div>
                    <div style="color:#666; font-size:0.85rem; margin-top:0.5rem">
                        Inference: {elapsed:.0f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Confidence Breakdown")
                bird_conf  = (1 - raw_prob) if class_indices.get('bird', 0) == 0 else raw_prob
                drone_conf = raw_prob if class_indices.get('drone', 1) == 1 else (1 - raw_prob)
                st.markdown("🐦 Bird")
                st.progress(float(bird_conf), text=f"{bird_conf*100:.1f}%")
                st.markdown("🛸 Drone")
                st.progress(float(drone_conf), text=f"{drone_conf*100:.1f}%")

                if use_yolo and yolo_model and detections:
                    st.markdown("#### 📦 Detected Objects")
                    for i, d in enumerate(detections):
                        st.markdown(
                            f"**{i+1}. {d['label'].upper()}** — "
                            f"conf: `{d['conf']:.3f}`  "
                            f"bbox: `[{d['bbox'][0]}, {d['bbox'][1]}, "
                            f"{d['bbox'][2]}, {d['bbox'][3]}]`"
                        )
            else:
                st.warning("Load best_model.h5 first by running notebooks 02 & 03.")

# ── Tab 2: Batch Analysis ──────────────────────────────────────────
with tab2:
    st.subheader("Batch Image Prediction")
    batch_files = st.file_uploader("Upload multiple images",
                                    type=['jpg', 'jpeg', 'png'],
                                    accept_multiple_files=True)

    if batch_files and classifier:
        if st.button("▶ Run Batch Prediction", type="primary"):
            results_data = []
            cols = st.columns(4)

            with st.spinner(f"Analyzing {len(batch_files)} images..."):
                for i, f in enumerate(batch_files):
                    img = Image.open(f).convert('RGB')
                    pred_name, conf, _ = predict_class(classifier, img, class_indices)
                    results_data.append({
                        'filename': f.name,
                        'prediction': pred_name,
                        'confidence': f"{conf*100:.1f}%"
                    })
                    with cols[i % 4]:
                        emoji = "🐦" if pred_name == "bird" else "🛸"
                        color = "#4C9BE8" if pred_name == "bird" else "#E8724C"
                        st.image(img, caption=f"{emoji} {pred_name} ({conf*100:.0f}%)",
                                  use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Batch Summary")
            import pandas as pd
            df = pd.DataFrame(results_data)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Images", len(df))
            c2.metric("Birds",  int((df['prediction'] == 'bird').sum()))
            c3.metric("Drones", int((df['prediction'] == 'drone').sum()))
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("⬇ Download Results CSV", csv,
                                "batch_results.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#555; font-size:0.85rem'>"
    "Aerial Object Detection · TensorFlow/Keras + YOLOv8 · Streamlit"
    "</center>",
    unsafe_allow_html=True
)
