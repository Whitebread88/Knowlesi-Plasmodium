import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import joblib
from my_gcn_model import GCN
import pandas as pd
from collections import defaultdict

# === Streamlit Setup ===
st.set_page_config(layout="wide")
st.title("🧬 YOLOv8 → ResNet → GCN Stage Classifier")

st.sidebar.header("Prediction Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
    help="Only predictions with confidence >= threshold will be shown"
)

# === Load Models ===
@st.cache_resource
def load_models():
    yolo = YOLO("best.pt")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    gcn = GCN(input_dim=2048, hidden_dim=32, output_dim=4)
    gcn.load_state_dict(torch.load("gcn_model.pt", map_location="cpu"))
    gcn.eval()

    label_encoder = joblib.load("label_encoder.pkl")

    return yolo, resnet, gcn, label_encoder

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# === Utility Functions ===
def extract_features(cropped_img, resnet):
    tensor = transform(cropped_img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(tensor)
    return features.view(1, -1)

def create_edge_index():
    return torch.empty((2, 0), dtype=torch.long)

def predict_stage(feature_tensor, gcn_model):
    edge_index = create_edge_index()
    with torch.no_grad():
        out = gcn_model(feature_tensor, edge_index)
        probs = torch.exp(out)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item(), probs.squeeze().numpy()

def yolo_box_to_pixel_coords(box, img_w, img_h):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return x1, y1, x2, y2

# === Main App ===
yolo_model, resnet_model, gcn_model, label_encoder = load_models()

uploaded_file = st.file_uploader("Upload an RBC image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_width, img_height = image.size
    st.image(image.resize((img_width // 2, img_height // 2)), caption="Original Uploaded Image")

    results = yolo_model.predict(image, conf=0.3, verbose=False)

    if len(results) == 0 or results[0].boxes is None:
        st.warning("No objects detected.")
    else:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        features = []
        stage_preds = []
        confidences = []
        pred_indices = []
        all_class_probs = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = yolo_box_to_pixel_coords(box, img_width, img_height)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            feature_tensor = extract_features(crop, resnet_model)
            pred_idx, conf, class_probs = predict_stage(feature_tensor, gcn_model)

            label = label_encoder.inverse_transform([pred_idx])[0]
            features.append(feature_tensor.numpy().flatten())
            stage_preds.append(label)
            confidences.append(conf)
            pred_indices.append(pred_idx)
            all_class_probs.append(class_probs)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), f"{label} ({conf:.2f})", fill="red")

        st.image(annotated_image.resize((img_width // 2, img_height // 2)), caption="YOLO + GCN Output")

        st.subheader("🧪 Prediction Summary")
        for i, (label, conf, probs) in enumerate(zip(stage_preds, confidences, all_class_probs)):
            st.markdown(f"**Object {i+1}:** `{label}` with confidence `{conf:.2f}`")

            with st.expander("🔍 View all class confidences"):
                df = pd.DataFrame({
                    "Stage": label_encoder.inverse_transform(np.arange(len(probs))),
                    "Confidence": [f"{p:.4f}" for p in probs]
                })
                st.dataframe(df)

        # Threshold filtering + class summary
        class_counts = defaultdict(int)
        class_kept = defaultdict(int)
        class_confidences = defaultdict(list)
        final_rows = []

        draw = ImageDraw.Draw(annotated_image)

        for i, (box, pred_idx, conf) in enumerate(zip(boxes, pred_indices, confidences)):
            label = label_encoder.inverse_transform([pred_idx])[0]
            class_counts[label] += 1
            class_confidences[label].append(conf)

            if conf >= confidence_threshold:
                class_kept[label] += 1
                x1, y1, x2, y2 = yolo_box_to_pixel_coords(box, img_width, img_height)
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                final_rows.append({"Object": i+1, "Stage": label, "Confidence": round(conf, 2)})

        st.image(annotated_image.resize((img_width // 2, img_height // 2)), caption="Filtered Predictions")

        if final_rows:
            st.subheader("📋 Predictions Above Threshold")
            st.dataframe(pd.DataFrame(final_rows))
        else:
            st.info("No predictions above confidence threshold.")

        st.subheader("📈 Per-Class Prediction Metrics")
        summary_data = []
        for label in class_counts:
            total = class_counts[label]
            kept = class_kept[label]
            avg_conf = np.mean(class_confidences[label]) if class_confidences[label] else 0
            summary_data.append({
                "Stage": label,
                "Total Detections": total,
                "Above Threshold": kept,
                "Avg Confidence": round(avg_conf, 3)
            })

        if summary_data:
            st.dataframe(pd.DataFrame(summary_data))
        else:
            st.info("No detections to summarize.")
