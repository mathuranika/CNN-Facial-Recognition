import streamlit as st
import torch
from torchvision import transforms
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from PIL import Image

# Configuration
IMAGE_SIZE = 112
LABELS = ['AmitabhBachchan', 'HrithikRoshan', 'JayaBhaduri', 'Kajol', 'KareenaKapoor', 'SharukhKhan']
CONFIDENCE_THRESHOLD = 0.8

#Load TorchScript Model
@st.cache_resource
def load_model():
    model = torch.jit.load("face_recognition_model.pt", map_location='cpu')
    model.eval()
    return model

model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Face Prediction
def predict_face(face_img):
    try:
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)
        return LABELS[predicted.item()] if confidence.item() > CONFIDENCE_THRESHOLD else "Unknown"
    except Exception:
        return "Unknown"

# Streamlit Interface
st.title("🎬 Face Recognition & Tracking App")
st.markdown("Upload a video to detect and label faces using your trained model.")

uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video:
    st.video(uploaded_video)

    # Save the uploaded file temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())
    temp_input.close()

    output_path = temp_input.name.replace(".mp4", "_labeled.mp4")

    # Setup video processing
    cap = cv2.VideoCapture(temp_input.name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4)))
    )

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        st.info("Processing video, please wait...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bbox.xmin * iw)
                    y = int(bbox.ymin * ih)
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)

                    face_crop = frame[y:y+h, x:x+w]
                    label = predict_face(face_crop)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            out.write(frame)

    cap.release()
    out.release()

    st.success("✅ Done! Download your labeled video below.")
    with open(output_path, "rb") as f:
        st.download_button("📥 Download Labeled Video", f, file_name="labeled_output.mp4")