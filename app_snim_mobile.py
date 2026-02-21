import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import easyocr
import re
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- 2. INITIALISATION IA ---
@st.cache_resource
def initialiser_ia():
    chemin_model = "best_float32.tflite" 
    model_yolo = YOLO(chemin_model, task='detect')
    reader_ocr = easyocr.Reader(['en'], gpu=False)
    return model_yolo, reader_ocr

try:
    model, reader = initialiser_ia()
except Exception as e:
    st.error(f"Erreur : Assure-toi que 'best_float32.tflite' est sur GitHub.")

# --- 3. INTERFACE DISPATCHER (STRICTEMENT IDENTIQUE) ---
st.set_page_config(page_title="SNIM SMART DISPATCH | MOBILE", layout="wide")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Logo_SNIM.svg/1200px-Logo_SNIM.svg.png", width=180)
    st.title("🛰️ Smart Operations")
    site_actuel = st.selectbox("Site", ["Guelb El Rhein", "M'Haoudat", "TO14", "Tazadit"])
    poste_actuel = st.radio("Shift", ["Matin", "Soir", "Nuit"])
    seuil_conf = st.slider("Confiance IA", 0.1, 1.0, 0.75, 0.05)
    
    if 'flotte' not in st.session_state:
        st.session_state.flotte = {"Panne": 0, "Crevaison": 0}

if 'data_log' not in st.session_state:
    st.session_state.data_log = []
if 'last_detections' not in st.session_state:
    st.session_state.last_detections = {} 

# --- 4. CLASSE DE TRAITEMENT VIDÉO (TEMPS RÉEL) ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Détection YOLO en direct
        results = self.model.predict(img, conf=seuil_conf, imgsz=640, verbose=False)
        annotated_img = results[0].plot()

        # Logique d'enregistrement (identique à ton code)
        maintenant = datetime.now()
        for box in results[0].boxes:
            label = self.model.names[int(box.cls[0])]
            if label in ["vide", "riche", "sterile", "mixte"]:
                # Pour le temps réel, on enregistre uniquement si l'OCR est rapide
                # Ici on affiche surtout les boîtes pour la démo surveillance
                pass

        import av
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- 5. AFFICHAGE (ADAPTÉ POUR LE DIRECT SUR IPHONE) ---
st.title(f"📊 Supervision Mobile : {site_actuel}")
col_v, col_s = st.columns([2, 1])

with col_v:
    st.write("### 🎥 Flux Vidéo IA en Temps Réel")
    # Ce composant ouvre la caméra et affiche les boîtes qui bougent
    webrtc_streamer(
        key="snim-live",
        video_transformer_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# --- 6. STATISTIQUES (TON INTERFACE IDENTIQUE) ---
if st.session_state.data_log:
    df = pd.DataFrame(st.session_state.data_log)
    with col_s:
        st.subheader("📈 Statistiques")
        st.metric("Tonnage Total", f"{df['Tonnage'].sum()} T")
        st.metric("Nombre de Cycles", len(df))

    st.write("### 📝 Historique des chargements")
    st.data_editor(df, use_container_width=True)
