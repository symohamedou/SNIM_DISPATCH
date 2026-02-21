import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import io
import easyocr
import re
import os
import time

# --- 1. CONFIGURATION MOBILE ---
# On utilise la caméra native (0) pour le mode hors-ligne sur Android
LISTE_CAMERAS = {
    "Caméra Android": 0
}

# --- 2. INITIALISATION IA (TFLITE) ---
@st.cache_resource
def initialiser_ia():
    # On pointe sur le fichier que je vois dans ton dossier
    chemin_model = "best_float32.tflite" 
    model_yolo = YOLO(chemin_model, task='detect')
    reader_ocr = easyocr.Reader(['en'], gpu=False)
    return model_yolo, reader_ocr

try:
    model, reader = initialiser_ia()
except Exception as e:
    st.error(f"Erreur : Assure-toi que 'best_float32.tflite' est dans le même dossier que ce script.")

# --- 3. INTERFACE DISPATCHER (IDENTIQUE) ---
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
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# --- 4. LOGIQUE DE TRAITEMENT ---
def traiter_image(frame, site, poste, nom_camera):
    st.session_state.frame_count += 1
    maintenant = datetime.now()
    
    # On saute 3 images sur 4 pour que le téléphone reste fluide (CPU)
    if st.session_state.frame_count % 4 != 0:
        return [], frame

    # Utilisation du modèle TFLite (imgsz=640 car c'est ton format d'export)
    results = model.predict(frame, conf=seuil_conf, imgsz=640, verbose=False) 
    donnees = []
    
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            if label in ["vide", "riche", "sterile", "mixte"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    ocr_res = reader.readtext(gray, allowlist='0123456789')
                    num = "N/A"
                    for res in ocr_res:
                        match = re.search(r'\d{3}', res[1])
                        if match: num = match.group(); break
                    
                    if num != "N/A":
                        if num not in st.session_state.last_detections or \
                           maintenant > st.session_state.last_detections[num] + timedelta(minutes=5):
                            st.session_state.last_detections[num] = maintenant
                            donnees.append({
                                "Heure": maintenant.strftime("%H:%M:%S"),
                                "Camion": num, "Nature": label.upper(),
                                "Point": nom_camera, "Site": site, "Tonnage": 200
                            })
    return donnees, results[0].plot()

# --- 5. AFFICHAGE ---
st.title(f"📊 Supervision Mobile : {site_actuel}")
col_v, col_s = st.columns([2, 1])

with col_v:
    st_frame = st.empty()
    if st.button("🚀 LANCER LA CAMÉRA"):
        cap = cv2.VideoCapture(0) # Accès caméra Android
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            data, plot = traiter_image(frame, site_actuel, poste_actuel, "Android_Internal")
            if data: st.session_state.data_log.extend(data)
            
            st_frame.image(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        cap.release()

# --- 6. ANALYSE (IDENTIQUE) ---
if st.session_state.data_log:
    df = pd.DataFrame(st.session_state.data_log)
    with col_s:
        st.subheader("📈 Statistiques")
        st.metric("Tonnage", f"{df['Tonnage'].sum()} T")
        st.metric("Cycles", len(df))
    st.data_editor(df, use_container_width=True)