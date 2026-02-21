import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import easyocr
import re

# --- 1. CONFIGURATION RESPONSIVE ---
st.set_page_config(
    page_title="SNIM SMART DISPATCH | MOBILE", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS pour que l'interface s'adapte parfaitement à l'écran de l'iPhone
st.markdown("""
    <style>
    .main { padding: 0rem !important; }
    [data-testid="column"] { width: 100% !important; min-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALISATION IA (TFLITE) ---
@st.cache_resource
def initialiser_ia():
    # Utilise ton fichier best_float32.tflite
    model_yolo = YOLO("best_float32.tflite", task='detect')
    reader_ocr = easyocr.Reader(['en'], gpu=False)
    return model_yolo, reader_ocr

try:
    model, reader = initialiser_ia()
except Exception as e:
    st.error("⚠️ Erreur : Le fichier 'best_float32.tflite' est introuvable sur GitHub.")

# --- 3. INTERFACE DISPATCHER (CONSERVÉE) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Logo_SNIM.svg/1200px-Logo_SNIM.svg.png", width=150)
    st.title("🛰️ Smart Ops")
    site_actuel = st.selectbox("Site SNIM", ["Guelb El Rhein", "M'Haoudat", "TO14", "Tazadit"])
    poste_actuel = st.radio("Shift", ["Matin", "Soir", "Nuit"])
    seuil_conf = st.slider("Confiance IA", 0.1, 1.0, 0.75, 0.05)

if 'data_log' not in st.session_state:
    st.session_state.data_log = []
if 'last_detections' not in st.session_state:
    st.session_state.last_detections = {} 

# --- 4. LOGIQUE DE TRAITEMENT IA ---
def traiter_image(frame, site):
    maintenant = datetime.now()
    results = model.predict(frame, conf=seuil_conf, imgsz=640, verbose=False) 
    donnees = []
    
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            if label in ["vide", "riche", "sterile", "mixte"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # OCR pour identifier les 3 chiffres du camion
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    ocr_res = reader.readtext(gray, allowlist='0123456789')
                    num = "N/A"
                    for res in ocr_res:
                        match = re.search(r'\d{3}', res[1])
                        if match: 
                            num = match.group()
                            break
                    
                    if num != "N/A":
                        # Anti-doublon (évite de compter 2 fois le même camion en 5 min)
                        if num not in st.session_state.last_detections or \
                           maintenant > st.session_state.last_detections[num] + timedelta(minutes=5):
                            st.session_state.last_detections[num] = maintenant
                            donnees.append({
                                "Heure": maintenant.strftime("%H:%M:%S"),
                                "Camion": num, 
                                "Nature": label.upper(),
                                "Site": site, 
                                "Tonnage": 200
                            })
    return donnees, results[0].plot()

# --- 5. AFFICHAGE DYNAMIQUE (RESPONSIVE) ---
st.title(f"📊 Supervision : {site_actuel}")

# Les colonnes s'empilent verticalement sur iPhone grâce au CSS ajouté en haut
col_v, col_s = st.columns([1, 1])

with col_v:
    st.subheader("📸 Caméra Dispatch")
    # camera_input est le seul moyen d'activer la caméra iPhone sur Streamlit Web
    img_input = st.camera_input("Scanner le circuit de mine")

    if img_input:
        file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # L'IA travaille et génère le suivi visuel (boîtes)
        data, plot = traiter_image(frame, site_actuel)
        
        if data:
            st.session_state.data_log.extend(data)
            st.success(f"✅ Camion {data[0]['Camion']} détecté !")
        
        # Affiche l'image avec les boîtes (Responsive)
        st.image(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB), use_container_width=True)

with col_s:
    st.subheader("📈 Rapport de Production")
    if st.session_state.data_log:
        df = pd.DataFrame(st.session_state.data_log)
        
        # Métriques côte à côte
        m1, m2 = st.columns(2)
        m1.metric("Tonnage Total", f"{df['Tonnage'].sum()} T")
        m2.metric("Total Cycles", len(df))
        
        st.write("### 📝 Historique")
        st.dataframe(df.iloc[::-1], use_container_width=True)
