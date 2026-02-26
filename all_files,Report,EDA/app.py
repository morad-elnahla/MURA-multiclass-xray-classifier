import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
import os

st.set_page_config(
    page_title="BONIFY — Fracture Detection AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0d12 !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"] {
    display: none !important;
}
.block-container {
    padding: 2.5rem 4rem !important;
    max-width: 1100px !important;
}
[data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(148,163,184,0.2) !important;
    border-radius: 10px !important;
    padding: 2.5rem !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #64748b !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(239,68,68,0.25) !important;
    color: rgba(239,68,68,0.6) !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    width: 100% !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(239,68,68,0.06) !important;
    color: #f87171 !important;
    border-color: rgba(239,68,68,0.45) !important;
}
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────
IM_SIZE = 320
BODY_PARTS = ['ELBOW','FINGER','FOREARM','HAND','HUMERUS','SHOULDER','WRIST']
CLASS_NAMES = []
for bp in BODY_PARTS:
    CLASS_NAMES.append(f'{bp}_Normal')
    CLASS_NAMES.append(f'{bp}_Abnormal')

PART_INFO = {
    'ELBOW':    {'icon': '🦾', 'label': 'Elbow',    'desc': 'Distal humerus, radius & ulna joint'},
    'FINGER':   {'icon': '☝️', 'label': 'Finger',   'desc': 'Phalanges — proximal, middle & distal'},
    'FOREARM':  {'icon': '💪', 'label': 'Forearm',  'desc': 'Radius and ulna shaft'},
    'HAND':     {'icon': '✋', 'label': 'Hand',     'desc': 'Metacarpals & carpal bones'},
    'HUMERUS':  {'icon': '🦴', 'label': 'Humerus',  'desc': 'Upper arm — proximal to distal'},
    'SHOULDER': {'icon': '🩻', 'label': 'Shoulder', 'desc': 'Glenohumeral joint & clavicle'},
    'WRIST':    {'icon': '⌚', 'label': 'Wrist',    'desc': 'Carpal bones & distal radius'},
}

FINDING_INFO = {
    True: {
        'title': 'No Fracture Detected',
        'sub': 'Bone structure appears within normal parameters.',
        'color': '#4ade80',
        'bg': 'rgba(34,197,94,0.07)',
        'border': 'rgba(34,197,94,0.2)',
        'badge': '✓  NORMAL',
    },
    False: {
        'title': 'Fracture / Abnormality Detected',
        'sub': 'Irregular bone pattern identified. Clinical review recommended.',
        'color': '#f87171',
        'bg': 'rgba(239,68,68,0.07)',
        'border': 'rgba(239,68,68,0.2)',
        'badge': '✕  ABNORMAL',
    }
}

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        return K.sum(weight * cross_entropy, axis=1)
    return focal_loss

@st.cache_resource
def load_model_cached():
    path = 'densenet169_multiclass_final.keras'
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(
        path, custom_objects={'categorical_focal_loss': categorical_focal_loss}
    )

def predict(model, img_rgb):
    img = cv2.resize(img_rgb, (IM_SIZE, IM_SIZE)) / 255.0
    probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    return probs, int(np.argmax(probs))

for k in ['result', 'img_rgb']:
    if k not in st.session_state:
        st.session_state[k] = None

# ── HEADER ────────────────────────────────────────────────
c1, c2 = st.columns([5, 2])
with c1:
    st.markdown("""
    <div style="margin-bottom:6px;">
        <span style="font-family:'Syne',sans-serif;font-size:34px;font-weight:800;color:#ffffff;letter-spacing:-0.02em;">
            🦴 BONIFY
        </span>
        <span style="font-size:11px;color:#334155;margin-left:14px;letter-spacing:0.18em;font-family:'DM Mono',monospace;
                     background:rgba(255,255,255,0.04);padding:3px 10px;border-radius:4px;border:1px solid rgba(255,255,255,0.07);">
            v1.0 BETA
        </span>
    </div>
    <div style="font-size:12px;color:#475569;font-family:'DM Mono',monospace;">
        AI-Powered Fracture &amp; Abnormality Detection &nbsp;·&nbsp; DenseNet169 &nbsp;·&nbsp; 7 Body Regions
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="text-align:right;padding-top:14px;">
        <span style="font-size:11px;color:#22d3ee;letter-spacing:0.18em;font-family:'DM Mono',monospace;
                     background:rgba(34,211,238,0.06);padding:5px 14px;border-radius:20px;border:1px solid rgba(34,211,238,0.15);">
            ● MODEL ONLINE
        </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

model = load_model_cached()
if model is None:
    st.error("Place `densenet169_multiclass_final.keras` in the same folder as app.py")
    st.stop()

# ── UPLOAD ────────────────────────────────────────────────
if st.session_state.result is None:
    st.markdown("""
    <div style="font-size:11px;color:#475569;letter-spacing:0.2em;margin-bottom:10px;font-family:'DM Mono',monospace;">
        UPLOAD X-RAY IMAGE
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop your X-ray here — PNG or JPG", type=['png','jpg','jpeg'])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        with st.spinner("Analyzing radiograph..."):
            probs, pred_idx = predict(model, img_rgb)
        st.session_state.result = (probs, pred_idx)
        st.session_state.img_rgb = img_rgb
        st.rerun()

# ── RESULTS ──────────────────────────────────────────────
if st.session_state.result is not None:
    probs, pred_idx = st.session_state.result
    img_rgb = st.session_state.img_rgb
    body_part, status = CLASS_NAMES[pred_idx].split('_')
    confidence = float(probs[pred_idx]) * 100
    is_normal = status == 'Normal'
    top5 = np.argsort(probs)[::-1][:5]
    info = FINDING_INFO[is_normal]
    part = PART_INFO[body_part]

    col1, col2 = st.columns([1, 1], gap="large")

    # ── LEFT: Image ───────────────────────────────────────
    with col1:
        st.markdown("""
        <div style="font-size:11px;color:#475569;letter-spacing:0.2em;margin-bottom:10px;font-family:'DM Mono',monospace;">
            RADIOGRAPH INPUT
        </div>
        """, unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

        # Body Part Info Card
        st.markdown(f"""
        <div style="margin-top:14px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                    border-radius:8px;padding:14px 18px;display:flex;align-items:center;gap:14px;">
            <div style="font-size:28px;">{part['icon']}</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:#e2e8f0;">
                    {part['label']}
                </div>
                <div style="font-size:11px;color:#475569;margin-top:2px;font-family:'DM Mono',monospace;">
                    {part['desc']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── RIGHT: Report ─────────────────────────────────────
    with col2:
        st.markdown("""
        <div style="font-size:11px;color:#475569;letter-spacing:0.2em;margin-bottom:12px;font-family:'DM Mono',monospace;">
            DIAGNOSTIC REPORT
        </div>
        """, unsafe_allow_html=True)

        # Verdict
        st.markdown(f"""
        <div style="background:{info['bg']};border:1px solid {info['border']};border-radius:10px;padding:20px 22px;margin-bottom:20px;">
            <div style="font-size:10px;color:{info['color']};letter-spacing:0.2em;margin-bottom:10px;font-family:'DM Mono',monospace;">
                {info['badge']}
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:{info['color']};line-height:1.2;margin-bottom:8px;">
                {info['title']}
            </div>
            <div style="font-size:12px;color:#64748b;font-family:'DM Mono',monospace;">
                {info['sub']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats Row
        conf_color = "#4ade80" if is_normal else "#f87171"
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:20px;">
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                        border-radius:8px;padding:14px;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:{conf_color};">{confidence:.0f}%</div>
                <div style="font-size:10px;color:#475569;margin-top:4px;letter-spacing:0.15em;font-family:'DM Mono',monospace;">CONFIDENCE</div>
            </div>
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                        border-radius:8px;padding:14px;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#e2e8f0;">{part['icon']}</div>
                <div style="font-size:10px;color:#475569;margin-top:4px;letter-spacing:0.15em;font-family:'DM Mono',monospace;">{part['label'].upper()}</div>
            </div>
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                        border-radius:8px;padding:14px;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#e2e8f0;">14</div>
                <div style="font-size:10px;color:#475569;margin-top:4px;letter-spacing:0.15em;font-family:'DM Mono',monospace;">CLASSES</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Bar
        st.markdown(f"""
        <div style="margin-bottom:20px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="font-size:10px;color:#475569;letter-spacing:0.2em;font-family:'DM Mono',monospace;">MODEL CONFIDENCE</span>
                <span style="font-size:12px;color:#fff;font-family:'DM Mono',monospace;">{confidence:.1f}%</span>
            </div>
            <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:2px;">
                <div style="height:100%;width:{confidence:.1f}%;background:{conf_color};border-radius:2px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 5
        st.markdown("""
        <div style="font-size:10px;color:#475569;letter-spacing:0.2em;margin-bottom:12px;font-family:'DM Mono',monospace;">
            PROBABILITY DISTRIBUTION — TOP 5
        </div>
        """, unsafe_allow_html=True)

        top5_html = ""
        for rank, i in enumerate(top5):
            bp_t, st_t = CLASS_NAMES[i].split('_')
            pct = float(probs[i]) * 100
            is_top = rank == 0
            row_color = "#e2e8f0" if is_top else "#475569"
            bar_color = ("#4ade80" if st_t == "Normal" else "#f87171") if is_top else "rgba(255,255,255,0.08)"
            dot_color = "#4ade80" if st_t == "Normal" else "#f87171"
            top5_html += f"""
            <div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                <div style="font-size:10px;color:#1e293b;width:12px;font-family:'DM Mono',monospace;">{rank+1}</div>
                <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};flex-shrink:0;"></div>
                <div style="font-size:11px;color:{row_color};width:150px;flex-shrink:0;font-family:'DM Mono',monospace;">
                    {PART_INFO[bp_t]['label']} · {st_t}
                </div>
                <div style="flex:1;height:2px;background:rgba(255,255,255,0.05);border-radius:1px;">
                    <div style="height:100%;width:{min(pct,100):.1f}%;background:{bar_color};border-radius:1px;"></div>
                </div>
                <div style="font-size:11px;color:{row_color};width:42px;text-align:right;font-family:'DM Mono',monospace;">{pct:.1f}%</div>
            </div>
            """
        st.markdown(top5_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div style="background:rgba(251,191,36,0.05);border:1px solid rgba(251,191,36,0.15);border-radius:6px;
                    padding:10px 14px;margin-bottom:14px;">
            <div style="font-size:10px;color:#92400e;letter-spacing:0.1em;font-family:'DM Mono',monospace;">
                ⚠️ &nbsp; FOR RESEARCH PURPOSES ONLY — NOT A MEDICAL DIAGNOSIS
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✕   Clear & Upload New Scan"):
            st.session_state.result = None
            st.session_state.img_rgb = None
            st.rerun()
