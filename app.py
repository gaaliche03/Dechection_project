import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime
import base64
from streamlit_option_menu import option_menu
from ultralytics import YOLO

st.set_page_config(
    page_title="Déchection — Tri intelligent des déchets",
    page_icon="logo.png",
    layout="wide",
)

####################################css###############################
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] { background: #f1f5f9; padding-top: 1rem; }
[data-testid="stSidebar"] * { color: #0f172a !important; }
[data-testid="stSidebar"] hr { border-color: #e2e8f0 !important; }

.sidebar-section {
    font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: #475569 !important; margin: 1rem 0 0.4rem;
}
.stat-mini {
    background: #ffffff; border-radius: 10px; padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem; display: flex; justify-content: space-between; font-size: 0.82rem;
}
.stat-mini-val { font-size: 1rem; font-weight: 700; }

.engine-card {
    background: #ffffff; border: 1.5px solid #e2e8f0;
    border-radius: 12px; padding: 0.75rem 1rem; margin-top: 0.5rem;
    font-size: 0.83rem; color: #334155; line-height: 1.55;
}
.engine-card strong { color: #0f172a; font-size: 0.88rem; }
.engine-card-yolo { border-left: 4px solid #f59e0b; }

.page-header { padding: 0.4rem 0 1.2rem; border-bottom: 1px solid #f1f5f9; margin-bottom: 1.4rem; }
.page-title { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin: 0; }
.page-desc { color: #64748b; font-size: 0.88rem; margin-top: 0.25rem; }

.badge-recyclable {
    display: inline-block; background: #d1fae5; color: #065f46;
    font-weight: 600; font-size: 0.88rem; padding: 0.3rem 1rem;
    border-radius: 999px; border: 1.5px solid #6ee7b7;
}
.badge-non-recyclable {
    display: inline-block; background: #fee2e2; color: #991b1b;
    font-weight: 600; font-size: 0.88rem; padding: 0.3rem 1rem;
    border-radius: 999px; border: 1.5px solid #fca5a5;
}
.result-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 16px; padding: 1.3rem 1.5rem; margin-top: 0.7rem;
}
.result-class { font-size: 1.5rem; font-weight: 700; color: #0f172a; text-transform: capitalize; }
.result-conf { color: #64748b; font-size: 0.87rem; margin-top: 0.2rem; }
.result-tip {
    background: #fffbeb; border-left: 3px solid #f59e0b;
    padding: 0.5rem 0.85rem; border-radius: 0 8px 8px 0;
    font-size: 0.83rem; color: #78350f; margin-top: 0.8rem;
}
.low-conf-alert {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-radius: 8px; padding: 0.5rem 0.9rem;
    font-size: 0.8rem; color: #9a3412; margin-top: 0.5rem;
}
.no-det-alert {
    background: #f1f5f9; border: 1px solid #cbd5e1;
    border-radius: 8px; padding: 0.6rem 0.9rem;
    font-size: 0.85rem; color: #475569; margin-top: 0.5rem;
    text-align: center;
}
.hist-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.42rem 0; border-bottom: 1px solid #f1f5f9;
    font-size: 0.86rem; color: #374151;
}
.hist-dot-rec    { width:9px; height:9px; border-radius:50%; background:#10b981; flex-shrink:0; }
.hist-dot-nonrec { width:9px; height:9px; border-radius:50%; background:#ef4444; flex-shrink:0; }
.yolo-card {
    background: #fefce8; border: 1px solid #fde68a;
    border-radius: 12px; padding: 0.85rem 1rem; margin-top: 0.6rem;
}
.yolo-obj-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.28rem 0; border-bottom: 1px solid #fef9c3; font-size: 0.84rem;
}
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.1rem 1.2rem; text-align: center;
}
.metric-val { font-size: 1.9rem; font-weight: 700; color: #0f172a; }
.metric-label { font-size: 0.78rem; color: #94a3b8; margin-top: 0.18rem; }
.model-tag-yolo {
    display:inline-block; background:#fef3c7; color:#92400e;
    font-size:0.72rem; font-weight:600; padding:2px 9px; border-radius:999px;
}

.how-section {
    display: flex; gap: 1rem; align-items: flex-start;
    padding: 0.85rem 0; border-bottom: 1px solid #f1f5f9;
}
.how-icon { font-size: 1.6rem; min-width: 2.2rem; text-align: center; margin-top: 0.1rem; }
.how-title { font-weight: 600; font-size: 0.95rem; color: #0f172a; margin-bottom: 0.15rem; }
.how-desc  { font-size: 0.84rem; color: #64748b; line-height: 1.55; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; margin-top: 1rem; }
.info-card { border-radius: 10px; padding: 0.75rem 0.95rem; font-size: 0.82rem; line-height: 1.5; }
.info-card-green  { background:#f0fdf4; border:1px solid #bbf7d0; color:#14532d; }
.info-card-red    { background:#fef2f2; border:1px solid #fecaca; color:#7f1d1d; }
.info-card-yellow { background:#fffbeb; border:1px solid #fde68a; color:#78350f; }
.info-card-blue   { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a5f; }
.info-card-title  { font-weight: 700; margin-bottom: 0.3rem; font-size: 0.86rem; }

.data-section-title {
    font-size: 1rem; font-weight: 600; color: #0f172a;
    margin: 1.6rem 0 0.7rem; padding-bottom: 0.35rem;
    border-bottom: 2px solid #f1f5f9;
}
</style>
""", unsafe_allow_html=True)


#####################  SUPERCATÉGORIES TACO  ################################

#classes détectées par model best.pt
SUPERCAT_RECYCLABLE_MAP = {
    "Bottle":               True,
    "Bottle cap":           True,
    "Can":                  True,
    "Carton":               True,
    "Cup":                  False,
    "Glass jar":            True,
    "Lid":                  True,
    "Metal":                True,
    "Paper":                True,
    "Plastic container":    True,
    "Pop tab":              True,
    "Scrap metal":          True,
    "Aluminium foil":       False,
    "Battery":              False,
    "Blister pack":         False,
    "Broken glass":         False,
    "Cigarette":            False,
    "Crisp packet":         False,
    "Diaper":               False,
    "Drink can":            True,
    "Food waste":           False,
    "Plastic bag & wrapper": False,
    "Plastic glooves":      False,
    "Plastic utensils":     False,
    "Rope & strings":       False,
    "Shoe":                 False,
    "Squeezable tube":      False,
    "Straw":                False,
    "Styrofoam piece":      False,
    "Unlabeled litter":     False,
}

SUPERCAT_TIPS = {
    "Bottle":               "Poubelle jaune — videz et rincez avant de déposer.",
    "Bottle cap":           "Poubelle jaune — les bouchons se recyclent séparément dans certaines communes.",
    "Can":                  "Poubelle jaune — canettes aluminium / fer-blanc, écrasez-les.",
    "Carton":               "Poubelle jaune — briques alimentaires et cartons aplatis.",
    "Cup":                  "Poubelle noire — les gobelets plastique ne sont pas recyclables en France.",
    "Glass jar":            "Conteneur à verre — jamais dans la poubelle jaune.",
    "Lid":                  "Poubelle jaune — couvercles métal ou plastique rigide.",
    "Metal":                "Poubelle jaune — ferraille et aluminium.",
    "Paper":                "Poubelle jaune — papiers, journaux, magazines.",
    "Plastic container":    "Poubelle jaune — barquettes rigides (vérifiez le logo ♻️).",
    "Pop tab":              "Poubelle jaune — languettes aluminium.",
    "Scrap metal":          "Poubelle jaune ou déchetterie selon la taille.",
    "Aluminium foil":       "Poubelle noire — papier alu souillé non recyclable.",
    "Battery":              "Point de collecte en magasin (obligation légale).",
    "Blister pack":         "Poubelle noire — plaquettes médicaments multicouches.",
    "Broken glass":         "Poubelle noire — verre brisé dangereux, enveloppez-le.",
    "Cigarette":            "Poubelle noire — mégots toxiques, jamais dans la rue.",
    "Crisp packet":         "Poubelle noire — sachets chips (alu + plastique).",
    "Diaper":               "Poubelle noire — couches non recyclables.",
    "Drink can":            "Poubelle jaune — canettes boisson recyclables.",
    "Food waste":           "Composteur ou bac de collecte organique.",
    "Plastic bag & wrapper": "Poubelle noire — sacs et films plastique non recyclables.",
    "Plastic glooves":      "Poubelle noire — gants plastique.",
    "Plastic utensils":     "Poubelle noire — couverts plastique.",
    "Rope & strings":       "Poubelle noire — cordes et ficelles.",
    "Shoe":                 "Borne textile / ressourcerie ou déchetterie.",
    "Squeezable tube":      "Poubelle noire — tubes multicouches.",
    "Straw":                "Poubelle noire — pailles plastique.",
    "Styrofoam piece":      "Déchetterie — polystyrène non recyclable bac jaune.",
    "Unlabeled litter":     "Poubelle noire — déchet non identifié, sécurité = non recyclable.",
}

SUPERCAT_EMOJIS = {
    "Bottle": "🍶", "Bottle cap": "🔩", "Can": "🥫", "Carton": "📦",
    "Cup": "☕", "Glass jar": "🫙", "Lid": "🔘", "Metal": "🔧",
    "Paper": "📰", "Plastic container": "🥡", "Pop tab": "🔗", "Scrap metal": "⚙️",
    "Aluminium foil": "🥈", "Battery": "🔋", "Blister pack": "💊",
    "Broken glass": "💔", "Cigarette": "🚬", "Crisp packet": "🍟",
    "Diaper": "🧷", "Drink can": "🥤", "Food waste": "🍂",
    "Plastic bag & wrapper": "🛍️", "Plastic glooves": "🧤", "Plastic utensils": "🍴",
    "Rope & strings": "🧵", "Shoe": "👟", "Squeezable tube": "🪥",
    "Straw": "🥤", "Styrofoam piece": "📦", "Unlabeled litter": "❓",
}

POUBELLE_COLOR = {
    "Bottle":"🟡","Bottle cap":"🟡","Can":"🟡","Carton":"🟡","Cup":"⚫",
    "Glass jar":"⚪","Lid":"🟡","Metal":"🟡","Paper":"🟡","Plastic container":"🟡",
    "Pop tab":"🟡","Scrap metal":"🟡","Aluminium foil":"⚫","Battery":"🔴",
    "Blister pack":"⚫","Broken glass":"⚫","Cigarette":"⚫","Crisp packet":"⚫",
    "Diaper":"⚫","Drink can":"🟡","Food waste":"🟤","Plastic bag & wrapper":"⚫",
    "Plastic glooves":"⚫","Plastic utensils":"⚫","Rope & strings":"⚫",
    "Shoe":"🟣","Squeezable tube":"⚫","Straw":"⚫","Styrofoam piece":"⚫",
    "Unlabeled litter":"⚫",
}

#########################  MODÈLE YOLO  #########################################

YOLO_MODEL_PATH = "best.pt"  

@st.cache_resource
def load_yolo_model():
    model = YOLO(YOLO_MODEL_PATH)
    return model


def predict_yolo(model, img: Image.Image, conf_thresh: float = 0.25):
    results = model.predict(source=img, conf=conf_thresh,iou=0.45,verbose=False,)
    result = results[0]

    annotated_img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_img)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, f"class_{cls_id}")
        score= float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        recyclable = SUPERCAT_RECYCLABLE_MAP.get(cls_name, False)
        color= "#10b981" if recyclable else "#ef4444"
        rgb = (16, 185, 129) if recyclable else (239, 68, 68)

        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)
        label = f"{cls_name} {score*100:.0f}%"
        # Fond du label
        tw, th = draw.textlength(label), 14
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=rgb)
        draw.text((x1 + 3, y1 - th - 2), label, fill="white")

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "class": cls_name,
            "score":score,
            "recyclable": recyclable,
        })

    return detections, annotated_img


def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


############  SESSION STATE  ##########################
for k, v in {"history": [], "total_analysed": 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

######################  SIDEBAR  #############
with st.sidebar:
    img_b64 = get_image_base64("assets/logo.png")
    st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:0.5rem;">
            <img src="data:image/png;base64,{img_b64}" width="130"/>
        </div>
        """, unsafe_allow_html=True)

    page = option_menu(
        None,
        ["Analyse", "Historique"],
        icons=["search", "bar-chart-line"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "menu-icon": {"color": "#10b981", "font-size": "14px"},
            "icon": {"color": "#10b981", "font-size": "15px"},
            "nav-link": {
                "font-size": "0.92rem", "text-align": "left",
                "margin": "2px 0", "border-radius": "8px",
                "--hover-color": "#e2e8f0", "color": "#0f172a",
            },
            "nav-link-selected": { "background-color": "#475569", "color": "#ffffff", "font-weight": "600"},
        },
    )
    

    conf_threshold = 50

    if page.startswith("Analyse"):
        st.markdown('<div class="sidebar-section">⚙️ Paramètres YOLO</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="engine-card engine-card-yolo">
            <strong>⚡ YOLO SuperCat — Détection multi-objets</strong><br>
            Détecte les déchets TACO (30 supercatégories) avec bounding boxes.
            Modèle entraîné sur le dataset TACO avec regroupement par supercatégorie.
        </div>
        """, unsafe_allow_html=True)

        conf_threshold = st.slider(
            "Seuil de confiance (%)",
            min_value=10, max_value=95, value=25, step=5,
            help="Détections en-dessous de ce seuil sont filtrées.",
        )
        

    hist_s   = st.session_state.history
    n_tot_s  = len(hist_s)
    n_rec_s  = sum(1 for h in hist_s if h["recyclable"])
    n_nrec_s = n_tot_s - n_rec_s

    st.markdown('<div class="sidebar-section">Session en cours</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-mini">
        <span>Analysées</span>
        <span class="stat-mini-val" style="color:#34d399 !important">{n_tot_s}</span>
    </div>
    <div class="stat-mini">
        <span>♻️ Recyclables</span>
        <span class="stat-mini-val" style="color:#34d399 !important">{n_rec_s}</span>
    </div>
    <div class="stat-mini">
        <span>🚫 Non recyclables</span>
        <span class="stat-mini-val" style="color:#f87171 !important">{n_nrec_s}</span>
    </div>
    """, unsafe_allow_html=True)


##################  PAGE ANALYSE  #######################
if page.startswith("Analyse"):
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🔍 Analyse d'image</div>
        <div class="page-desc">Uploadez une ou plusieurs photos de déchets — détection par supercatégories TACO.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("**♻️ Comment utiliser Déchection ?**", expanded=False):
        st.markdown("""
        <div class="how-section">
            <div class="how-icon">📸</div>
            <div>
                <div class="how-title">1. Uploadez votre image</div>
                <div class="how-desc">Glissez-déposez ou cliquez pour sélectionner une photo (JPG, PNG, WEBP). Plusieurs images acceptées.</div>
            </div>
        </div>
        <div class="how-section">
            <div class="how-icon">⚡</div>
            <div>
                <div class="how-title">2. Détection YOLO SuperCat</div>
                <div class="how-desc">
                    Le modèle détecte automatiquement les déchets et les classe dans l'une des <strong>30 supercatégories TACO</strong>.
                    Ajustez le seuil de confiance dans la barre latérale.
                </div>
            </div>
        </div>
        <div class="how-section">
            <div class="how-icon">📊</div>
            <div>
                <div class="how-title">3. Lisez les résultats</div>
                <div class="how-desc">
                    L'image annotée avec les bounding boxes, la classe, le score et la consigne de tri s'affichent immédiatement.
                </div>
            </div>
        </div>
        <div class="how-section" style="border-bottom:none;padding-bottom:0;">
            <div class="how-icon">📁</div>
            <div>
                <div class="how-title">4. Explorez vos données</div>
                <div class="how-desc"><strong>Historique</strong> — visualisez vos statistiques et exportez en CSV.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown(
            "<p style='text-align:center;color:#94a3b8;margin-top:0.8rem;font-size:0.95rem'>"
            "⬆️ Glissez une ou plusieurs images ici pour commencer</p>",
            unsafe_allow_html=True,
        )
        st.stop()

    # chargement du model######################################
    with st.spinner("Chargement du modèle YOLO…"):
        model = load_yolo_model()

    conf_thresh_float = conf_threshold / 100.0

    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")

        with st.spinner(f"Analyse de {uploaded_file.name}…"):
            detections, annotated_img = predict_yolo(model, img, conf_thresh=conf_thresh_float)

        col_img, col_res = st.columns([1, 1.3], gap="large")

        with col_img:
            st.image(annotated_img, use_container_width=True,
                     caption=f"Détections — {len(detections)} objet(s)")

        with col_res:
            if not detections:
                st.markdown(
                    '<div class="no-det-alert">🔍 Aucun objet détecté au-dessus du seuil.<br>'
                    'Essayez de baisser le seuil de confiance dans la barre latérale.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**{len(detections)} objet(s) détecté(s)**")
                st.markdown('<div class="yolo-card">', unsafe_allow_html=True)
                for i, det in enumerate(detections):
                    cls  = det["class"]
                    score = det["score"] * 100
                    recyclable = det["recyclable"]
                    tip  = SUPERCAT_TIPS.get(cls, "")
                    badge = ('<span class="badge-recyclable" style="font-size:0.7rem;padding:2px 8px">✔ Recyclable</span>'
                             if recyclable else
                             '<span class="badge-non-recyclable" style="font-size:0.7rem;padding:2px 8px">✘ Non recyclable</span>')
                    alert = (f'<div class="low-conf-alert">⚠️ Confiance faible ({score:.0f}%) — résultat incertain.</div>'
                             if score < conf_threshold else "")
                    emoji = SUPERCAT_EMOJIS.get(cls, "🗑️")
                    poubelle = POUBELLE_COLOR.get(cls, "⚫")
                    st.markdown(f"""
                    <div class="yolo-obj-row">
                        <span>{emoji} <strong>{cls}</strong></span>
                        <span style="color:#64748b;font-size:0.79rem">{score:.0f}% conf.</span>
                        <span>{poubelle}</span>
                        {badge}
                    </div>{alert}
                    """, unsafe_allow_html=True)

                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "name": f"{uploaded_file.name} [obj {i+1}]",
                        "class": cls, "confidence": score,
                        "recyclable": recyclable, "model": "YOLO-SuperCat",
                    })
                st.markdown('</div>', unsafe_allow_html=True)

                tip0 = SUPERCAT_TIPS.get(detections[0]["class"], "")
                if tip0:
                    st.markdown(f'<div class="result-tip">💡 {tip0}</div>', unsafe_allow_html=True)

        # graphe de conf########################################
        if detections:
            yc      = [d["class"] for d in detections]
            ys      = [d["score"] * 100 for d in detections]
            ycolors = ["#10b981" if d["recyclable"] else "#ef4444" for d in detections]
            fig = go.Figure(go.Bar(
                x=ys,
                y=[f"{SUPERCAT_EMOJIS.get(c,'🗑️')} {c}" for c in yc],
                orientation="h", marker_color=ycolors,
                text=[f"{s:.0f}%" for s in ys], textposition="outside",
            ))
            fig.add_vline(x=conf_threshold, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"seuil {conf_threshold}%", annotation_font_size=10)
            fig.update_layout(
                title=dict(text="Scores de confiance YOLO", font=dict(size=12), x=0),
                xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=50, t=36, b=10),
                height=max(130, len(detections) * 35  + 50),
                font=dict(family="DM Sans, sans-serif", size=12), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()


###################################################  PAGE HISTORIQUE  ###############################
elif page.startswith("Historique"):
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📊 Données de session</div>
        <div class="page-desc">Statistiques et historique complet de toutes vos analyses.</div>
    </div>
    """, unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.info("Aucune analyse pour l'instant — rendez-vous dans **🔍 Analyse** pour commencer.")
        st.stop()

    df = pd.DataFrame(hist)
    n_total  = len(df)
    n_rec    = int(df["recyclable"].sum())
    n_nonrec = n_total - n_rec
    avg_conf = df["confidence"].mean()

    st.markdown('<div class="data-section-title">Vue d\'ensemble</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, n_total,            "Objets détectés"),
        (c2, n_rec,              "♻️ Recyclables"),
        (c3, n_nonrec,           "🚫 Non recyclables"),
        (c4, f"{avg_conf:.1f}%", "Confiance moyenne"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="data-section-title">Statistiques</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        fig_donut = go.Figure(go.Pie(
            labels=["Recyclable", "Non recyclable"],
            values=[n_rec, n_nonrec],
            hole=0.55,
            marker_colors=["#10b981", "#ef4444"],
            textinfo="label+percent",
        ))
        fig_donut.update_layout(
            title=dict(text="Répartition recyclabilité", font=dict(size=13), x=0),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10), height=270,
            font=dict(family="DM Sans, sans-serif"),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        cc = df["class"].value_counts().reset_index()
        cc.columns = ["class", "count"]
        cc["color"] = cc["class"].apply(
            lambda c: "#10b981" if SUPERCAT_RECYCLABLE_MAP.get(c, False) else "#ef4444"
        )
        fig_bar = go.Figure(go.Bar(
            x=cc["class"], y=cc["count"],
            marker_color=cc["color"],
            text=cc["count"], textposition="outside",
        ))
        fig_bar.update_layout(
            title=dict(text="Fréquence par supercatégorie", font=dict(size=13), x=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10), height=270,
            font=dict(family="DM Sans, sans-serif"), showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    fig_hist_conf = px.histogram(
        df, x="confidence", nbins=15,
        color="recyclable",
        color_discrete_map={True: "#10b981", False: "#ef4444"},
        labels={"confidence": "Confiance (%)", "recyclable": "Recyclable"},
        title="Distribution des niveaux de confiance",
    )
    fig_hist_conf.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10), height=255,
        font=dict(family="DM Sans, sans-serif"), bargap=0.05,
    )
    st.plotly_chart(fig_hist_conf, use_container_width=True)

    st.markdown('<div class="data-section-title">Historique des prédictions</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_rec = st.selectbox(
            "Recyclabilité",
            ["Tout", "Recyclable seulement", "Non recyclable seulement"],
        )
    with col_f2:
        filter_cls = st.selectbox(
            "Classe", ["Tout"] + sorted(df["class"].unique().tolist())
        )
    with col_f3:
        sort_opt = st.selectbox("Trier par", ["Plus récent", "Plus ancien"])

    items = list(hist)
    if filter_rec == "Recyclable seulement":
        items = [h for h in items if h["recyclable"]]
    elif filter_rec == "Non recyclable seulement":
        items = [h for h in items if not h["recyclable"]]
    if filter_cls != "Tout":
        items = [h for h in items if h.get("class") == filter_cls]
    if sort_opt == "Plus récent":
        items = list(reversed(items))

    st.markdown(f"**{len(items)} entrée(s) affichée(s)**")

    for item in items:
        dot       = "hist-dot-rec" if item["recyclable"] else "hist-dot-nonrec"
        label_rec = "recyclable" if item["recyclable"] else "non recyclable"
        ts        = item.get("timestamp", "--:--:--")
        emoji     = SUPERCAT_EMOJIS.get(item["class"], "🗑️")
        st.markdown(f"""
        <div class="hist-item">
            <div class="{dot}"></div>
            <span style="color:#94a3b8;font-size:0.78rem;min-width:52px">{ts}</span>
            <span style="flex:1"><strong>{item['name']}</strong></span>
            <span>{emoji} {item['class']}</span>
            <span style="color:#94a3b8;font-size:0.78rem">{item['confidence']:.0f}%</span>
            <span class="model-tag-yolo">YOLO-SuperCat</span>
            <span style="font-size:0.82rem">{label_rec}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_act1, col_act2 = st.columns(2)
    with col_act1:
        csv_buf = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Exporter en CSV",
            data=csv_buf,
            file_name=f"dechection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_act2:
        if st.button("🗑  Vider l'historique", use_container_width=True):
            st.session_state.history = []
            st.rerun()