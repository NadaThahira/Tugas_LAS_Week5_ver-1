"""
TomaLeaf Dx — Diagnosa Penyakit Daun Tomat
Deployment model: CNN Custom
Nada Thahira Sosa — 2601 — MBC Lab Week 5
"""

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TomaLeaf Dx — Diagnosa Daun Tomat",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

IMG_SIZE = (224, 224)
MODEL_PATH = "tomato_leaf_best_model.h5"

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "nama": "Bercak Bakteri (Bacterial Spot)",
        "tingkat": "sedang",
        "penyebab": "Bakteri Xanthomonas spp.",
        "gejala": "Bercak kecil kehitaman/kecoklatan pada daun, sering dikelilingi lingkaran kuning (halo).",
        "saran": "Buang daun yang terinfeksi, hindari penyiraman dari atas, semprot bakterisida berbasis tembaga.",
    },
    "Tomato___Early_blight": {
        "nama": "Bercak Daun Awal (Early Blight)",
        "tingkat": "sedang",
        "penyebab": "Jamur Alternaria solani.",
        "gejala": "Bercak coklat berbentuk cincin konsentris (seperti target), dimulai dari daun tua bagian bawah.",
        "saran": "Rotasi tanaman, buang daun terinfeksi, gunakan fungisida (mis. klorotalonil) sesuai dosis.",
    },
    "Tomato___Late_blight": {
        "nama": "Bercak Daun Akhir (Late Blight)",
        "tingkat": "berat",
        "penyebab": "Oomycete Phytophthora infestans.",
        "gejala": "Bercak basah kehijauan-kehitaman yang cepat meluas, dapat menghancurkan tanaman dalam hitungan hari.",
        "saran": "Segera isolasi/musnahkan tanaman terinfeksi berat, semprot fungisida sistemik, perbaiki sirkulasi udara.",
    },
    "Tomato___Leaf_Mold": {
        "nama": "Jamur Daun (Leaf Mold)",
        "tingkat": "ringan",
        "penyebab": "Jamur Passalora fulva (dulu Fulvia fulva).",
        "gejala": "Bercak kuning pucat di permukaan atas daun, lapisan beludru zaitun di permukaan bawah.",
        "saran": "Kurangi kelembapan rumah kaca/greenhouse, tingkatkan ventilasi, gunakan fungisida bila perlu.",
    },
    "Tomato___Septoria_leaf_spot": {
        "nama": "Bercak Septoria (Septoria Leaf Spot)",
        "tingkat": "sedang",
        "penyebab": "Jamur Septoria lycopersici.",
        "gejala": "Bercak bulat kecil dengan pusat abu-abu dan tepi gelap, menyebar dari daun bawah ke atas.",
        "saran": "Mulsa tanah, hindari daun basah berlama-lama, buang daun terinfeksi, rotasi tanaman.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "nama": "Tungau Laba-laba (Two-Spotted Spider Mite)",
        "tingkat": "sedang",
        "penyebab": "Hama tungau Tetranychus urticae.",
        "gejala": "Bintik kuning kecil (stippling), jaring halus di bawah daun, daun mengering saat parah.",
        "saran": "Semprot air bertekanan pada bawah daun, gunakan akarisida/minyak neem, jaga kelembapan udara.",
    },
    "Tomato___Target_Spot": {
        "nama": "Bercak Target (Target Spot)",
        "tingkat": "sedang",
        "penyebab": "Jamur Corynespora cassiicola.",
        "gejala": "Bercak coklat dengan cincin konsentris mirip early blight, dapat menyebar ke batang dan buah.",
        "saran": "Rotasi tanaman, perbaiki sirkulasi udara, aplikasikan fungisida preventif.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "nama": "Virus Keriting Daun Kuning (TYLCV)",
        "tingkat": "berat",
        "penyebab": "Virus yang ditularkan oleh kutu kebul (whitefly).",
        "gejala": "Daun menguning, menggulung ke atas, tanaman kerdil dan pertumbuhan terhambat.",
        "saran": "Kendalikan populasi kutu kebul, cabut & musnahkan tanaman terinfeksi, gunakan varietas tahan virus.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "nama": "Virus Mosaik Tomat (ToMV)",
        "tingkat": "berat",
        "penyebab": "Tobamovirus, menular lewat kontak/alat pertanian.",
        "gejala": "Pola mosaik hijau muda-tua pada daun, daun keriting dan pertumbuhan terhambat.",
        "saran": "Sterilkan alat pertanian, cuci tangan sebelum menangani tanaman, musnahkan tanaman terinfeksi.",
    },
    "Tomato___healthy": {
        "nama": "Sehat",
        "tingkat": "sehat",
        "penyebab": "—",
        "gejala": "Tidak ditemukan tanda-tanda penyakit pada daun.",
        "saran": "Lanjutkan perawatan rutin: penyiraman cukup, pemupukan seimbang, pantau berkala.",
    },
}

SEVERITY_COLOR = {
    "sehat": "#4F7A3D",
    "ringan": "#8AA24C",
    "sedang": "#D98E30",
    "berat": "#B33A3A",
}


# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Work+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Work Sans', sans-serif; }
    h1, h2, h3, .hero-title { font-family: 'Fraunces', serif; }

    .stApp { background: #F2F5EC; }

    .hero {
        text-align: center;
        padding: 1.4rem 1rem 0.6rem 1rem;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0.3rem 0 0.3rem 0;
        color: #2F4B32;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #556B4C;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Step indicator */
    .steps { display: flex; justify-content: center; gap: 0.5rem; margin: 1.4rem 0 1.6rem 0; }
    .step {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.85rem; color: #8A9A80;
        padding: 0.4rem 0.9rem; border-radius: 20px;
        background: #E5EADA;
    }
    .step.active { background: #2F4B32; color: #F2F5EC; font-weight: 600; }
    .step.done { background: #DCE8D2; color: #2F4B32; }
    .step-num {
        width: 20px; height: 20px; border-radius: 50%;
        background: rgba(255,255,255,0.25);
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 0.75rem;
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E1E7D8;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }

    .tips-list { font-size: 0.88rem; color: #4B5A44; margin: 0; padding-left: 1.1rem; }
    .tips-list li { margin-bottom: 0.3rem; }

    .result-card {
        border-radius: 14px;
        padding: 1.5rem 1.6rem;
        color: #FFFFFF;
        margin: 0.4rem 0 1rem 0;
        text-align: center;
    }
    .result-label { font-size: 0.82rem; letter-spacing: 0.5px; opacity: 0.85; text-transform: uppercase; }
    .result-name { font-family: 'Fraunces', serif; font-size: 1.7rem; font-weight: 600; margin: 0.3rem 0; }
    .result-conf { font-size: 1rem; opacity: 0.95; }

    .barrow { display: flex; align-items: center; margin: 0.35rem 0; gap: 0.6rem; }
    .barrow-label { width: 230px; font-size: 0.8rem; color: #2B3A28; flex-shrink: 0; }
    .barrow-track { flex: 1; background: #E5EADA; border-radius: 6px; height: 9px; overflow: hidden; }
    .barrow-fill { height: 100%; border-radius: 6px; }
    .barrow-pct { width: 44px; text-align: right; font-size: 0.78rem; color: #2B3A28; }

    /* Paksa semua teks native Streamlit ikut warna tema ini, jangan ikut dark-mode bawaan browser/Streamlit */
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stCaption, [data-testid="stCaptionContainer"] {
        color: #22301C !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important; border: 2px dashed #DDE5CE !important; border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #22301C !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #2F4B32 !important; color: #F2F5EC !important; border: none !important;
    }
    .stButton button, .stDownloadButton button {
        background: #2F4B32 !important; color: #F2F5EC !important;
        border: none !important; border-radius: 10px !important;
    }
    [data-testid="stExpander"] { background: #FFFFFF !important; border: 1px solid #DDE5CE !important; border-radius: 12px !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #22301C !important; }
    [data-testid="stExpander"] p, [data-testid="stExpander"] li, [data-testid="stExpander"] span { color: #22301C !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(image: Image.Image):
    model = load_model()
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return model.predict(batch, verbose=0)[0]


def render_steps(active: int):
    labels = ["Upload Foto", "Proses", "Lihat Hasil"]
    html = '<div class="steps">'
    for i, label in enumerate(labels, start=1):
        cls = "active" if i == active else ("done" if i < active else "")
        html += f'<div class="step {cls}"><span class="step-num">{i}</span>{label}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div style="font-size:2.2rem;">🍅</div>
        <div class="hero-title">TomaLeaf Dx</div>
        <div class="hero-sub">Foto daun tomatmu, sistem langsung mendeteksi apakah sehat atau
        terkena salah satu dari 9 penyakit umum.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ----------------------------------------------------------------------------
# ALUR UTAMA: 3 LANGKAH
# ----------------------------------------------------------------------------
if uploaded is None:
    render_steps(1)
    st.markdown(
        """
        <div class="card">
        <b>Cara pakai (30 detik):</b>
        <ul class="tips-list">
            <li>Klik kotak di atas, atau tarik & lepas foto daun tomat.</li>
            <li>Ambil foto <b>close-up 1 daun</b>, cahaya cukup, latar polos.</li>
            <li>Hasil diagnosa & saran penanganan langsung muncul otomatis.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    image = Image.open(uploaded)
    render_steps(2)

    col_img, col_gap = st.columns([1, 0.001])
    with col_img:
        st.image(image, use_column_width=True, caption="Foto yang kamu unggah")

    with st.spinner("Sedang menganalisis daun..."):
        probs = predict(image)

    render_steps(3)

    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    info = DISEASE_INFO[top_class]
    color = SEVERITY_COLOR[info["tingkat"]]

    st.markdown(
        f"""
        <div class="result-card" style="background:{color};">
            <div class="result-label">Hasil Diagnosa</div>
            <div class="result-name">{info['nama']}</div>
            <div class="result-conf">Tingkat keyakinan model: {probs[top_idx]*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="card">
            <p style="margin:0 0 0.4rem 0;"><b>Penyebab:</b> {info['penyebab']}</p>
            <p style="margin:0 0 0.4rem 0;"><b>Gejala:</b> {info['gejala']}</p>
            <p style="margin:0;"><b>Saran penanganan:</b> {info['saran']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat rincian keyakinan untuk semua kelas"):
        order = np.argsort(probs)[::-1]
        bars_html = ""
        for i in order:
            pct = probs[i] * 100
            bars_html += f"""
            <div class="barrow">
                <div class="barrow-label">{DISEASE_INFO[CLASSES[i]]['nama']}</div>
                <div class="barrow-track"><div class="barrow-fill" style="width:{pct:.1f}%; background:{SEVERITY_COLOR[DISEASE_INFO[CLASSES[i]]['tingkat']]};"></div></div>
                <div class="barrow-pct">{pct:.1f}%</div>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)

    if st.button("🔄 Coba foto lain", use_container_width=True):
        st.rerun()

# ----------------------------------------------------------------------------
# INFO TAMBAHAN (di luar alur utama, tidak mengganggu)
# ----------------------------------------------------------------------------
with st.expander("📖 Daftar penyakit yang bisa dikenali"):
    for cls in CLASSES:
        info = DISEASE_INFO[cls]
        color = SEVERITY_COLOR[info["tingkat"]]
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid {color}; margin-bottom:0.6rem;">
                <b>{info['nama']}</b>
                <span style="font-size:0.72rem; color:{color}; font-weight:600;"> · {info['tingkat'].upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("TomaLeaf Dx · Model: CNN Custom · Nada Thahira Sosa — 2601 · MBC Lab Week 5")
