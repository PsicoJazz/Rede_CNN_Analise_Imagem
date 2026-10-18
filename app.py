"""
Interface Streamlit para o classificador Gatos x Cachorros (InceptionV3 + transfer learning).

Como rodar:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# Configurações (devem espelhar o que foi usado no treinamento do notebook)
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "Redes_Neurais_CNN_Reconhecimento_de_imagens.keras"
IMG_SIZE = (150, 150)          # target_size do flow_from_directory
# flow_from_directory ordena as pastas alfabeticamente: cats=0, dogs=1
CLASS_NAMES = {0: "Gato 🐱", 1: "Cachorro 🐶"}

st.set_page_config(page_title="Gato ou Cachorro?", page_icon="🐾", layout="centered")


# ----------------------------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando a rede neural...")
def load_model(path: str):
    """Carrega o modelo uma única vez e mantém em cache entre as interações."""
    import keras  # importado aqui para a tela abrir rápido

    return keras.models.load_model(path, compile=False)


def preprocess(image: Image.Image) -> np.ndarray:
    """Mesmo pré-processamento do treino: RGB, 150x150 e rescale 1/255."""
    image = ImageOps.exif_transpose(image)      # corrige fotos de celular rotacionadas
    image = image.convert("RGB")                # remove canal alfa / converte escala de cinza
    image = image.resize(IMG_SIZE)
    arr = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)          # shape (1, 150, 150, 3)


def predict(model, image: Image.Image) -> float:
    """Retorna a probabilidade (0 a 1) de ser cachorro."""
    return float(model.predict(preprocess(image), verbose=0)[0][0])


def show_result(prob_dog: float, threshold: float):
    prob_cat = 1.0 - prob_dog
    label = CLASS_NAMES[1] if prob_dog >= 0.5 else CLASS_NAMES[0]
    confidence = max(prob_dog, prob_cat)

    if confidence < threshold:
        st.warning(f"Resultado incerto — mais provável: **{label}** ({confidence:.1%})")
    else:
        st.success(f"## {label}\nConfiança: **{confidence:.1%}**")

    col1, col2 = st.columns(2)
    col1.metric("Gato 🐱", f"{prob_cat:.1%}")
    col2.metric("Cachorro 🐶", f"{prob_dog:.1%}")
    st.progress(prob_dog, text="Probabilidade de ser cachorro")


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("🐾 Gato ou Cachorro?")
st.caption("Classificador de imagens baseado em InceptionV3 (transfer learning).")

with st.sidebar:
    st.header("Configurações")
    threshold = st.slider(
        "Confiança mínima",
        min_value=0.5, max_value=0.99, value=0.70, step=0.01,
        help="Abaixo desse valor, o resultado é marcado como incerto.",
    )
    st.markdown("---")
    st.markdown(
        "**Atenção:** o modelo só conhece gatos e cachorros. "
        "Qualquer outra imagem (carro, pessoa, paisagem...) será "
        "forçada a cair em uma dessas duas classes."
    )

if not MODEL_PATH.exists():
    st.error(
        f"Arquivo do modelo não encontrado:\n\n`{MODEL_PATH}`\n\n"
        "Coloque o arquivo `.keras` na mesma pasta do `app.py`."
    )
    st.stop()

model = load_model(str(MODEL_PATH))

tab_upload, tab_camera = st.tabs(["📁 Enviar imagens", "📷 Usar câmera"])

with tab_upload:
    files = st.file_uploader(
        "Escolha uma ou mais imagens",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )
    for f in files or []:
        st.divider()
        img = Image.open(f)
        left, right = st.columns([1, 1.2])
        with left:
            st.image(img, caption=f.name, use_container_width=True)
        with right:
            with st.spinner("Classificando..."):
                p = predict(model, img)
            show_result(p, threshold)

with tab_camera:
    photo = st.camera_input("Tire uma foto")
    if photo is not None:
        img = Image.open(photo)
        with st.spinner("Classificando..."):
            p = predict(model, img)
        show_result(p, threshold)