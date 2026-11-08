"""KneeOA AI — Streamlit public demo from the user's three EfficientNet-B3 notebooks.

Uses raw 0–255 RGB input, as in tf.keras's EfficientNet built-in rescaling.
Grad-CAM targets the last EfficientNet convolution: 'top_conv'.
"""

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="KneeOA AI — Klasifikasi Osteoarthritis",
    page_icon="🦴",
    layout="wide",
)

LABELS = ["0 — Normal", "1 — Doubtful", "2 — Mild", "3 — Moderate", "4 — Severe"]
LOCAL_MODEL_FILES = [
    Path("model/EfficientNetB3_Knee_Finall_fixed.h5"),
    Path("model/EfficientNetB3_Knee_Finall.h5"),
    Path("model/EfficientNetB3_KneeOA_best.keras"),
]


def setting(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_model_path() -> str:
    custom_path = setting("MODEL_PATH")
    if custom_path and Path(custom_path).is_file():
        return custom_path

    for path in LOCAL_MODEL_FILES:
        if path.is_file():
            return str(path)

    repo_id = setting("HF_MODEL_REPO", "name065/kneeoa-efficientnetb3")
    filename = setting("HF_MODEL_FILENAME", "EfficientNetB3_Knee_Finall_fixed.h5")
    token = setting("HF_TOKEN") or None

    if not repo_id:
        raise FileNotFoundError(
            "Model belum tersedia. Upload file model di folder model/ atau "
            "atur HF_MODEL_REPO dan HF_MODEL_FILENAME di Streamlit Secrets."
        )

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=token,
    )


@st.cache_resource(show_spinner=False)
def load_classifier():
    # Import in cached function so TensorFlow is loaded only when needed.
    import tensorflow as tf

    # Limit unnecessary CPU threads on small free-tier machines.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    model_path = get_model_path()
    model = tf.keras.models.load_model(model_path, compile=False)
    if len(model.outputs) != 1 or int(model.output_shape[-1]) != len(LABELS):
        raise ValueError(
            f"Model harus menghasilkan {len(LABELS)} skor kelas. "
            f"Output sekarang: {model.output_shape}"
        )
    return model


def prepare_image(image: Image.Image, model):
    # The EfficientNetB3 notebook trains on unnormalized RGB [0,255].
    image = ImageOps.exif_transpose(image).convert("RGB")
    height, width = model.input_shape[1:3]
    width = int(width or 299)
    height = int(height or 299)
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    return image, batch


def predict_and_gradcam(model, batch):
    """One forward pass for predictions + one gradient pass for Grad-CAM."""
    import tensorflow as tf

    scores = np.asarray(model(batch, training=False))[0].astype(np.float32)
    predicted_index = int(np.argmax(scores))

    heatmap = None
    gradcam_message = None
    try:
        # In the training notebook EfficientNetB3 receives input_tensor=inputs,
        # so 'top_conv' should be a directly accessible layer.
        conv_layer = model.get_layer("top_conv")
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.outputs[0]],
        )
        with tf.GradientTape() as tape:
            conv_activation, predictions = grad_model(batch, training=False)
            selected_score = predictions[:, predicted_index]
        gradient = tape.gradient(selected_score, conv_activation)
        if gradient is None:
            raise ValueError("Gradient terhadap 'top_conv' tidak tersedia")

        # Upcast to float32 to avoid overflow/underflow for mixed precision.
        gradient = tf.cast(gradient, tf.float32)
        conv_activation = tf.cast(conv_activation, tf.float32)
        channel_weights = tf.reduce_mean(gradient, axis=(0, 1, 2))
        heat = tf.reduce_sum(conv_activation[0] * channel_weights, axis=-1)
        heat = tf.nn.relu(heat)
        peak = float(tf.reduce_max(heat).numpy())
        heatmap = (heat / peak).numpy() if peak > 1e-8 else np.zeros(heat.shape, dtype=np.float32)
    except Exception as exc:
        # A nested or custom model can require an adapted Grad-CAM graph.
        gradcam_message = (
            "Prediksi berhasil, tetapi Grad-CAM belum dapat dibuat. "
            f"Periksa layer 'top_conv' dan struktur model: {type(exc).__name__}: {exc}"
        )
    return scores, predicted_index, heatmap, gradcam_message


def heatmap_images(original: Image.Image, heatmap: np.ndarray):
    """Resize + colorize without opencv/matplotlib to save free-tier RAM."""
    height, width = original.height, original.width
    heat_uint8 = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
    heat_uint8 = heat_uint8.resize((width, height), Image.Resampling.BILINEAR)
    value = np.asarray(heat_uint8, dtype=np.float32) / 255.0
    # RGB approximation of the classic Jet color map used in the notebook.
    red = np.clip(1.5 - np.abs(4 * value - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * value - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * value - 1), 0, 1)
    colored_array = np.stack([red, green, blue], axis=-1)
    colored = Image.fromarray((colored_array * 255).astype(np.uint8), mode="RGB")
    overlay = Image.blend(original, colored, alpha=0.40)
    return colored, overlay


st.title("🦴 KneeOA AI")
st.caption("Klasifikasi citra X-ray lutut menggunakan EfficientNet-B3 dan Grad-CAM")
st.info(
    "Prototipe penelitian/edukasi, bukan alat diagnosis medis. "
    "Hasil prediksi harus dikaji tenaga kesehatan yang kompeten."
)

with st.sidebar:
    st.subheader("Tentang model")
    st.write("Arsitektur: EfficientNet-B3")
    st.write("Klasifikasi: 5 kelas Kellgren–Lawrence (0–4)")
    st.write("Visualisasi: Grad-CAM (layer `top_conv`)")
    st.caption("Gambar yang diunggah hanya diproses untuk prediksi dalam sesi aplikasi.")

uploaded_file = st.file_uploader(
    "Unggah gambar X-ray lutut (JPG, JPEG, atau PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        if image.width * image.height > 25_000_000:
            st.error("Ukuran gambar terlalu besar. Gunakan gambar dengan maksimal 25 megapiksel.")
            st.stop()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        st.error(f"File gambar tidak dapat dibuka: {exc}")
        st.stop()

    st.image(image, caption="X-ray yang diunggah", width=340)
    if st.button("Analisis gambar", type="primary", use_container_width=True):
        with st.spinner("Mengunduh/memuat model dan menjalankan prediksi ..."):
            try:
                model = load_classifier()
                original, batch = prepare_image(image, model)
                scores, index, heat, warning = predict_and_gradcam(model, batch)
            except Exception as exc:
                st.error(f"Model belum dapat dijalankan: {type(exc).__name__}: {exc}")
                st.info(
                    "Pastikan file model sudah diunggah ke Hugging Face Model Repository "
                    "atau ada di folder model/. Jika model lama gagal dibuka karena mixed_float16, "
                    "gunakan file EfficientNetB3_Knee_Finall_fixed.h5 yang kompatibel."
                )
                st.stop()

        st.subheader("Hasil klasifikasi")
        a, b = st.columns(2)
        a.metric("Kelas prediksi", LABELS[index])
        b.metric("Skor model", f"{scores[index] * 100:.2f}%")
        st.caption("Skor model bukan probabilitas diagnosis klinis yang telah dikalibrasi.")
        st.subheader("Skor setiap kelas")
        st.bar_chart({"Kelas": LABELS, "Skor": [float(x) for x in scores]}, x="Kelas", y="Skor", horizontal=True)

        st.subheader("Visualisasi Grad-CAM")
        if heat is not None:
            colored, overlay = heatmap_images(original, heat)
            left, right = st.columns(2)
            with left:
                st.image(colored, caption="Peta panas Grad-CAM", use_container_width=True)
            with right:
                st.image(overlay, caption="Grad-CAM ditumpangkan pada X-ray", use_container_width=True)
            st.caption(
                "Area berwarna lebih hangat menunjukkan lokasi yang berkontribusi "
                "terhadap skor kelas yang diprediksi. Ini bukan penandaan lesi yang tervalidasi."
            )
        else:
            st.warning(warning or "Grad-CAM tidak tersedia untuk model ini.")
else:
    st.write("Pilih satu gambar, lalu klik **Analisis gambar** untuk memulai.")
