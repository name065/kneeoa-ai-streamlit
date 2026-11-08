# KneeOA AI — Streamlit Community Cloud (gratis)

Template demo klasifikasi Knee Osteoarthritis menggunakan model EfficientNet-B3
berdasarkan notebook training, testing, dan Grad-CAM yang diunggah.

## Isi

- `app.py`: upload citra X-ray, inferensi 5 kelas, Grad-CAM (`top_conv`).
- `requirements.txt`: dependensi Python untuk server.
- `.streamlit/config.toml`: pengaturan dasar tampilan dan unggahan.
- `model/README.txt`: petunjuk lokasi model.

**PENTING:** file model `.h5` tidak termasuk ZIP, karena tidak tersedia dalam notebook.
Anda harus menyediakan `EfficientNetB3_Knee_Finall_fixed.h5` yang sudah berhasil dibuka pada PC Anda.

## Langkah A — unggah model ke Hugging Face (Model Repository, BUKAN Spaces)

1. Buka https://huggingface.co/new, buat repository berjenis **Model** (public).
2. Gunakan nama repo `name065/kneeoa-efficientnetb3` jika akun Anda memang `name065`,
   atau nama repo lain milik Anda.
3. Buka **Files and versions > Add file > Upload files**.
4. Upload `EfficientNetB3_Knee_Finall_fixed.h5` dan commit.
5. Jika nama repo atau filename berbeda, tambahkan nilai `HF_MODEL_REPO` dan
   `HF_MODEL_FILENAME` ke **Streamlit Community Cloud > Advanced settings > Secrets**:

```toml
HF_MODEL_REPO = "username-anda/nama-repo-model"
HF_MODEL_FILENAME = "EfficientNetB3_Knee_Finall_fixed.h5"
```

Kalau model repo bersifat private, tambah `HF_TOKEN` (read token), lewat
Streamlit Secrets saja, jangan masuk ke file yang diunggah ke GitHub.

## Langkah B — unggah code ke GitHub tanpa Git CLI

1. Ekstrak ZIP ini.
2. Di https://github.com/new, buat repository misalnya `kneeoa-ai-streamlit`.
3. Pada halaman repo gunakan **Add file > Upload files**, lalu unggah `app.py`,
   `requirements.txt`, `.gitignore`, dan folder `.streamlit` beserta `config.toml`.
   Jika unggahan folder tersembunyi sulit, Anda cukup unggah `app.py` dan
   `requirements.txt`. Berkas `.streamlit/config.toml` opsional.
4. **Jangan unggah** virtual environment `.venv` atau model `.h5` ke GitHub.

## Langkah C — deploy ke Streamlit Community Cloud

1. Buka https://share.streamlit.io dan masuk dengan GitHub.
2. Klik **Create app** → aplikasi dari existing repository.
3. Pilih repo Anda, branch `main`, entrypoint `app.py`.
4. Klik **Advanced settings**, pilih **Python 3.11** bila tersedia.
5. Tambahkan secrets (lihat Langkah A) jika nama repo model Anda berbeda,
   atau model repo private.
6. Klik **Deploy**. Tunggu proses instalasi dependencies dan cek logs jika gagal.
7. Setelah aktif, bagikan URL yang berakhiran `.streamlit.app`.

## Pengujian lokal di Windows tanpa Activate.ps1

Buka PowerShell di folder proyek ini:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Anda tidak perlu mengaktifkan `Activate.ps1`, sehingga kebijakan PowerShell tidak memblokirnya.
Jika `py -3.10` tidak ada, gunakan versi Python yang terpasang dan kompatibel.

## Kesesuaian model dan batasan

- Notebook training menggunakan input `299×299` RGB dan model EfficientNet
  melakukan normalisasi internal; aplikasi mengambil ukuran input secara otomatis.
- Notebook Grad-CAM menggunakan layer `top_conv`.
- Model `.h5` yang asli pernah bermasalah karena konfigurasi `mixed_float16`.
  Gunakan model `_fixed.h5` yang telah dikonversi/terverifikasi sebelumnya.
- Deploy ini **template yang belum diuji terhadap file model asli**, karena
  file model tidak dilampirkan. Struktur Grad-CAM dapat perlu perubahan apabila
  model tersimpan dalam struktur nested atau backend Keras berbeda.
- Streamlit Community Cloud memberikan RAM terbatas (~2.7GB maksimum). Jika
  TensorFlow + Grad-CAM melebihi batas, gunakan Oracle Cloud Always Free atau
  Cloud Run free quota dengan pengamanan billing sebagai alternatif.
- Aplikasi ini untuk penelitian/edukasi dan bukan perangkat diagnosis medis.
