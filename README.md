# SMSML Project Experiment - Stefan Aprilio

Projek ini mendokumentasikan alur MLOps lengkap mulai dari pengolahan data hingga pembangunan model dengan pemantauan eksperimen.

## Progres Proyek

### 1. Eksperimen Preprocessing
- **Dataset**: Menggunakan dataset **Iris** dari UCI ML Repository.
- **Folder**: `Eksperimen_SML_Stefan-Aprilio/`
- **Output**:
  - `iris_raw.csv`: Data mentah hasil download.
  - `preprocessing/Eksperimen_Stefan-Aprilio.ipynb`: Notebook analisis data dan preprocessing.
  - `preprocessing/automate_Stefan-Aprilio.py`: Skrip otomasi untuk membersihkan data secara terprogram.
  - `preprocessing/iris_preprocessing.csv`: Data hasil pembersihan dan standarisasi.

### 2. Membangun Model & Tracking
- **Folder**: `Membangun_model/`
- **Model Implementasi**:
  - `modelling.py`: Pembangunan model baseline menggunakan Random Forest.
  - `modelling_tuning.py`: Optimasi hyperparameter menggunakan GridSearchCV.
- **Monitoring**:
  - Integrasi dengan **MLflow** (Localhost: `http://127.0.0.1:5000`).
  - Integrasi dengan **DagsHub** untuk tracking eksperimen jarak jauh.
- **Artefak**:
  - Menyimpan metrik (accuracy, f1-score), parameter, dan model (.pkl).
  - Menyertakan mockup dashboard dan struktur artefak MLflow.

### 3. Manajemen Keamanan
- **Folder**: `secrets/`
- Menyediakan file `.env` untuk menyimpan `DAGSHUB_USER_TOKEN`, `MLFLOW_TRACKING_USERNAME`, dan `MLFLOW_TRACKING_PASSWORD`.
- Dilengkapi dengan `.gitignore` untuk melindungi rahasia agar tidak terunggah ke repositori publik.

## Cara Menjalankan
1. Jalankan MLflow UI: `python -m mlflow ui`
2. Eksekusi preprocessing: `python Eksperimen_SML_Stefan-Aprilio/preprocessing/automate_Stefan-Aprilio.py`
3. Latih model: `python Eksperimen_SML_Stefan-Aprilio/Membangun_model/modelling_tuning.py`