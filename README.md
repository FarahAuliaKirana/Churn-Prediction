# Customer Churn Prediction — Olist Brazilian E-Commerce

Prediksi customer churn menggunakan dataset e-commerce publik Olist (Brazil) dengan pendekatan end-to-end: dari raw data hingga REST API production-ready yang disertai interpretabilitas berbasis SHAP.

---

## Hasil Utama

| Metrik | Nilai |
|---|---|
| **Champion Model** | XGBoost |
| **ROC-AUC** | 0.8365 |
| **CV AUC (5-fold)** | 0.8334 ± 0.0084 |
| **Churn Recall** | 99.2% (threshold 0.470) |
| **Non-Churn Recall** | 76% |
| **Prediktor #1** | `avg_freight` — dikonfirmasi EDA + SHAP |
| **Leakage terdeteksi** | ✅ AUC 0.97 → 0.84 terdokumentasi |

---

## Struktur Proyek

```
olist-churn-prediction/
│
├── notebooks
|   ├── 📓 01_data_loading.ipynb          # Load, merge 7 tabel, validasi
|   ├── 📓 02_eda.ipynb                   # EDA, definisi churn, freight analysis
|   ├── 📓 03_feature_engineering.ipynb   # Feature matrix, multikolinearitas
|   ├── 📓 04_modeling.ipynb              # Benchmark, leakage fix, SHAP, XGBoost
|   └── 📓 05_final_report.ipynb          # Ringkasan eksekutif ← baca ini dulu
│
├── train_pipeline.py                 # CLI end-to-end training
├── requirements.txt
├── Dockerfile
├── .dockerignore
│
├── src/
│   ├── api/
│   │   ├── main.py                   # FastAPI app & endpoints
│   │   └── schemas.py                # Pydantic request/response models
│   ├── data/
│   │   └── load.py                   # Load & merge 7 tabel Olist
│   ├── features/
│   │   └── engineer.py               # Churn labeling & feature engineering
│   └── models/
│       ├── train.py                  # XGBoost training pipeline
│       └── evaluate.py               # SHAP & threshold tuning
│
├── data/
│   ├── raw/                          # CSV mentah dari Kaggle (tidak di-push)
│   └── processed/
│       ├── master.csv                # 113,425 baris × 23 kolom
│       ├── rfm_labeled.csv           # 96,096 customers + label churn
│       └── features.csv              # Feature matrix untuk modeling
│
└── outputs/
    ├── model_xgb.pkl                 # Pipeline XGBoost siap deploy
    ├── feature_cols.json             # Kolom training — wajib ada untuk API
    ├── metrics.json                  # ROC-AUC, CV AUC, F1
    ├── shap_top_features.csv         # Top 10 fitur berdasarkan SHAP
    ├── model_comparison_final.csv    # Scorecard semua model
    └── figures/
        └── shap_summary.png          # SHAP summary plot
```

---

## Cara Menjalankan

### 1. Clone & Install

```bash
git clone https://github.com/username/olist-churn-prediction.git
cd olist-churn-prediction
pip install -r requirements.txt
```

### 2. Download Dataset

Download dari [Kaggle — Olist Brazilian E-Commerce](https://www.kaggle.com/olistbr/brazilian-ecommerce) dan ekstrak semua CSV ke folder `data/raw/`.

### 3. Training Pipeline

Jalankan satu perintah untuk proses end-to-end: load data → feature engineering → train → SHAP → simpan model.

```bash
python train_pipeline.py --raw-path data/raw --output-dir outputs
```

Output yang dihasilkan:
- `outputs/model_xgb.pkl` — model siap serve
- `outputs/feature_cols.json` — kolom training (wajib ada untuk API)
- `outputs/metrics.json` — ROC-AUC, CV AUC, F1
- `outputs/figures/shap_summary.png` — SHAP summary plot

### 4. Jalankan API

```bash
uvicorn src.api.main:app --reload
```

API tersedia di `http://localhost:8000`. Dokumentasi interaktif Swagger UI di `http://localhost:8000/docs`.

#### Contoh request `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_001",
    "avg_payment_value": 150.0,
    "total_payment_value": 150.0,
    "avg_installments": 3.0,
    "max_installments": 3,
    "avg_price": 120.0,
    "total_price": 120.0,
    "avg_freight": 30.0,
    "freight_ratio": 0.20,
    "avg_review_score": 4.0,
    "min_review_score": 4.0,
    "review_gap": 0.0,
    "review_count": 1,
    "total_items": 1,
    "total_orders": 1,
    "avg_order_value": 150.0,
    "frequency": 1,
    "monetary": 150.0,
    "log_monetary": 5.01,
    "log_frequency": 0.69,
    "customer_lifetime_days": 0,
    "customer_state": "SP",
    "dominant_payment_type": "credit_card",
    "dominant_category": "health_beauty"
  }'
```

Response:
```json
{
  "customer_id": "cust_001",
  "churn_probability": 0.9847,
  "is_churn": true,
  "threshold_used": 0.47,
  "model_version": "1.0.0"
}
```

### 5. Jalankan Notebook (opsional — eksplorasi)

```bash
jupyter notebook
```

Urutan eksekusi:
1. `01_data_loading.ipynb`
2. `02_eda.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_modeling.ipynb`
5. `05_final_report.ipynb`

---

## Menjalankan dengan Docker

### Build & Run

```bash
# Build image
docker build -t olist-churn-api .

# Run container
docker run -p 8000:8000 olist-churn-api
```

API tersedia di `http://localhost:8000/docs`.

### Catatan

Dockerfile mengasumsikan `outputs/model_xgb.pkl` dan `outputs/feature_cols.json` sudah ada (hasil `train_pipeline.py`). Jalankan training pipeline terlebih dahulu sebelum build image, atau mount folder `outputs/` sebagai volume:

```bash
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs olist-churn-api
```

---

## Endpoints API

| Method | Endpoint | Deskripsi |
|---|---|---|
| GET | `/health` | Status API & model |
| POST | `/predict` | Prediksi churn 1 customer |
| POST | `/predict/batch` | Prediksi batch hingga 1000 customer |

---

## Temuan Kritis: Data Leakage

Model pertama menghasilkan AUC **0.9752** — mencurigakan. Investigasi menemukan `first_order_month` mendominasi feature importance dengan score 0.57.

Penyebabnya: customer yang first order-nya dekat cutoff Oktober 2018 otomatis non-churn (belum sempat churn karena dataset berakhir). Ini artefak temporal, bukan sinyal perilaku.

Setelah drop `first_order_month` dan `first_order_dayofweek`:

| Model | AUC Sebelum | AUC Sesudah |
|---|---|---|
| Random Forest | 0.9752 | 0.8059 |
| Decision Tree | 0.9669 | 0.7892 |
| Logistic Regression | 0.7132 | 0.6049 |

AUC **0.8365** (XGBoost) adalah performa genuine yang bisa dipercaya untuk deployment.

---

## Key Insights dari SHAP

SHAP mengungkap arah pengaruh yang tidak terlihat dari feature importance biasa:

| Fitur | Arah | Interpretasi |
|---|---|---|
| `avg_freight` ↑ | → churn ↑ | Ongkir tinggi adalah friction utama repeat purchase |
| `customer_state_SP` = True | → churn ↓ | Customer SP lebih loyal — dekat pusat logistik Olist |
| `dominant_payment_type_debit_card` = True | → churn ↓ | Pengguna debit lebih terencana & loyal |
| `dominant_category_toys` = True | → churn ↑ | Pembeli mainan cenderung one-time buyer musiman |

> **Temuan penting:** `debit_card` muncul tinggi di feature importance XGBoost, tapi SHAP menunjukkan pengguna debit justru **lebih loyal** — bukan lebih churn. Feature importance tanpa SHAP bisa menyesatkan.

---

## Business Recommendations

**Prioritas 1 — Free Shipping Threshold**
Program 'gratis ongkir untuk order > BRL 150'. Langsung menyerang friction #1 dan bisa di-A/B test dalam 30 hari. Estimasi: jika 10% dari 17K churn predicted berhasil di-retain → **BRL 361K revenue recovery per cohort**.

**Prioritas 2 — Second Purchase Activation**
Voucher 15% untuk order kedua, dikirim D+14 setelah first order. Masalah utama bukan retention tapi activation — 96.9% customer tidak pernah kembali bahkan untuk kali kedua.

**Prioritas 3 — Geo-Targeted Retention**
Subsidi ongkir diferensial ke state dengan churn rate tertinggi (identifikasi via SHAP dependence plot). Customer SP lebih loyal secara struktural karena ongkir murah — replikasi kondisi ini ke state lain.

**Prioritas 4 — Segmentasi Kategori**
Jangan buang budget retensi ke pembeli `toys` (musiman, one-time by nature). Fokus ke `health_beauty` dan `bed_bath_table` — produk habis pakai dengan potensi repeat purchase alami.

---

## Stack

| Tool | Versi | Kegunaan |
|---|---|---|
| Python | 3.9+ | |
| pandas | ≥ 1.5.0 | Data manipulation |
| numpy | ≥ 1.23.0 | Numerical operations |
| scikit-learn | ≥ 1.1.0 | Pipeline, models, metrics |
| xgboost | ≥ 1.7.0 | Champion model |
| shap | ≥ 0.41.0 | Model interpretability |
| matplotlib / seaborn | ≥ 3.6.0 / 0.12.0 | Visualisasi |
| joblib | ≥ 1.2.0 | Model serialization |
| fastapi | ≥ 0.100.0 | REST API serving |
| uvicorn | ≥ 0.22.0 | ASGI server |
| pydantic | ≥ 2.0.0 | Request/response validation |

---

## Keterbatasan

- **Right-censoring:** Churn rate 89.9% mengandung artefak cutoff Oktober 2018 — angka ini *upper bound*, bukan gambaran bisnis sesungguhnya
- **Random split:** Metodologi yang lebih ketat untuk time-series adalah temporal split (train 2016–2017, test 2018) — dipertahankan di sini untuk menjaga jumlah sampel
- **One-time buyer dominance:** 96.9% customer hanya order 1 kali — model ini paling relevan untuk customer dengan riwayat ≥ 1 order

---

## Dataset

**Sumber:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)  
**Lisensi:** CC BY-NC-SA 4.0  
**Periode:** September 2016 – Oktober 2018  
**Ukuran:** ~100K orders, 96K unique customers

---

*Farah Aulia Kirana — Churn Prediction Portfolio Project*