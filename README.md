# Customer Churn Prediction — Olist Brazilian E-Commerce

Prediksi customer churn menggunakan dataset e-commerce publik Olist (Brazil) dengan pendekatan end-to-end: dari raw data hingga REST API production-ready yang disertai interpretabilitas berbasis SHAP.

---

## Hasil Utama

Proyek ini punya dua versi model:

| | **v1 — Churn (desain lama)** | **v2 — Repeat purchase (snapshot)** |
|---|---|---|
| Pertanyaan | Apakah order terakhir > 90 hari sebelum akhir dataset? | Apakah customer order lagi dalam 90 hari setelah tanggal T? |
| Validasi | Random split 60/20/20 | Out-of-time + backtest 3 periode (Des 2017, Mar 2018, Jun 2018) |
| ROC-AUC | 0.8207 | 0.592 rata-rata backtest (RFM LogReg) |
| Lift | — (didominasi informasi waktu) | RFM LogReg: 6.1–6.7× di top 1%, ~2× di top 10% |
| Status | ⚠️ Sinyal didominasi periode pembelian (uji placebo) | ✅ Signifikan mengalahkan aturan "recency saja" di 3/3 periode |
| Dipakai API | Ya | Belum (perlu skema request baru) |

### v2 — Repeat Purchase (Snapshot) · `train_snapshot_pipeline.py` · NB06

Sumber: `outputs/metrics_snapshot.json`. Test snapshot 1 Juni 2018: 77,808 customer, 440 repeater (base rate 0.57%).

| Model | ROC-AUC | PR-AUC | Precision top 1% | Lift top 1% | Lift top 10% |
|---|---|---|---|---|---|
| Recency rule (hanya timing) | 0.582 | 0.0080 | 0.77% | 1.36× | 1.86× |
| RFM Logistic Regression | 0.602 | 0.0188 | 3.60% | 6.36× | 2.25× |
| **XGBoost** | **0.619** | **0.0192** | **3.86%** | **6.82×** | **2.55×** |

- XGBoost vs recency rule: selisih PR-AUC **+0.011, CI 95% [0.004, 0.022]** — model menangkap sinyal di luar timing
- XGBoost vs RFM LogReg: +0.0005, CI 95% [−0.005, 0.005] — tidak signifikan
- Sinyal utama: pernah order > 1 kali (repeat rate 3.4× base), recency yang baru, review tinggi. Ongkir berpengaruh kecil

#### Backtesting — 3 periode test (expanding window)

Sumber: `outputs/backtest_snapshot.csv` (`python train_snapshot_pipeline.py --backtest`). Setiap fold mengulang pemilihan hyperparameter, refit, dan evaluasi hanya dengan data sebelum periode test-nya.

| Test snapshot | Repeater | Recency rule (PR-AUC · lift top 1%) | RFM LogReg | XGBoost |
|---|---|---|---|---|
| Des 2017 | 312 | 0.0102 · 2.25× | **0.0179 · 6.10×** | 0.0158 · 4.81× |
| Mar 2018 | 403 | 0.0098 · 1.74× | **0.0249 · 6.70×** | 0.0207 · 5.96× |
| Jun 2018 | 440 | 0.0080 · 1.36× | 0.0188 · 6.36× | **0.0192 · 6.82×** |
| **Rata-rata** | | 1.78× | **6.39×** (PR-AUC 3.0× base rate) | 5.86× (PR-AUC 2.8× base rate) |

- XGBoost signifikan mengalahkan recency rule di **3/3** periode → sinyal di luar timing stabil
- XGBoost **tidak pernah** signifikan lebih baik dari RFM LogReg; LogReg lebih tinggi di 2/3 periode dan lebih stabil (lift top 1% 6.1–6.7× vs 4.8–6.8×)
- Jun 2018 ternyata periode terbaik XGBoost — hanya melihat satu test snapshot akan melebih-lebihkan keunggulannya
- **RFM Logistic Regression direkomendasikan** sebagai model produksi: setara atau lebih baik dari XGBoost di semua periode, lebih stabil, dan interpretable

### v1 — Churn (desain lama) · `train_pipeline.py` · NB02–NB04

Semua angka di bawah dihasilkan oleh `train_pipeline.py` (sumber: `outputs/metrics.json`) pada **test set 20% yang tidak dipakai untuk training maupun pemilihan threshold**.

| Metrik | Nilai |
|---|---|
| **Champion Model** | XGBoost (tanpa class reweighting) |
| **ROC-AUC (test)** | 0.8207 |
| **CV AUC (5-fold, train+val)** | 0.8213 ± 0.0087 |
| **Threshold** | 0.900 — Youden's J, dipilih di validation set |
| **Churn Recall / Precision** | 69% / 96.7% |
| **Non-Churn Recall** | 79% |
| **Lift — 10% skor risiko terendah** | 40.4% customer kembali vs base rate 10.1% → 4.0× (lihat catatan uji placebo) |
| **Fitur SHAP terbesar** | `avg_freight` — tetapi ongkir tinggi justru **menurunkan** prediksi churn |
| **Leakage `first_order_month`** | ✅ AUC 0.973 → 0.801 (Random Forest) terdokumentasi |
| **Uji placebo** | ⚠️ Fitur yang sama menebak jendela waktu pembelian acak dengan AUC 0.68–0.80 |

Baseline naif "tandai semua customer sebagai churn" sudah mendapat precision 89.9%, jadi metrik utama yang dipakai adalah AUC dan lift, bukan accuracy atau churn recall.

> ⚠️ **Status validitas v1:** label churn identik dengan "order terakhir sebelum 19 Juli 2018", dan uji placebo menunjukkan sebagian besar AUC berasal dari kemampuan fitur mengenali *periode* pembelian. Selain itu, data order praktis berhenti di akhir Agustus 2018, sehingga "90 hari terakhir" hanya berisi ~43 hari data. Model v1 dipertahankan untuk dokumentasi dan untuk API yang sudah ada; untuk keputusan targeting gunakan **v2**.

Notebook NB01–NB06 sudah dijalankan ulang dengan semua perbaikan. Angka di NB04 identik dengan `outputs/metrics.json`, dan angka di NB06 identik dengan `outputs/metrics_snapshot.json`.

---

## Struktur Proyek

```
olist-churn-prediction/
│
├── notebooks
|   ├── 📓 01_data_loading.ipynb          # Load, merge 7 tabel, validasi
|   ├── 📓 02_eda.ipynb                   # EDA, definisi churn, freight analysis
|   ├── 📓 03_feature_engineering.ipynb   # Feature matrix, multikolinearitas
|   ├── 📓 04_modeling.ipynb              # Benchmark, leakage fix, SHAP, threshold, lift, uji placebo
|   ├── 📓 05_final_report.ipynb          # Ringkasan eksekutif ← baca ini dulu
|   └── 📓 06_snapshot_redesign.ipynb     # v2: label snapshot, out-of-time test, uji signifikansi
│
├── train_pipeline.py                 # CLI training v1 (churn) — dipakai API
├── train_snapshot_pipeline.py        # CLI training v2 (repeat purchase, snapshot)
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
│   │   ├── engineer.py               # v1: churn labeling & feature engineering
│   │   └── snapshot.py               # v2: snapshot builder (fitur < T, label di [T, T+90))
│   └── models/
│       ├── train.py                  # v1: XGBoost training pipeline
│       ├── evaluate.py               # v1: SHAP, threshold tuning, lift
│       ├── snapshot_model.py         # v2: XGBoost + baseline, metrik ranking, bootstrap
│       └── backtest.py               # v2: backtest expanding window di 3 periode test
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
- `outputs/metrics.json` — ROC-AUC, CV AUC, F1, threshold, confusion matrix & lift di test set, baseline mean fitur untuk drift monitoring
- `outputs/figures/shap_summary.png` — SHAP summary plot

Model v2 (repeat purchase, snapshot):

```bash
python train_snapshot_pipeline.py --raw-path data/raw --output-dir outputs
```

Output: `outputs/model_snapshot_rfm_logreg.pkl`, `outputs/model_snapshot_xgboost.pkl`, `outputs/metrics_snapshot.json`.

Tambahkan `--backtest` untuk menjalankan backtest 3 periode (sekitar 5–10 menit), yang menghasilkan `outputs/backtest_snapshot.csv` dan `outputs/backtest_snapshot_significance.csv`.

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
  "churn_probability": 0.8596,
  "is_churn": false,
  "threshold_used": 0.9004,
  "model_version": "1.0.0"
}
```

`log_monetary` dan `log_frequency` dihitung di server, jadi tidak perlu dikirim. Probabilitas tidak di-reweight, sehingga nilainya berada di sekitar base rate churn (~0.9). Karena itu threshold juga tinggi.

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
| GET | `/monitoring/drift` | Mean fitur di rolling window vs baseline training |
| GET | `/monitoring/recent` | N prediksi terakhir |

---

## Temuan Kritis 1: Data Leakage `first_order_month`

Model pertama menghasilkan AUC **0.9728** (Random Forest) — mencurigakan. Investigasi menemukan `first_order_month` mendominasi feature importance dengan score 0.55.

Penyebabnya: customer yang first order-nya dekat cutoff Oktober 2018 otomatis non-churn (belum sempat churn karena dataset berakhir). Ini artefak temporal, bukan sinyal perilaku.

Setelah drop `first_order_month` dan `first_order_dayofweek` (hasil NB04):

| Model | AUC Sebelum | AUC Sesudah |
|---|---|---|
| Random Forest | 0.9728 | 0.8005 |
| Decision Tree | 0.9633 | 0.7748 |
| Logistic Regression | 0.7127 | 0.6021 |
| **XGBoost (champion)** | — | **0.8207** |

---

## Temuan Kritis 2: Uji Placebo — Leakage Waktu Lewat Fitur Lain

Karena tanggal referensi tetap (17 Okt 2018), label `is_churned` identik dengan *"order terakhir sebelum 19 Juli 2018"*. Selain itu, 96.3% customer non-churn adalah one-time buyer yang **kebetulan baru berbelanja**.

Untuk menguji apakah model hanya mengenali *kapan* customer berbelanja, XGBoost yang sama dilatih untuk menebak jendela waktu pembelian **lain** (label palsu):

| Jendela order terakhir | ROC-AUC |
|---|---|
| **Churn asli** (19 Jul – 17 Okt 2018) | **0.826** |
| Placebo A (20 Apr – 19 Jul 2018) | 0.804 |
| Placebo C (20 Jul – 18 Okt 2017) | 0.738 |
| Placebo B (20 Jan – 20 Apr 2018) | 0.675 |

Fitur yang sama menebak periode pembelian acak dengan AUC 0.68–0.80 — hanya sedikit di bawah AUC churn. **Sebagian besar kemampuan model berasal dari mengenali periode pembelian, bukan niat customer untuk kembali.** Ini juga berarti decile "40.4% customer kembali" pada dasarnya berisi customer yang belanja di periode akhir dataset.

Bukti pendukung dari EDA (NB02–NB03):
- Rata-rata ongkir bulanan berkorelasi **−0.44** dengan churn rate bulanan
- Porsi pembayaran debit card melonjak dari ~1% menjadi 4% di Q3 2018 — periode yang hampir pasti terlabel non-churn

---

## Koreksi v1.3: Evaluasi & Bug Fitur

Audit ulang menemukan beberapa masalah di versi sebelumnya. Semuanya sudah diperbaiki di `src/`, dicakup oleh unit test, dan diterapkan juga di notebook.

| Masalah | Dampak | Perbaikan |
|---|---|---|
| `scale_pos_weight=9` di `src/models/train.py`, padahal kelas positif (churn) adalah **mayoritas** 90% | Bobot 9× ke kelas mayoritas; model menandai hampir semua customer sebagai churn | Tanpa reweighting (opsi `class_weighting="balanced"` = neg/pos ≈ 0.11). Tanpa reweighting sedikit lebih baik: AUC 0.8207 vs 0.8189 |
| Threshold dipilih di **test set** dengan memaksimalkan F1 kelas churn | Metrik optimistis; model menandai ~97–99% customer sebagai churn | Split train/val/test 60/20/20; threshold = Youden's J di validation; semua metrik dilaporkan di test |
| Angka README/NB05 lama mencampur dua model | AUC 0.8365 & recall 75/76% berasal dari XGBoost di threshold 0.5, sedangkan threshold 0.470 & churn recall 99.2% berasal dari **Random Forest** | Semua angka sekarang dari satu model dan satu sumber (`metrics.json` = NB04) |
| `total_items` = **sum** `order_item_id` (nomor urut item) | Order berisi 3 item terhitung 6 | `count` |
| `payment_value`, `review_score`, dan `payment_type` berulang di setiap baris item | `monetary`, `total_payment_value`, dan `review_count` terhitung ganda untuk order multi-item (rata-rata monetary BRL 213 → 166.59) | Agregasi per order dulu (`order_level()`), baru per customer |
| Revenue per kategori di EDA menjumlahkan `payment_value` per item | Revenue kategori tergelembung | Revenue dihitung dari `price` per item |
| Imputasi median di seluruh dataset sebelum split | Statistik test bocor ke train | Imputasi hanya oleh `SimpleImputer` di pipeline (fit di train) |
| `train_pipeline.py` menimpa `metrics.json` tanpa `feature_means_train` | Endpoint drift kehilangan baseline | `feature_means_train` ditulis ulang setiap training |

### Kenapa lift di sisi "kembali", bukan di sisi "churn"?

Dengan base rate churn 89.9%, lift churn maksimum yang mungkin hanya 1/0.899 ≈ 1.11× (model mencapai 1.107× di decile teratas). Sisi yang lebih informatif adalah kebalikannya: 10% customer dengan skor risiko terendah berisi 40.4% customer yang kembali, dibandingkan base rate 10.1%. Namun, karena hasil uji placebo, angka ini **belum bisa** diartikan sebagai kemampuan menemukan customer yang berniat kembali.

---

## Key Insights dari EDA & SHAP

| Fitur | Arah (SHAP) | Interpretasi |
|---|---|---|
| `avg_freight` ↑ | → churn **↓** (tidak monoton; risiko tertinggi di Q2) | Hipotesis "ongkir tinggi → churn" **tidak didukung**. Di EDA, Q4 ongkir punya churn terendah (87.7%), juga di dalam SP maupun non-SP |
| `customer_state_SP` = True | → churn ↓ | Customer SP berisiko lebih rendah (churn 88.3% vs 91.1% non-SP) |
| `dominant_payment_type_debit_card` = True | → churn ↓ | Kemungkinan besar **proksi waktu**, bukan perilaku loyal (lonjakan debit di Q3 2018) |
| `dominant_category_toys` = True | → churn ↑ | Pembeli mainan cenderung one-time buyer musiman |

> **Temuan penting:** `debit_card` adalah fitur #1 di feature importance XGBoost, dan SHAP menunjukkan arahnya negatif. Tetapi SHAP menjelaskan **perilaku model**, bukan sebab-akibat — jika fitur adalah proksi waktu, SHAP tetap menampilkannya sebagai fitur penting.

---

## Business Recommendations

Rekomendasi diurutkan berdasarkan kekuatan bukti. Detail lengkap ada di NB05.

**Prioritas 1 — Second Purchase Activation (bukti kuat)**
Voucher 15% untuk order kedua, dikirim D+14 setelah first order, dengan holdout group 50%. 96.9% customer tidak pernah kembali untuk kali kedua — masalah utamanya activation, bukan retention.

**Prioritas 2 — Free Shipping sebagai Eksperimen (bukti lemah)**
EDA dan SHAP **tidak** menunjukkan ongkir tinggi menaikkan churn. Program gratis ongkir boleh diuji lewat A/B test (metrik: repeat purchase rate 90 hari dan biaya subsidi per customer yang kembali), tetapi tidak boleh diasumsikan berhasil.

**Prioritas 3 — Segmentasi Geografis**
Gunakan wilayah (SP vs non-SP) sebagai strata dalam eksperimen — customer non-SP berisiko lebih tinggi secara konsisten.

**Prioritas 4 — Segmentasi Kategori**
Arahkan insentif next-purchase ke kategori habis pakai (`health_beauty`, `bed_bath_table`), bukan ke pembeli `toys` yang musiman.

**Tidak direkomendasikan:** cashback khusus debit card — sinyal "debit lebih loyal" kemungkinan besar artefak waktu.

Skenario ukuran dampak (gross, 1 order tambahan per customer yang kembali, rata-rata nilai order BRL 161, **belum dikurangi biaya voucher/subsidi**):

| Customer yang kembali per 10,000 customer yang ditargetkan | Customer | Revenue gross |
|---|---|---|
| 2% | 200 | ~BRL 32K |
| 5% | 500 | ~BRL 81K |
| 10% | 1,000 | ~BRL 161K |

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
- **v1 — Label = tanggal order terakhir (kritis):** uji placebo di NB04 membuktikan fitur yang berubah seiring waktu (ongkir, metode bayar, kategori) menjadi proksi tanggal order. Sudah diatasi dengan desain snapshot di v2
- **v2 — Sinyal lemah & sampel positif kecil:** hanya 312–618 repeater per bagian, sehingga CI bootstrap masih lebar. Backtest dibatasi 3 periode karena panjang data (order praktis berhenti akhir Agustus 2018)
- **v2 — Bukan model uplift:** model memprediksi siapa yang kembali secara alami, bukan siapa yang terpengaruh kampanye. Alokasi voucher sebaiknya didasarkan pada A/B test + uplift modeling
- **API masih v1:** migrasi ke model v2 membutuhkan skema request baru (`recency_days`, `tenure_days`, dll.)

---

## Dataset

**Sumber:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)  
**Lisensi:** CC BY-NC-SA 4.0  
**Periode data:** September 2016 – Oktober 2018 (transaksi praktis berhenti akhir Agustus 2018)  
**Ukuran:** ~100K orders, 96K unique customers

> Dataset ini adalah data historis yang dirilis publik oleh Olist dan tidak diperbarui lagi. Tahun 2016–2018 adalah periode transaksi di dalam data, bukan waktu proyek dikerjakan.

---

*Farah Aulia Kirana — Churn Prediction Portfolio Project · dikerjakan 2026*