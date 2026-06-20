"""
streamlit_app.py
Dashboard Streamlit untuk Churn Prediction API.
Taruh file ini di root folder proyek (sejajar dengan src/ dan tests/).

Jalankan lokal:
    streamlit run streamlit_app.py

Deploy ke Streamlit Cloud / HuggingFace Spaces — set environment variable:
    API_URL=https://your-api.onrender.com

Default fallback: http://127.0.0.1:8000 (local dev)
"""

import math
import os

import pandas as pd
import requests
import streamlit as st

# ── API URL — satu baris, bisa di-override tanpa edit file ───────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 0.5rem;
    }
    .metric-card.danger  { border-left-color: #e05252; }
    .metric-card.success { border-left-color: #52c97a; }
    .metric-card.warning { border-left-color: #f5a623; }
    .metric-label { color: #8b92a5; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
    .metric-value { color: #f0f2f6; font-size: 2rem; font-weight: 700; }
    .metric-sub   { color: #8b92a5; font-size: 0.8rem; margin-top: 2px; }
    .status-ok    { color: #52c97a; font-weight: 600; }
    .status-err   { color: #e05252; font-weight: 600; }
    .section-title {
        font-size: 1.05rem; font-weight: 600; color: #c8cdd8;
        margin: 1.2rem 0 0.6rem;
        border-bottom: 1px solid #2a2f45; padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: health check (cached 10 detik) ───────────────────────────────────
@st.cache_data(ttl=10)
def get_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception:
        return None


# ── Helper: build payload dari form values ────────────────────────────────────
def build_payload(
    customer_id, avg_payment_value, total_payment_value,
    avg_installments, max_installments, avg_price, total_price,
    avg_freight, freight_ratio, avg_review_score, min_review_score,
    review_count, total_items, total_orders, avg_order_value,
    frequency, monetary, customer_lifetime_days,
    customer_state, dominant_payment_type, dominant_category,
):
    min_review_score = min(float(min_review_score), float(avg_review_score))
    review_gap = round(float(avg_review_score) - min_review_score, 4)

    return {
        "customer_id":           customer_id or None,
        "avg_payment_value":     float(avg_payment_value),
        "total_payment_value":   float(total_payment_value),
        "avg_installments":      float(avg_installments),
        "max_installments":      int(max_installments),
        "avg_price":             float(avg_price),
        "total_price":           float(total_price),
        "avg_freight":           float(avg_freight),
        "freight_ratio":         float(freight_ratio),
        "avg_review_score":      float(avg_review_score),
        "min_review_score":      min_review_score,
        "review_gap":            review_gap,
        "review_count":          int(review_count),
        "total_items":           int(total_items),
        "total_orders":          int(total_orders),
        "avg_order_value":       float(avg_order_value),
        "frequency":             int(frequency),
        "monetary":              float(monetary),
        "customer_lifetime_days": int(customer_lifetime_days),
        "customer_state":        customer_state,
        "dominant_payment_type": dominant_payment_type,
        "dominant_category":     dominant_category,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Churn Prediction")
    st.markdown("---")

    health = get_health()
    if health:
        st.markdown('<span class="status-ok">● API Online</span>', unsafe_allow_html=True)
        st.caption(f"Model v{health.get('model_version','–')} · Threshold {health.get('threshold','–')}")
    else:
        st.markdown('<span class="status-err">● API Offline</span>', unsafe_allow_html=True)
        st.caption(f"Target: `{API_URL}`")
        st.caption("Set env var `API_URL` untuk mengubah target.")

    st.markdown("---")
    page = st.radio("Navigasi", ["🔍 Prediksi Single", "📦 Prediksi Batch (CSV)", "📊 Monitoring"])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Predict
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Prediksi Single":
    st.title("🔍 Prediksi Satu Customer")
    st.caption("Isi data customer di bawah, lalu klik **Prediksi**.")

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="section-title">💳 Payment</div>', unsafe_allow_html=True)
            avg_payment_value     = st.number_input("Avg Payment Value (BRL)", min_value=0.0, value=150.0, step=10.0)
            total_payment_value   = st.number_input("Total Payment Value (BRL)", min_value=0.0, value=150.0, step=10.0)
            avg_installments      = st.number_input("Avg Installments", min_value=1.0, value=3.0, step=1.0)
            max_installments      = st.number_input("Max Installments", min_value=1, value=3, step=1)
            dominant_payment_type = st.selectbox(
                "Payment Type",
                ["credit_card", "boleto", "debit_card", "voucher", "not_defined", "unknown"]
            )

        with c2:
            st.markdown('<div class="section-title">📦 Order & Produk</div>', unsafe_allow_html=True)
            avg_price       = st.number_input("Avg Price (BRL)", min_value=0.0, value=120.0, step=10.0)
            total_price     = st.number_input("Total Price (BRL)", min_value=0.0, value=120.0, step=10.0)
            avg_freight     = st.number_input("Avg Freight (BRL)", min_value=0.0, value=30.0, step=5.0)
            freight_ratio   = st.number_input("Freight Ratio", min_value=0.0, max_value=2.0, value=0.2, step=0.05)
            total_items     = st.number_input("Total Items", min_value=1, value=1, step=1)
            total_orders    = st.number_input("Total Orders", min_value=1, value=1, step=1)
            avg_order_value = st.number_input("Avg Order Value (BRL)", min_value=0.0, value=150.0, step=10.0)
            dominant_category = st.selectbox(
                "Kategori Dominan",
                ["health_beauty", "bed_bath_table", "sports_leisure", "furniture_decor",
                 "computers_accessories", "housewares", "telephony", "toys",
                 "watches_gifts", "auto", "other"]
            )

        with c3:
            st.markdown('<div class="section-title">⭐ Review & Lainnya</div>', unsafe_allow_html=True)

            avg_review_score = st.number_input(
                "Avg Review Score (1–5)", min_value=1.0, max_value=5.0, value=4.0, step=0.1
            )
            min_review_score = st.number_input(
                "Min Review Score (1–5)", min_value=1.0, max_value=5.0, value=4.0, step=0.1
            )
            min_review_clamped = min(float(min_review_score), float(avg_review_score))
            review_gap_preview = round(float(avg_review_score) - min_review_clamped, 4)
            st.caption(f"Review Gap (auto-hitung): **{review_gap_preview}**")
            if min_review_score > avg_review_score:
                st.warning("⚠️ Min Review Score lebih besar dari Avg — akan otomatis di-clamp ke Avg saat prediksi.")

            review_count           = st.number_input("Review Count", min_value=0, value=1, step=1)
            frequency              = st.number_input("Frequency (order unik)", min_value=1, value=1, step=1)
            monetary               = st.number_input("Monetary (BRL)", min_value=0.0, value=150.0, step=10.0)
            customer_lifetime_days = st.number_input("Customer Lifetime (hari)", min_value=0, value=0, step=30)
            customer_id            = st.text_input("Customer ID (opsional)", value="cust_001")
            customer_state         = st.selectbox(
                "State",
                ["SP","RJ","MG","BA","PR","RS","PE","CE","GO","PA",
                 "SC","MA","ES","PB","PI","RN","AL","MT","MS","DF",
                 "SE","AM","RO","AC","TO","AP","RR"]
            )

        submitted = st.form_submit_button("🚀 Prediksi", use_container_width=True)

    if submitted:
        payload = build_payload(
            customer_id, avg_payment_value, total_payment_value,
            avg_installments, max_installments, avg_price, total_price,
            avg_freight, freight_ratio, avg_review_score, min_review_score,
            review_count, total_items, total_orders, avg_order_value,
            frequency, monetary, customer_lifetime_days,
            customer_state, dominant_payment_type, dominant_category,
        )
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            r.raise_for_status()
            resp     = r.json()
            prob     = resp["churn_probability"]
            is_churn = resp["is_churn"]

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                card_class = "danger" if is_churn else "success"
                label      = "🔴 CHURN" if is_churn else "🟢 RETAINED"
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <div class="metric-label">Status</div>
                    <div class="metric-value">{label}</div>
                    <div class="metric-sub">Threshold {resp['threshold_used']}</div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                prob_class = "danger" if prob > 0.7 else "warning" if prob > 0.47 else "success"
                st.markdown(f"""
                <div class="metric-card {prob_class}">
                    <div class="metric-label">Churn Probability</div>
                    <div class="metric-value">{prob*100:.1f}%</div>
                    <div class="metric-sub">Model v{resp['model_version']}</div>
                </div>""", unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Customer ID</div>
                    <div class="metric-value" style="font-size:1.2rem">{resp.get('customer_id') or '—'}</div>
                    <div class="metric-sub">XGBoost ROC-AUC 0.8365</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("##### Churn Probability")
            st.progress(float(prob))

        except requests.HTTPError as e:
            try:
                detail = e.response.json()
                st.error(f"API Error {e.response.status_code}: {detail}")
            except Exception:
                st.error(f"API Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch CSV
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Prediksi Batch (CSV)":
    st.title("📦 Prediksi Batch dari CSV")
    st.caption("Upload CSV dengan kolom sesuai skema API. Bisa ribuan baris — diproses per 1000.")

    sample_data = [
        {
            "customer_id": "cust_001", "avg_payment_value": 150, "total_payment_value": 150,
            "avg_installments": 3, "max_installments": 3, "avg_price": 120, "total_price": 120,
            "avg_freight": 30, "freight_ratio": 0.2, "avg_review_score": 4, "min_review_score": 4,
            "review_gap": 0, "review_count": 1, "total_items": 1, "total_orders": 1,
            "avg_order_value": 150, "frequency": 1, "monetary": 150, "customer_lifetime_days": 0,
            "customer_state": "SP", "dominant_payment_type": "credit_card", "dominant_category": "health_beauty"
        },
        {
            "customer_id": "cust_002", "avg_payment_value": 80, "total_payment_value": 240,
            "avg_installments": 1, "max_installments": 1, "avg_price": 60, "total_price": 180,
            "avg_freight": 20, "freight_ratio": 0.25, "avg_review_score": 3, "min_review_score": 2,
            "review_gap": 1, "review_count": 3, "total_items": 3, "total_orders": 3,
            "avg_order_value": 80, "frequency": 3, "monetary": 240, "customer_lifetime_days": 180,
            "customer_state": "RJ", "dominant_payment_type": "boleto", "dominant_category": "sports_leisure"
        },
    ]
    template_df = pd.DataFrame(sample_data)
    st.download_button(
        "⬇️ Download Template CSV",
        template_df.to_csv(index=False).encode(),
        "template_churn.csv",
        "text/csv",
    )

    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV kamu", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**{len(df):,} baris** terdeteksi. Preview 5 baris pertama:")
        st.dataframe(df.head(5), use_container_width=True)

        required_cols = [
            "avg_payment_value","total_payment_value","avg_installments","max_installments",
            "avg_price","total_price","avg_freight","freight_ratio",
            "avg_review_score","min_review_score","review_gap","review_count",
            "total_items","total_orders","avg_order_value","frequency","monetary",
            "customer_lifetime_days","customer_state","dominant_payment_type","dominant_category"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Kolom tidak ditemukan: `{'`, `'.join(missing)}`")
            st.stop()

        df["min_review_score"] = df.apply(
            lambda r: min(float(r["min_review_score"]), float(r["avg_review_score"])), axis=1
        )
        df["review_gap"] = (df["avg_review_score"] - df["min_review_score"]).round(4)

        if st.button("🚀 Jalankan Prediksi Batch", use_container_width=True):
            all_results = []
            n_chunks    = math.ceil(len(df) / 1000)
            progress    = st.progress(0)
            status      = st.empty()
            errors      = []

            int_cols = ["max_installments","review_count","total_items",
                        "total_orders","frequency","customer_lifetime_days"]

            for i in range(n_chunks):
                chunk     = df.iloc[i*1000:(i+1)*1000].copy()
                customers = chunk.to_dict(orient="records")

                for c in customers:
                    for col in int_cols:
                        if col in c and c[col] is not None:
                            c[col] = int(c[col])
                    for col in ["avg_payment_value","total_payment_value","avg_installments",
                                "avg_price","total_price","avg_freight","freight_ratio",
                                "avg_review_score","min_review_score","review_gap",
                                "avg_order_value","monetary"]:
                        if col in c and c[col] is not None:
                            c[col] = float(c[col])

                try:
                    r = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"customers": customers},
                        timeout=60,
                    )
                    r.raise_for_status()
                    all_results.extend(r.json()["predictions"])
                except Exception as e:
                    errors.append(f"Chunk {i+1}: {e}")

                progress.progress((i+1) / n_chunks)
                status.caption(f"Memproses {min((i+1)*1000, len(df)):,} / {len(df):,} baris...")

            progress.empty()
            status.empty()

            if errors:
                for err in errors:
                    st.error(err)

            if all_results:
                result_df  = pd.DataFrame(all_results)
                total      = len(result_df)
                churn_n    = int(result_df["is_churn"].sum())
                churn_rate = churn_n / total

                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Total Customer</div>
                        <div class="metric-value">{total:,}</div></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-card danger">
                        <div class="metric-label">Predicted Churn</div>
                        <div class="metric-value">{churn_n:,}</div></div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-card success">
                        <div class="metric-label">Retained</div>
                        <div class="metric-value">{total - churn_n:,}</div></div>""", unsafe_allow_html=True)
                with m4:
                    cr_class = "danger" if churn_rate > 0.5 else "warning"
                    st.markdown(f"""<div class="metric-card {cr_class}">
                        <div class="metric-label">Churn Rate</div>
                        <div class="metric-value">{churn_rate*100:.1f}%</div></div>""", unsafe_allow_html=True)

                st.markdown("##### Distribusi Churn Probability")
                hist = (
                    result_df["churn_probability"]
                    .pipe(lambda s: pd.cut(s, bins=10))
                    .value_counts()
                    .sort_index()
                    .rename(index=str)
                )
                st.bar_chart(hist)

                st.markdown("##### Hasil Prediksi")
                display_df = result_df.copy()
                display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x*100:.2f}%")
                display_df["is_churn"]          = display_df["is_churn"].map({True: "🔴 Churn", False: "🟢 Retained"})
                st.dataframe(display_df, use_container_width=True, height=400)

                st.download_button(
                    "⬇️ Download Hasil CSV",
                    result_df.to_csv(index=False).encode(),
                    "hasil_prediksi.csv",
                    "text/csv",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Monitoring
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Monitoring":
    st.title("📊 Monitoring & Drift Detection")

    col_left, _ = st.columns([1, 3])
    with col_left:
        window = st.slider("Window (jam)", 1, 720, 24)
        st.button("🔄 Refresh")

    try:
        drift  = requests.get(f"{API_URL}/monitoring/drift",  params={"window_hours": window}, timeout=5).json()
        recent = requests.get(f"{API_URL}/monitoring/recent", params={"limit": 100},            timeout=5).json()
    except Exception as e:
        st.error(f"Tidak bisa konek ke API: {e}")
        st.caption(f"Target API: `{API_URL}` — set env var `API_URL` untuk mengubah.")
        st.stop()

    st.markdown("---")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Request ({window}h terakhir)</div>
            <div class="metric-value">{drift.get('request_count', 0):,}</div></div>""",
            unsafe_allow_html=True)
    with d2:
        churn_rate = drift.get("churn_rate_recent")
        cr_str     = f"{churn_rate*100:.1f}%" if churn_rate is not None else "—"
        cr_class   = "danger" if churn_rate and churn_rate > 0.5 else "success"
        st.markdown(f"""<div class="metric-card {cr_class}">
            <div class="metric-label">Churn Rate Recent</div>
            <div class="metric-value">{cr_str}</div></div>""",
            unsafe_allow_html=True)
    with d3:
        flagged = drift.get("flagged_features", [])
        f_class = "warning" if flagged else "success"
        f_sub   = "⚠️ " + ", ".join(flagged[:3]) if flagged else "✅ Tidak ada drift"
        st.markdown(f"""<div class="metric-card {f_class}">
            <div class="metric-label">Flagged Features</div>
            <div class="metric-value">{len(flagged)}</div>
            <div class="metric-sub">{f_sub}</div></div>""",
            unsafe_allow_html=True)

    feature_means = drift.get("feature_means", {})
    if feature_means:
        st.markdown("##### Rata-rata Fitur (rolling window)")
        means_df = pd.DataFrame(
            list(feature_means.items()), columns=["Feature", "Mean"]
        ).sort_values("Mean", ascending=False)
        st.dataframe(means_df, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada data fitur dalam window ini.")

    drift_flags = drift.get("drift_flags", {})
    if drift_flags:
        st.markdown("##### Drift per Fitur (vs Training Baseline)")
        flags_rows = [
            {
                "Feature":     feat,
                "Train Mean":  info.get("train_mean"),
                "Recent Mean": info.get("recent_mean"),
                "% Change":    f"{info.get('pct_change', 0):.2f}%",
                "Status":      "⚠️ Drift" if info.get("flagged") else "✅ Normal",
            }
            for feat, info in drift_flags.items()
        ]
        st.dataframe(pd.DataFrame(flags_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Drift comparison belum tersedia — `metrics.json` perlu berisi `feature_means_train`.")

    preds = recent.get("predictions", [])
    if preds:
        st.markdown("##### Prediksi Terbaru (100 terakhir)")
        rec_df = pd.DataFrame(preds)
        rec_df["is_churn"]          = rec_df["is_churn"].map({1: "🔴 Churn", 0: "🟢 Retained"})
        rec_df["churn_probability"] = rec_df["churn_probability"].map(lambda x: f"{x*100:.2f}%")
        st.dataframe(rec_df, use_container_width=True, height=350, hide_index=True)
    else:
        st.info("Belum ada prediksi tercatat. Lakukan prediksi dulu di halaman lain.")