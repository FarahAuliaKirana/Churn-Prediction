"""
streamlit_app.py
Dashboard untuk model repeat purchase (v2).

Halaman:
    Overview              — cerita project, bukti validasi, rekomendasi, grafik pendukung
    Score a customer      — peluang satu customer belanja lagi dalam 90 hari + contoh siap klik
    Score a customer list — data demo (customer asli Jun 2018) atau upload CSV → daftar prioritas

Jalankan lokal (API harus berjalan untuk halaman scoring):
    uvicorn src.api.main:app --reload
    streamlit run streamlit_app.py

Target API bisa diganti dengan env var API_URL (default http://127.0.0.1:8000).
Halaman Overview tidak butuh API: datanya dibaca dari outputs/ dan r/.
Halaman bisa dibuka langsung lewat URL: ?page=overview | score | list
"""
from __future__ import annotations

import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
ROOT = Path(__file__).parent
REPO_URL = "https://github.com/FarahAuliaKirana/Churn-Prediction"

ACCENT = "#0F766E"
INK = "#1F2933"
MUTED = "#52606D"
SOFT = "#B8C2CC"
MODEL_COLORS = {"RFM logistic regression": ACCENT, "XGBoost": MUTED, "Recency-only rule": SOFT}
MODEL_LABELS = {"rfm_logreg": "RFM logistic regression", "xgboost": "XGBoost",
                "recency_rule": "Recency-only rule"}
SEGMENT_ORDER = ["top1pct", "top5pct", "top10pct", "rest"]
# Segmen API bersifat eksklusif: top5pct = peringkat 1–5% (tidak termasuk top 1%), dst.
SEGMENT_TEXT = {"top1pct": "Top 1%", "top5pct": "Top 1–5%", "top10pct": "Top 5–10%",
                "rest": "Below the top 10%"}
PERIOD_LABELS = {"2017-12-01": "Dec 2017", "2018-03-01": "Mar 2018", "2018-06-01": "Jun 2018"}
PAGES = {"overview": "Overview", "score": "Score a customer", "list": "Score a customer list"}

# Contoh customer realistis (rentang nilai umum di data Olist) untuk halaman Score
EXAMPLES = {
    "Loyal repeat buyer":     {"frequency": 3, "recency": 25,  "tenure": 300, "monetary": 450.0},
    "Recent one-time buyer":  {"frequency": 1, "recency": 10,  "tenure": 10,  "monetary": 120.0},
    "Dormant one-time buyer": {"frequency": 1, "recency": 250, "tenure": 250, "monetary": 90.0},
}

st.set_page_config(page_title="Olist Repeat Purchase", layout="wide")

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
    h1 {{ font-weight: 700; letter-spacing: -0.01em; }}
    .lede {{ font-size: 1.08rem; color: {MUTED}; max-width: 780px; margin-top: -0.4rem; }}
    .kpi {{ border: 1px solid #E3E5DE; border-radius: 8px; padding: 1rem 1.1rem; background: #FFFFFF;
            height: 100%; }}
    .kpi-value {{ font-size: 2.1rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
    .kpi-value.accent {{ color: {ACCENT}; }}
    .kpi-text {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.35rem; }}
    .proof {{ border-left: 3px solid {ACCENT}; padding: 0.6rem 0 0.6rem 1rem; margin: 1.2rem 0 0.4rem;
              font-size: 1.05rem; color: {INK}; }}
    .rec {{ border: 1px solid #E3E5DE; border-radius: 8px; padding: 0.9rem 1rem; background: #FFFFFF;
            height: 100%; }}
    .rec-title {{ font-weight: 600; color: {INK}; margin-bottom: 0.3rem; }}
    .rec-text {{ color: {MUTED}; font-size: 0.93rem; }}
    .result {{ border: 1px solid #E3E5DE; border-radius: 8px; padding: 1.2rem 1.3rem; background: #FFFFFF; }}
    .result-big {{ font-size: 2.6rem; font-weight: 700; color: {ACCENT}; line-height: 1.05; }}
    .result-line {{ font-size: 1.05rem; color: {INK}; margin-top: 0.5rem; }}
    .note {{ color: {MUTED}; font-size: 0.9rem; }}
    .footer {{ color: {MUTED}; font-size: 0.88rem; border-top: 1px solid #E3E5DE;
               padding-top: 0.8rem; margin-top: 2rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data — semua dari file output project
# ---------------------------------------------------------------------------

@st.cache_data
def load_backtest() -> pd.DataFrame | None:
    fp = ROOT / "outputs" / "backtest_snapshot.csv"
    return pd.read_csv(fp) if fp.exists() else None


@st.cache_data
def load_snapshot_metrics() -> dict | None:
    fp = ROOT / "outputs" / "metrics_snapshot.json"
    return json.loads(fp.read_text()) if fp.exists() else None


@st.cache_data
def load_km_curve() -> pd.DataFrame | None:
    fp = ROOT / "r" / "km_curve.csv"
    return pd.read_csv(fp) if fp.exists() else None


@st.cache_data
def load_ab_summary() -> dict | None:
    fp = ROOT / "r" / "ab_summary.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    return dict(zip(df["key"], df["value"]))


@st.cache_data
def load_demo_customers() -> pd.DataFrame | None:
    fp = ROOT / "outputs" / "demo_customers_2018-06.csv"
    return pd.read_csv(fp) if fp.exists() else None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def get_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except requests.RequestException:
        return None


def api_offline_message() -> None:
    st.error(
        f"The scoring API is not reachable at `{API_URL}`. Start it with "
        "`uvicorn src.api.main:app --reload`, or set `API_URL` to where it runs."
    )


@st.cache_data(ttl=300, show_spinner=False)
def score_batch(customers: tuple[tuple, ...]) -> list[dict]:
    """Panggil /v2/repeat/score/batch per 1000 baris (input tuple agar bisa di-cache)."""
    rows = [dict(zip(("customer_id", "recency_days", "frequency", "monetary", "tenure_days"), c))
            for c in customers]
    out = []
    for i in range(0, len(rows), 1000):
        r = requests.post(f"{API_URL}/v2/repeat/score/batch",
                          json={"customers": rows[i:i + 1000]}, timeout=60)
        r.raise_for_status()
        out.extend(r.json()["predictions"])
    return out


def to_tuples(df: pd.DataFrame) -> tuple[tuple, ...]:
    return tuple(
        (str(r.customer_id), int(r.recency_days), int(r.frequency), float(r.monetary), int(r.tenure_days))
        for r in df.itertuples(index=False)
    )


def pct(x: float, digits: int = 2) -> str:
    """Persentase dengan pembulatan half-up (0.565% → 0.57%), konsisten dengan README & notebook."""
    q = Decimal(1).scaleb(-digits)
    return f"{Decimal(str(round(x * 100, 10))).quantize(q, rounding=ROUND_HALF_UP)}%"


def kpi(value: str, text: str, accent: bool = False) -> str:
    return (f'<div class="kpi"><div class="kpi-value{" accent" if accent else ""}">{value}</div>'
            f'<div class="kpi-text">{text}</div></div>')


def footer() -> None:
    st.markdown(
        f'<div class="footer">Built by Farah Aulia Kirana · '
        f'<a href="{REPO_URL}">Code and full write-up</a> · '
        f'<a href="{REPO_URL}/blob/main/r/ab_test_design.md">Experiment design (R)</a> · '
        "Data: Olist Brazilian E-Commerce public dataset (CC BY-NC-SA 4.0), orders September 2016 – "
        "August 2018. Historical data; the project was built in 2026.</div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar (halaman bisa dipilih lewat ?page=)
# ---------------------------------------------------------------------------

slug = st.query_params.get("page", "overview")
with st.sidebar:
    st.markdown("### Olist repeat purchase")
    page = st.radio("Page", list(PAGES.values()),
                    index=list(PAGES).index(slug) if slug in PAGES else 0,
                    label_visibility="collapsed")
    st.query_params["page"] = next(k for k, v in PAGES.items() if v == page)
    st.divider()
    health = get_health()
    if health and health.get("repeat_model_loaded"):
        st.caption(f"API connected · model {health.get('repeat_model_version')}")
    elif health:
        st.caption("API connected, but the model is not loaded. "
                   "Run `python train_snapshot_pipeline.py`.")
    else:
        st.caption(f"API offline (`{API_URL}`). The Overview page still works.")


# ═══════════════════════════════════════════════════════════════════════════
# Overview
# ═══════════════════════════════════════════════════════════════════════════
if page == PAGES["overview"]:
    st.title("Who will buy again?")
    st.markdown(
        '<p class="lede">96.9% of Olist customers order only once. This project predicts which '
        "customers are most likely to place another order in the next 90 days, checks that "
        "prediction on months the model never saw, and sizes the A/B test needed to prove that a "
        "second-purchase incentive actually works.</p>",
        unsafe_allow_html=True,
    )

    backtest = load_backtest()
    metrics = load_snapshot_metrics()
    km = load_km_curve()
    ab = load_ab_summary()

    if backtest is None or metrics is None:
        st.info("Model outputs not found. Run `python train_snapshot_pipeline.py --backtest` "
                "to generate `outputs/backtest_snapshot.csv` and `outputs/metrics_snapshot.json`.")
        st.stop()

    lr = backtest[backtest["model"] == "rfm_logreg"]
    k1, k2, k3 = st.columns(3)
    with k1:
        if ab:
            st.markdown(kpi(pct(ab["repeat_rate_90d"]),
                            "of new customers buy again within 90 days of their first order"),
                        unsafe_allow_html=True)
    with k2:
        st.markdown(kpi(f"{lr['lift_top1pct'].mean():.1f}×",
                        "higher repeat rate among the top 1% of scored customers than average "
                        f"(range {lr['lift_top1pct'].min():.1f}–{lr['lift_top1pct'].max():.1f}× "
                        f"across {len(lr)} test periods)", accent=True),
                    unsafe_allow_html=True)
    with k3:
        if ab:
            st.markdown(kpi(f"{ab['n_total_for_20pct_lift'] / 1000:.0f}K",
                            "new customers needed in an A/B test to detect a +20% uplift from a "
                            f"second-purchase incentive (~{ab['months_for_20pct_lift']:.0f} months)"),
                        unsafe_allow_html=True)

    test = metrics["test"]["rfm_logreg"]
    top_n = round(test["n"] * 0.01)
    top_hits = round(test["precision_top1pct"] * top_n)
    st.markdown(
        f'<div class="proof"><b>Checked on customers the model never saw.</b> On 1 June 2018 the model '
        f"ranked {test['n']:,} customers. Of the {top_n:,} it ranked highest, <b>{top_hits} ordered "
        f"again</b> within 90 days ({pct(test['precision_top1pct'], 1)}), against "
        f"{pct(test['base_rate'])} for all customers.</div>",
        unsafe_allow_html=True)

    st.subheader("What to do with this")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(
            '<div class="rec"><div class="rec-title">Win the second order, not loyalty</div>'
            '<div class="rec-text">Almost everyone buys once. A customer who has ordered twice is '
            "about 3× more likely to order again, so the first goal is a second purchase, early: "
            "most returns happen in the first month.</div></div>", unsafe_allow_html=True)
    with r2:
        mde = f"+{ab['mde_relative_3_months'] * 100:.0f}%" if ab else "large"
        st.markdown(
            '<div class="rec"><div class="rec-title">Prove the incentive with an A/B test</div>'
            '<div class="rec-text">A 3-month test can only detect uplifts of about '
            f"{mde}. A 15% voucher pays off only if it doubles repeat purchases at a 30% margin, "
            "so test smaller or pay-on-return incentives.</div></div>", unsafe_allow_html=True)
    with r3:
        seg_text = (f"the top 1% return at {pct(test['precision_top1pct'], 1)} "
                    f"({test['lift_top1pct']:.1f}× average), the top 5% at "
                    f"{pct(test['precision_top5pct'], 1)} ({test['lift_top5pct']:.1f}×)")
        st.markdown(
            '<div class="rec"><div class="rec-title">Use the score for small, focused campaigns</div>'
            f'<div class="rec-text">The score works best at the very top: {seg_text}. For a broad '
            "programme, simple rules perform almost as well.</div></div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Does the model beat simple rules?")
        bt = backtest[backtest["model"].isin(MODEL_LABELS)].copy()
        bt["Model"] = bt["model"].map(MODEL_LABELS)
        bt["Test period"] = bt["test_snapshot"].map(PERIOD_LABELS)
        bars = alt.Chart(bt).mark_bar().encode(
            x=alt.X("Test period:N", title=None, sort=list(PERIOD_LABELS.values()),
                    axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Model:N", sort=list(MODEL_COLORS)),
            y=alt.Y("lift_top1pct:Q", title="Top-1% repeat rate ÷ average"),
            color=alt.Color("Model:N", scale=alt.Scale(domain=list(MODEL_COLORS),
                                                       range=list(MODEL_COLORS.values())),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "Test period",
                     alt.Tooltip("lift_top1pct:Q", title="Lift (top 1%)", format=".2f"),
                     alt.Tooltip("pr_auc:Q", title="PR-AUC", format=".4f")],
        )
        baseline = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(
            strokeDash=[4, 4], color=MUTED).encode(y="y:Q")
        st.altair_chart((bars + baseline).properties(height=300), use_container_width=True)
        st.markdown(
            '<p class="note">Each period is a true out-of-time test: the model is trained only on '
            "earlier data. The 4-feature logistic regression is the most stable, so it is the model "
            "the API serves. Dashed line = no better than random.</p>", unsafe_allow_html=True)

    with right:
        st.subheader("When do new customers come back?")
        if km is not None:
            km_plot = km.assign(pct=km["repeat_rate"] * 100, lo=km["ci_low"] * 100,
                                hi=km["ci_high"] * 100)
            band = alt.Chart(km_plot).mark_area(opacity=0.18, color=ACCENT).encode(
                x=alt.X("days:Q", title="Days since first order"),
                y=alt.Y("lo:Q", title="% of customers who ordered again"), y2="hi:Q")
            line = alt.Chart(km_plot).mark_line(color=ACCENT, strokeWidth=2.5).encode(
                x="days:Q", y="pct:Q",
                tooltip=[alt.Tooltip("days:Q", title="Day"),
                         alt.Tooltip("pct:Q", title="Ordered again (%)", format=".2f")])
            mark90 = alt.Chart(pd.DataFrame({"days": [90]})).mark_rule(
                strokeDash=[4, 4], color=MUTED).encode(x="days:Q")
            st.altair_chart((band + line + mark90).properties(height=300), use_container_width=True)
            st.markdown(
                '<p class="note">Kaplan–Meier estimate (R), counting only orders at least one day '
                "after the first. The dashed line marks 90 days.</p>", unsafe_allow_html=True)
        else:
            st.info("Run `Rscript r/render.R` to generate the survival curve.")

    st.subheader("What makes another purchase more likely?")
    coefs = metrics.get("rfm_logreg_coefficients")
    if coefs:
        names = {"log_frequency": "More past orders", "log_recency_days": "More days since last order",
                 "log_tenure_days": "Longer time since first order", "log_monetary": "Higher total spend"}
        cdf = pd.DataFrame({"Factor": [names.get(k, k) for k in coefs], "Effect": list(coefs.values())})
        cdf["Direction"] = cdf["Effect"].map(lambda v: "Raises the chance" if v > 0 else "Lowers the chance")
        coef_chart = alt.Chart(cdf).mark_bar().encode(
            x=alt.X("Effect:Q", title="Effect on the log-odds of buying again (per 1 SD)"),
            y=alt.Y("Factor:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=320, labelOverlap=False, labelFontSize=13)),
            color=alt.Color("Direction:N", scale=alt.Scale(
                domain=["Raises the chance", "Lowers the chance"], range=[ACCENT, SOFT]),
                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Factor", alt.Tooltip("Effect:Q", format=".3f")],
        ).properties(height=alt.Step(38))
        st.altair_chart(coef_chart, use_container_width=True)
        st.markdown(
            '<p class="note">Coefficients of the served logistic regression (features log-transformed '
            "and standardised). These are associations, not causes.</p>", unsafe_allow_html=True)

    with st.expander("How this model replaced the first churn model"):
        st.markdown(
            "- The first churn model scored **ROC-AUC 0.82**, but its label was equivalent to "
            "*“last order before 19 July 2018”*. A placebo test showed the same features predicted "
            "arbitrary purchase-date windows with AUC 0.68–0.80, so it mostly learned **when** people "
            "bought.\n"
            "- The current model uses a snapshot design: features only from orders before a date *T*, "
            "label = a new order in the 90 days after *T*, tested on later periods.\n"
            "- Its ROC-AUC is lower (~0.60), but it answers the right question and beats a "
            "recency-only rule in every test period.")
    footer()


# ═══════════════════════════════════════════════════════════════════════════
# Score a customer
# ═══════════════════════════════════════════════════════════════════════════
elif page == PAGES["score"]:
    st.title("Score a customer")
    st.markdown('<p class="lede">How likely is this customer to order again in the next 90 days, '
                "and should they be in a priority group?</p>", unsafe_allow_html=True)

    metrics = load_snapshot_metrics()
    base_rate = metrics["test"]["rfm_logreg"]["base_rate"] if metrics else None
    segments = sorted(metrics["segments"]["rfm_logreg"], key=lambda s: s["min_score"],
                      reverse=True) if metrics else []

    def apply_example() -> None:
        ex = EXAMPLES.get(st.session_state["example"])
        if ex:
            st.session_state["frequency"] = ex["frequency"]
            st.session_state["recency"] = ex["recency"]
            st.session_state["tenure"] = ex["tenure"]
            st.session_state["monetary"] = ex["monetary"]

    if "frequency" not in st.session_state:
        st.session_state["example"] = "Loyal repeat buyer"
        apply_example()

    st.radio("Start from an example", [*EXAMPLES, "Custom"], key="example", horizontal=True,
             on_change=apply_example)

    inputs, result = st.columns([1, 1.35], gap="large")
    with inputs:
        st.markdown("**Customer history on the scoring date**")
        frequency = st.number_input("Orders placed so far", min_value=1, step=1, key="frequency")
        recency = st.number_input("Days since the last order", min_value=0, step=1, key="recency")
        if frequency == 1:
            st.session_state["tenure"] = int(recency)
        tenure = st.number_input(
            "Days since the first order", min_value=0, step=1, key="tenure",
            disabled=frequency == 1,
            help="For a one-order customer this equals the days since the last order.")
        monetary = st.number_input("Total spent so far (BRL)", min_value=0.0, step=10.0, key="monetary")

    payload = {"customer_id": None, "recency_days": int(recency), "frequency": int(frequency),
               "monetary": float(monetary), "tenure_days": int(tenure)}

    with result:
        if tenure < recency:
            st.warning("Days since the first order cannot be smaller than days since the last order.")
            st.stop()
        try:
            r = requests.post(f"{API_URL}/v2/repeat/score", json=payload, timeout=10)
        except requests.RequestException:
            api_offline_message()
            st.stop()
        if r.status_code == 503:
            st.error("The API is running but the model is not loaded. "
                     "Run `python train_snapshot_pipeline.py`, then restart the API.")
            st.stop()
        if r.status_code == 422:
            st.error(f"The API rejected these inputs: {r.json().get('detail')}")
            st.stop()
        resp = r.json()
        p = resp["repeat_probability"]
        times = p / base_rate if base_rate else None
        seg = resp["segment"]

        seg_line = (
            f"Priority group: <b>{SEGMENT_TEXT[seg]}</b> of customers. Historically, "
            f"{resp['segment_precision'] * 100:.1f}% of this group ordered again "
            f"({resp['segment_lift']:.1f}× the average)."
            if seg != "rest" else
            "Not in a priority group: the score is below the top-10% cut-off."
        )
        st.markdown(
            f'<div class="result"><div class="note">Chance of ordering again in the next 90 days</div>'
            f'<div class="result-big">{p * 100:.2f}%</div>'
            + (f'<div class="result-line">{times:.1f}× the average customer '
               f'({pct(base_rate)}).</div>' if times else "")
            + f'<div class="result-line">{seg_line}</div></div>',
            unsafe_allow_html=True)

        if base_rate and segments:
            refs = pd.DataFrame(
                [{"Label": "Average customer", "Score": base_rate, "Kind": "reference"}]
                + [{"Label": f"{SEGMENT_TEXT[s['segment']]} cut-off", "Score": s["min_score"],
                    "Kind": "reference"} for s in reversed(segments)]
                + [{"Label": "This customer", "Score": p, "Kind": "customer"}])
            refs["Percent"] = refs["Score"] * 100
            st.altair_chart(alt.Chart(refs).mark_bar().encode(
                x=alt.X("Percent:Q", title="Chance of ordering again (%)"),
                y=alt.Y("Label:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color("Kind:N", scale=alt.Scale(domain=["reference", "customer"],
                                                          range=[SOFT, ACCENT]), legend=None),
                tooltip=["Label", alt.Tooltip("Percent:Q", format=".2f")],
            ).properties(height=alt.Step(30), title="Where this customer sits"),
                use_container_width=True)

    st.subheader("What if this customer waits longer before ordering again?")
    days = list(range(0, 366, 15))
    grid = tuple(("what-if", d, int(frequency), float(monetary),
                  d if frequency == 1 else max(int(tenure), d)) for d in days)
    try:
        curve = score_batch(grid)
        cdf = pd.DataFrame({"Days since last order": days,
                            "Chance (%)": [c["repeat_probability"] * 100 for c in curve]})
        line = alt.Chart(cdf).mark_line(color=ACCENT, strokeWidth=2.5).encode(
            x="Days since last order:Q", y="Chance (%):Q",
            tooltip=["Days since last order", alt.Tooltip("Chance (%):Q", format=".2f")])
        now = alt.Chart(pd.DataFrame({"Days since last order": [int(recency)],
                                      "Chance (%)": [p * 100]})).mark_point(
            color=INK, size=90, filled=True).encode(x="Days since last order:Q", y="Chance (%):Q")
        st.altair_chart((line + now).properties(height=260), use_container_width=True)
        st.markdown('<p class="note">Other inputs held constant. The dot is this customer today: '
                    "the longer the silence, the lower the chance of another order.</p>",
                    unsafe_allow_html=True)
    except requests.RequestException:
        st.caption("What-if curve unavailable: the batch endpoint did not respond.")

    st.info("A high score means the customer is likely to return **on their own**. Whether an "
            "incentive changes that can only be measured with an A/B test.")
    footer()


# ═══════════════════════════════════════════════════════════════════════════
# Score a customer list
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.title("Score a customer list")
    st.markdown('<p class="lede">Turn a customer list into a priority list for a campaign. Try it on '
                "real customers the model never saw, or upload your own file.</p>",
                unsafe_allow_html=True)

    required = ["customer_id", "recency_days", "frequency", "monetary", "tenure_days"]
    demo = load_demo_customers()
    source = st.radio("Data", ["Real held-out customers (June 2018 sample)", "Upload a CSV"],
                      horizontal=True, label_visibility="collapsed",
                      index=0 if demo is not None else 1)

    actual = None
    if source.startswith("Real"):
        if demo is None:
            st.info("Demo sample not found. Run `python train_snapshot_pipeline.py` to create "
                    "`outputs/demo_customers_2018-06.csv`.")
            st.stop()
        df = demo[required].copy()
        actual = demo[["customer_id", "ordered_again_90d"]]
        st.markdown(f'<p class="note">{len(df):,} randomly sampled customers as of 1 June 2018. '
                    "The model never saw this period. The file also records whether each customer "
                    "actually ordered again in the following 90 days, so you can check the "
                    "ranking against what really happened.</p>", unsafe_allow_html=True)
    else:
        template = pd.DataFrame(
            [["cust_001", 12, 3, 420.0, 300], ["cust_002", 45, 1, 89.9, 45],
             ["cust_003", 200, 2, 150.0, 410]], columns=required)
        st.download_button("Download a CSV template", template.to_csv(index=False).encode(),
                           "repeat_purchase_template.csv", "text/csv")
        uploaded = st.file_uploader("Customer CSV", type=["csv"])
        if uploaded is None:
            st.markdown('<p class="note">No file yet. Required columns: '
                        + ", ".join(f"<code>{c}</code>" for c in required)
                        + ". Measure every value on the scoring date. For one-order customers, "
                          "<code>tenure_days</code> equals <code>recency_days</code>.</p>",
                        unsafe_allow_html=True)
            footer()
            st.stop()
        df = pd.read_csv(uploaded)
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}. Download the template to see the format.")
            st.stop()
        df = df[required].copy()
        bad = (df["tenure_days"] < df["recency_days"]) | ((df["frequency"] == 1)
                                                         & (df["tenure_days"] != df["recency_days"]))
        if bad.any():
            st.error(f"{int(bad.sum())} rows have inconsistent dates (first order more recent than the "
                     "last, or a one-order customer with different values). Fix them and upload again.")
            st.dataframe(df[bad].head(20), use_container_width=True, hide_index=True)
            st.stop()

    with st.spinner(f"Scoring {len(df):,} customers…"):
        try:
            preds = pd.DataFrame(score_batch(to_tuples(df)))
        except requests.HTTPError as e:
            st.error(f"The API rejected the data: {e.response.text[:300]}")
            st.stop()
        except requests.RequestException:
            api_offline_message()
            st.stop()

    if actual is not None:
        preds = preds.merge(actual, on="customer_id", how="left")

    agg = {"customers": ("customer_id", "size"), "hist_rate": ("segment_precision", "first")}
    if actual is not None:
        agg["returned"] = ("ordered_again_90d", "sum")
    summary = preds.groupby("segment").agg(**agg).reindex(SEGMENT_ORDER).reset_index()
    summary["customers"] = summary["customers"].fillna(0).astype(int)
    summary["Segment"] = summary["segment"].map(SEGMENT_TEXT)

    priority = int(summary.loc[summary["segment"] != "rest", "customers"].sum())
    if actual is not None:
        summary["returned"] = summary["returned"].fillna(0).astype(int)
        prio = summary[summary["segment"] != "rest"]
        rest = summary[summary["segment"] == "rest"]
        prio_rate = prio["returned"].sum() / max(prio["customers"].sum(), 1)
        rest_rate = rest["returned"].sum() / max(rest["customers"].sum(), 1)

    k1, k2 = st.columns(2)
    with k1:
        st.markdown(kpi(f"{priority:,}", f"of {len(preds):,} customers fall in a priority group "
                                         "(top 10% or better)", accent=True), unsafe_allow_html=True)
    with k2:
        if actual is not None:
            st.markdown(kpi(f"{prio_rate / rest_rate:.1f}×" if rest_rate else "—",
                            f"how often priority customers actually ordered again vs. everyone else "
                            f"in this sample ({pct(prio_rate, 1)} vs {pct(rest_rate, 1)})"),
                        unsafe_allow_html=True)

    st.write("")
    t_col, c_col = st.columns([1.25, 1], gap="large")
    with t_col:
        table = summary[["Segment", "customers"]].rename(columns={"customers": "Customers"})
        table["Expected repeat rate"] = summary["hist_rate"].map(
            lambda v: pct(v, 1) if pd.notna(v) else "—")
        if actual is not None:
            table["Actually returned"] = [
                f"{r:,} ({pct(r / c, 1)})" if c else "—"
                for r, c in zip(summary["returned"], summary["customers"])]
        st.dataframe(table, use_container_width=True, hide_index=True)
        note = ("Expected rates come from the full June 2018 test period (each group counted "
                "separately: Top 1–5% excludes the top 1%).")
        if actual is not None:
            note += (f" This sample has only {int(summary['returned'].sum())} repeat buyers, so "
                     "its group rates are noisy.")
        st.markdown(f'<p class="note">{note}</p>', unsafe_allow_html=True)
    with c_col:
        st.altair_chart(alt.Chart(summary).mark_bar(color=ACCENT).encode(
            x=alt.X("customers:Q", title="Customers"),
            y=alt.Y("Segment:N", sort=[SEGMENT_TEXT[s] for s in SEGMENT_ORDER], title=None,
                    axis=alt.Axis(labelLimit=220)),
            tooltip=["Segment", "customers"],
        ).properties(height=alt.Step(40), title="How many customers land in each group?"),
            use_container_width=True)

    st.subheader("Priority list")
    out = preds.sort_values("repeat_probability", ascending=False).copy()
    out["segment"] = out["segment"].map(SEGMENT_TEXT)
    cols = {"customer_id": "Customer", "repeat_probability": "Chance in 90 days", "segment": "Group"}
    if actual is not None:
        cols["ordered_again_90d"] = "Actually ordered again"
        out["ordered_again_90d"] = out["ordered_again_90d"].map({1: "Yes", 0: "No"})
    st.dataframe(out[list(cols)].rename(columns=cols), use_container_width=True, hide_index=True,
                 height=360,
                 column_config={"Chance in 90 days": st.column_config.NumberColumn(format="%.4f")})
    priority_rows = out[out["segment"] != SEGMENT_TEXT["rest"]]
    st.download_button(
        f"Download the priority list ({len(priority_rows):,} customers)",
        priority_rows.to_csv(index=False).encode(), "priority_customers.csv", "text/csv",
        disabled=priority_rows.empty, type="primary")
    footer()
