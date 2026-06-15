import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import random
from sklearn.impute import SimpleImputer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 60%, #0f172a 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Main background */
    .stApp { background: #0f172a; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-value  { font-size: 2rem; font-weight: 800; margin: 0; }
    .kpi-label  { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; letter-spacing:.05em; text-transform:uppercase; }

    /* Intervention boxes */
    .iv-box {
        border-radius: 10px;
        padding: 16px 18px;
        margin: 8px 0;
        color: #f1f5f9 !important;
    }
    .iv-box strong { font-size: 0.72rem; letter-spacing: .1em; text-transform: uppercase; }

    /* Member badge */
    .member-badge {
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 18px;
    }
    .member-badge h2 { margin: 0; font-size: 1.8rem; }
    .member-badge p  { margin: 6px 0 0; opacity: .75; font-size: .9rem; }

    /* Sample member card */
    .sample-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        cursor: pointer;
        color: #e2e8f0 !important;
    }
    .sample-card:hover { border-color: #60a5fa; }

    /* Profile row */
    .profile-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #1e293b;
        color: #e2e8f0;
    }
    .profile-key   { color: #94a3b8; font-size:.85rem; }
    .profile-value { font-weight: 600; font-size:.85rem; }

    /* Section header */
    .section-hdr {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 18px 0 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEG_COLORS = {
    "Champions":      {"bg":"#14532d", "border":"#22c55e", "text":"#bbf7d0", "pill":"#22c55e"},
    "At-Risk VIPs":   {"bg":"#450a0a", "border":"#ef4444", "text":"#fecaca", "pill":"#ef4444"},
    "Promising":      {"bg":"#1e3a5f", "border":"#3b82f6", "text":"#bfdbfe", "pill":"#3b82f6"},
    "About to Lapse": {"bg":"#431407", "border":"#f97316", "text":"#fed7aa", "pill":"#f97316"},
    "Hibernating":    {"bg":"#3b0764", "border":"#a855f7", "text":"#e9d5ff", "pill":"#a855f7"},
}

IV_BOX_COLORS = {
    "who":    ("#1e3a5f", "#60a5fa"),
    "when":   ("#14532d", "#4ade80"),
    "what":   ("#431407", "#fb923c"),
    "why":    ("#3b0764", "#c084fc"),
    "metric": ("#1e293b", "#94a3b8"),
}

INTERVENTIONS = {
    "At-Risk VIPs": {
        "who":    "High CLV members with churn probability > 60%",
        "when":   "Triggered when Q4 flights drop > 50% vs Q3",
        "what":   "Personal outreach from loyalty manager + upgrade offer to Aurora if on Nova",
        "why":    "Losing one VIP costs more revenue than retaining 10 average members",
        "metric": "Reactivation rate within 90 days; revenue recovered per campaign",
    },
    "About to Lapse": {
        "who":    "Low CLV members with churn probability > 60% and 0 flights in last 3 months",
        "when":   "Monthly batch trigger",
        "what":   "Low-cost email: double points on next booking (no free flight needed)",
        "why":    "Cost-efficient: email only. Even 10% reactivation is positive ROI",
        "metric": "Email open rate; flights booked within 30 days of campaign",
    },
    "Promising": {
        "who":    "Mid-low CLV, churn probability < 30%, flight consistency > 25%",
        "when":   "After 3rd flight in a calendar year",
        "what":   "Card upgrade nudge: show exact $ gap to next tier",
        "why":    "Engaged but undervalued — tier aspiration drives spend increase",
        "metric": "Card upgrade conversion rate; CLV growth 12 months post-campaign",
    },
    "Champions": {
        "who":    "Top quartile CLV, churn probability < 30%, redemption rate > 10%",
        "when":   "Annual loyalty anniversary",
        "what":   "Exclusive lounge access trial + referral bonus",
        "why":    "Champions are already loyal — goal is advocacy, not retention",
        "metric": "Referral conversion rate; NPS score for this segment",
    },
    "Hibernating": {
        "who":    "Members with 0 flights across 2017 and 2018 H1",
        "when":   "Bi-annual win-back campaign",
        "what":   "Points expiry extension + 20% off next booking",
        "why":    "Points expiry is a known reactivation trigger; low cost to attempt",
        "metric": "Win-back rate; suppress after 2 failed campaigns",
    },
}

SAMPLE_MEMBERS = {
    "Champions":      {"desc": "High CLV · Active flyer · Low risk"},
    "At-Risk VIPs":   {"desc": "High CLV · Declining activity · Urgent"},
    "Promising":      {"desc": "Mid CLV · Consistent · Growth potential"},
    "About to Lapse": {"desc": "Low CLV · Inactive · Needs nudge"},
    "Hibernating":    {"desc": "Any CLV · No recent flights · Win-back"},
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists("customer_segments.csv"):
        df = pd.read_csv("customer_segments.csv")
        return df

    loyalty = pd.read_csv("Customer Loyalty History.csv")
    flight  = pd.read_csv("Customer Flight Activity.csv")

    loyalty.loc[loyalty["Salary"] < 0, "Salary"] = np.nan
    loyalty["Salary"] = loyalty.groupby("Province")["Salary"] \
        .transform(lambda x: x.fillna(x.median()))
    loyalty["Salary"].fillna(loyalty["Salary"].median(), inplace=True)

    tier_map = {"Star": 1, "Nova": 2, "Aurora": 3}
    loyalty["card_tier"] = loyalty["Loyalty Card"].map(tier_map)
    loyalty["is_promo"]  = (loyalty["Enrollment Type"] == "2018 Promotion").astype(int)

    loyalty["enroll_date"] = pd.to_datetime(
        loyalty["Enrollment Year"].astype(str) + "-" +
        loyalty["Enrollment Month"].astype(str).str.zfill(2) + "-01")
    dataset_end = pd.Timestamp("2018-12-01")
    loyalty["tenure_months"] = ((dataset_end - loyalty["enroll_date"])
                                 .dt.days / 30.44).astype(int)

    flight["Date"] = pd.to_datetime(flight[["Year","Month"]].assign(DAY=1))
    data = flight.merge(loyalty, on="Loyalty Number", how="left")
    data = data.sort_values(["Loyalty Number","Date"]).reset_index(drop=True)

    flight_2018_h1 = data[(data["Year"]==2018) & (data["Month"]<=6)]
    active_h1 = flight_2018_h1.groupby("Loyalty Number")["Total Flights"].sum().reset_index()
    active_h1.columns = ["Loyalty Number","flights_h1_2018"]
    loyalty = loyalty.merge(active_h1, on="Loyalty Number", how="left")
    loyalty["flights_h1_2018"].fillna(0, inplace=True)
    loyalty["churn_behavioral"] = (loyalty["flights_h1_2018"] == 0).astype(int)

    d17 = data[data["Year"]==2017]
    feat = d17.groupby("Loyalty Number").agg(
        total_flights       = ("Total Flights",  "sum"),
        avg_monthly_flights = ("Total Flights",  "mean"),
        max_monthly_flights = ("Total Flights",  "max"),
        active_months       = ("Total Flights",  lambda x: (x > 0).sum()),
        total_distance      = ("Distance",       "sum"),
        avg_distance        = ("Distance",       "mean"),
        points_accumulated  = ("Points Accumulated", "sum"),
        points_redeemed     = ("Points Redeemed",    "sum"),
        dollar_redeemed     = ("Dollar Cost Points Redeemed", "sum"),
    ).reset_index()

    feat["redemption_rate"]    = np.where(feat["points_accumulated"]>0,
        feat["points_redeemed"]/feat["points_accumulated"], 0)
    feat["flight_consistency"] = feat["active_months"] / 12

    summer = d17[d17["Month"].isin([6,7,8])].groupby("Loyalty Number")["Total Flights"].sum().rename("summer_flights")
    winter = d17[d17["Month"].isin([12,1,2])].groupby("Loyalty Number")["Total Flights"].sum().rename("winter_flights")
    q4     = d17[d17["Month"].isin([10,11,12])].groupby("Loyalty Number")["Total Flights"].sum().rename("q4_flights")
    feat = feat.merge(summer, on="Loyalty Number", how="left") \
               .merge(winter, on="Loyalty Number", how="left") \
               .merge(q4,     on="Loyalty Number", how="left")
    feat[["summer_flights","winter_flights","q4_flights"]] = \
        feat[["summer_flights","winter_flights","q4_flights"]].fillna(0)

    edu_map = {"High School or Below":1,"College":2,"Bachelor":3,"Master":4,"Doctor":5}
    loyalty["edu_level"] = loyalty["Education"].map(edu_map)

    demo_cols = ["Loyalty Number","Gender","Education","Salary","Marital Status",
                 "card_tier","CLV","is_promo","tenure_months","churn_behavioral","edu_level"]
    feat = feat.merge(loyalty[demo_cols], on="Loyalty Number", how="left")
    feat = pd.get_dummies(feat, columns=["Gender","Marital Status"], drop_first=True)

    feature_cols = ["total_flights","avg_monthly_flights","max_monthly_flights",
                    "active_months","total_distance","avg_distance",
                    "points_accumulated","points_redeemed","redemption_rate",
                    "flight_consistency","summer_flights","winter_flights","q4_flights",
                    "Salary","card_tier","CLV","is_promo","tenure_months","edu_level"]
    feature_cols += [c for c in feat.columns if c.startswith("Gender_") or c.startswith("Marital")]
    feature_cols = [c for c in feature_cols if c in feat.columns]

    X = feat[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    if os.path.exists("churn_model.pkl"):
        with open("churn_model.pkl","rb") as f:
            model = pickle.load(f)
        feat["churn_prob"] = model.predict_proba(X_imp)[:,1]
    else:
        feat["churn_prob"] = 1 - feat["flight_consistency"].clip(0,1)

    feat["value_tier"] = pd.qcut(feat["CLV"], q=4,
        labels=["Low Value","Mid-Low Value","Mid-High Value","High Value"])
    feat["engaged"] = (
        (feat["flight_consistency"] > 0.5) | (feat["redemption_rate"] > 0.1)
    ).astype(int)
    feat["risk_bucket"] = pd.cut(feat["churn_prob"],
        bins=[0,0.3,0.6,1.0], labels=["Low Risk","Medium Risk","High Risk"])

    def assign_segment(row):
        high_clv  = row["value_tier"] in ["High Value","Mid-High Value"]
        high_risk = row["risk_bucket"] == "High Risk"
        engaged   = row["engaged"] == 1
        if high_clv and not high_risk and engaged:       return "Champions"
        elif high_clv and high_risk:                     return "At-Risk VIPs"
        elif not high_clv and not high_risk and engaged: return "Promising"
        elif high_risk and not high_clv:                 return "About to Lapse"
        else:                                            return "Hibernating"

    feat["segment"] = feat.apply(assign_segment, axis=1)
    out = feat.merge(
        loyalty[["Loyalty Number","Province","City","Gender","Education",
                 "Marital Status","Loyalty Card","Enrollment Type"]],
        on="Loyalty Number", how="left")
    return out


def safe_get(row, col, fmt="int"):
    val = row.get(col, None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if fmt == "pct":  return f"{float(val):.1%}"
    if fmt == "int":  return f"{int(val):,}"
    if fmt == "usd":  return f"${float(val):,.0f}"
    return str(val)


def iv_block(label, text, bg, border):
    return f"""
    <div class="iv-box" style="background:{bg};border-left:4px solid {border};">
        <strong style="color:{border}">{label}</strong><br>
        <span style="color:#f1f5f9">{text}</span>
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Loyalty Intelligence")
    st.markdown("*CA Club · IIT Guwahati · SP'26*")
    st.divider()
    page = st.radio("Navigate", [
        "📊  Overview",
        "🚨  At-Risk Members",
        "🔍  Segment Deep Dive",
        "👤  Member Lookup",
    ])
    st.divider()
    st.caption("16,737 Canadian members · 2017–2018")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading..."):
    df = load_data()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊  Overview":
    st.markdown("# ✈️ Airline Loyalty — Executive Dashboard")
    st.markdown("Real-time intelligence for marketing & retention teams.")
    st.divider()

    total   = len(df)
    churned = int(df["churn_behavioral"].sum())
    at_risk = int((df["churn_prob"] > 0.6).sum())
    avg_clv = df["CLV"].mean()
    vip_risk= int((df["segment"]=="At-Risk VIPs").sum())

    kpis = [
        (f"{total:,}",       "Total Members",        "#60a5fa"),
        (f"{churned:,}",     "Behaviorally Churned", "#f87171"),
        (f"{at_risk:,}",     "High Churn Risk",      "#fb923c"),
        (f"${avg_clv:,.0f}", "Avg Customer CLV",     "#4ade80"),
        (f"{vip_risk:,}",    "At-Risk VIPs",         "#c084fc"),
    ]
    cols = st.columns(5)
    for col, (val, label, color) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color}">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-hdr">Customer Segments</div>', unsafe_allow_html=True)
        seg_counts = df["segment"].value_counts()
        colors = [SEG_COLORS[s]["border"] for s in seg_counts.index]
        fig, ax = plt.subplots(figsize=(6,3.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        bars = ax.barh(seg_counts.index, seg_counts.values,
                       color=colors, edgecolor="none", height=0.55)
        for bar, val in zip(bars, seg_counts.values):
            ax.text(bar.get_width()+40, bar.get_y()+bar.get_height()/2,
                    f"{val:,}", va="center", color="#e2e8f0", fontsize=9)
        ax.tick_params(colors="#94a3b8")
        ax.set_xlabel("Members", color="#94a3b8")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(axis="both", colors="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown('<div class="section-hdr">Average CLV by Segment ($)</div>', unsafe_allow_html=True)
        seg_clv = df.groupby("segment")["CLV"].mean().sort_values()
        colors2 = [SEG_COLORS[s]["border"] for s in seg_clv.index]
        fig2, ax2 = plt.subplots(figsize=(6,3.5), facecolor="#0f172a")
        ax2.set_facecolor("#0f172a")
        bars2 = ax2.barh(seg_clv.index, seg_clv.values,
                         color=colors2, edgecolor="none", height=0.55)
        for bar, val in zip(bars2, seg_clv.values):
            ax2.text(bar.get_width()+40, bar.get_y()+bar.get_height()/2,
                     f"${val:,.0f}", va="center", color="#e2e8f0", fontsize=9)
        for spine in ax2.spines.values(): spine.set_visible(False)
        ax2.tick_params(axis="both", colors="#94a3b8")
        ax2.set_xlabel("Avg CLV ($)", color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown('<div class="section-hdr">Churn Probability Distribution by Segment</div>',
                unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(12,3.5), facecolor="#0f172a")
    ax3.set_facecolor("#0f172a")
    for seg, c in SEG_COLORS.items():
        subset = df[df["segment"]==seg]["churn_prob"]
        if len(subset) > 5:
            subset.plot.kde(ax=ax3, label=seg, color=c["border"], linewidth=2.5)
    ax3.axvline(0.6, color="#f87171", linestyle="--", linewidth=1.5,
                label="High Risk threshold")
    ax3.set_xlabel("Churn Probability", color="#94a3b8")
    ax3.set_ylabel("Density", color="#94a3b8")
    ax3.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax3.spines.values(): spine.set_visible(False)
    ax3.tick_params(colors="#94a3b8")
    plt.tight_layout()
    st.pyplot(fig3); plt.close()

    if "Province" in df.columns:
        st.markdown('<div class="section-hdr">Top Provinces — Churn Risk</div>',
                    unsafe_allow_html=True)
        prov = df.groupby("Province").agg(
            Members        = ("Loyalty Number","count"),
            Avg_CLV        = ("CLV","mean"),
            Avg_Churn_Prob = ("churn_prob","mean"),
            Pct_High_Risk  = ("churn_prob", lambda x: (x>0.6).mean())
        ).sort_values("Pct_High_Risk", ascending=False).head(10).round(2)
        prov["Avg_CLV"]       = prov["Avg_CLV"].map("${:,.0f}".format)
        prov["Pct_High_Risk"] = prov["Pct_High_Risk"].map("{:.1%}".format)
        st.dataframe(prov, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – AT-RISK MEMBERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨  At-Risk Members":
    st.markdown("# 🚨 At-Risk Member List")
    st.markdown("Filterable list ready for CRM export.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        seg_filter = st.multiselect("Segment", list(SEG_COLORS.keys()),
                                    default=["At-Risk VIPs","About to Lapse"])
    with c2:
        risk_thresh = st.slider("Min. Churn Probability", 0.0, 1.0, 0.6, 0.05)
    with c3:
        card_opts = df["Loyalty Card"].dropna().unique().tolist() if "Loyalty Card" in df.columns else []
        card_filter = st.multiselect("Loyalty Card", card_opts, default=card_opts)

    filtered = df[
        (df["segment"].isin(seg_filter)) &
        (df["churn_prob"] >= risk_thresh) &
        (df["Loyalty Card"].isin(card_filter) if "Loyalty Card" in df.columns else True)
    ].copy()

    rev_at_risk = filtered["CLV"].sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Members Matched", f"{len(filtered):,}")
    m2.metric("CLV at Risk",     f"${rev_at_risk:,.0f}")
    m3.metric("Avg Churn Prob",  f"{filtered['churn_prob'].mean():.1%}" if len(filtered) else "—")

    display_cols = ["Loyalty Number","segment","churn_prob","CLV",
                    "total_flights","active_months","redemption_rate",
                    "Loyalty Card","Province"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    styled = filtered[display_cols].sort_values("churn_prob", ascending=False).reset_index(drop=True)
    styled["churn_prob"]      = styled["churn_prob"].round(3)
    styled["CLV"]             = styled["CLV"].map("${:,.0f}".format)
    styled["redemption_rate"] = styled["redemption_rate"].map("{:.1%}".format)
    st.dataframe(styled, use_container_width=True, height=420)

    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Export to CSV", csv, "at_risk_members.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – SEGMENT DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Segment Deep Dive":
    st.markdown("# 🔍 Segment Deep Dive")
    st.divider()

    seg_choice = st.selectbox("Select a segment", list(SEG_COLORS.keys()))
    c = SEG_COLORS[seg_choice]
    seg_df = df[df["segment"] == seg_choice]

    st.markdown(f"""
    <div style="background:{c['bg']};border:2px solid {c['border']};
         border-radius:12px;padding:16px 22px;margin-bottom:16px;">
        <span style="color:{c['border']};font-size:1.4rem;font-weight:800">{seg_choice}</span>
        <span style="color:{c['text']};font-size:.9rem;margin-left:12px">
            {len(seg_df):,} members · Avg CLV ${seg_df['CLV'].mean():,.0f}
        </span>
    </div>""", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Members",        f"{len(seg_df):,}")
    k2.metric("Avg CLV",        f"${seg_df['CLV'].mean():,.0f}")
    k3.metric("Avg Churn Prob", f"{seg_df['churn_prob'].mean():.1%}")
    k4.metric("Avg Flights/Yr", f"{seg_df['total_flights'].mean():.1f}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-hdr">Flight Consistency</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.hist(seg_df["flight_consistency"], bins=13,
                color=c["border"], edgecolor="none", alpha=0.85)
        ax.set_xlabel("Consistency (0=never, 1=every month)", color="#94a3b8")
        ax.set_ylabel("Members", color="#94a3b8")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(colors="#94a3b8")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.markdown('<div class="section-hdr">Loyalty Card Breakdown</div>', unsafe_allow_html=True)
        card_col = "Loyalty Card" if "Loyalty Card" in seg_df.columns else "card_tier"
        card_counts = seg_df[card_col].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5,3), facecolor="#0f172a")
        ax2.set_facecolor("#0f172a")
        bar_colors = ["#60a5fa","#a78bfa","#34d399"][:len(card_counts)]
        card_counts.plot(kind="bar", ax=ax2, color=bar_colors, edgecolor="none")
        ax2.set_xlabel("", color="#94a3b8")
        ax2.set_ylabel("Members", color="#94a3b8")
        ax2.tick_params(axis="x", rotation=0, colors="#94a3b8")
        ax2.tick_params(axis="y", colors="#94a3b8")
        for spine in ax2.spines.values(): spine.set_visible(False)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.divider()
    st.markdown(f'<div class="section-hdr">📋 Retention Playbook — {seg_choice}</div>',
                unsafe_allow_html=True)
    iv = INTERVENTIONS.get(seg_choice, {})
    if iv:
        left, right = st.columns(2)
        items = list(iv.items())
        for i, (k, v) in enumerate(items):
            bg, border = IV_BOX_COLORS.get(k, ("#1e293b","#94a3b8"))
            block = iv_block(k.upper(), v, bg, border)
            (left if i % 2 == 0 else right).markdown(block, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – MEMBER LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤  Member Lookup":
    st.markdown("# 👤 Member Lookup")
    st.markdown("Search any member — or pick a sample from each segment below.")
    st.divider()

    # ── Sample member picker ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🎲 Pick a Sample Member by Segment</div>',
                unsafe_allow_html=True)
    st.markdown("Click a segment to load a random real member from that group.")

    btn_cols = st.columns(5)
    picked_id = None

    for i, (seg, info) in enumerate(SAMPLE_MEMBERS.items()):
        c = SEG_COLORS[seg]
        if btn_cols[i].button(
            f"{seg}\n{info['desc']}",
            key=f"btn_{seg}",
            use_container_width=True,
            help=f"Load a random {seg} member"
        ):
            sample_pool = df[df["segment"] == seg]["Loyalty Number"].values
            if len(sample_pool) > 0:
                picked_id = str(random.choice(sample_pool))
                st.session_state["lookup_id"] = picked_id

    st.divider()

    # ── Manual search ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🔎 Or Enter a Loyalty Number Manually</div>',
                unsafe_allow_html=True)

    default_val = st.session_state.get("lookup_id", "")
    loyalty_input = st.text_input(
        "Loyalty Number",
        value=default_val,
        placeholder="e.g. 480934  — or click a segment button above"
    )

    if loyalty_input:
        try:
            lid = int(loyalty_input)
            member = df[df["Loyalty Number"] == lid]

            if member.empty:
                st.error(f"No member found with Loyalty Number {lid}.")
            else:
                m   = member.iloc[0]
                seg = m.get("segment", "Unknown")
                c   = SEG_COLORS.get(seg, {"bg":"#1e293b","border":"#94a3b8",
                                            "text":"#e2e8f0","pill":"#94a3b8"})

                # ── Member badge ──────────────────────────────────────────────
                st.markdown(f"""
                <div class="member-badge"
                     style="background:{c['bg']};border:2px solid {c['border']};">
                    <h2 style="color:{c['border']}">{seg}</h2>
                    <p style="color:{c['text']}">
                        Loyalty #{lid} &nbsp;·&nbsp;
                        Churn Risk:
                        <strong style="color:{c['border']}">
                            {m['churn_prob']:.1%}
                        </strong>
                    </p>
                </div>""", unsafe_allow_html=True)

                # ── KPIs ──────────────────────────────────────────────────────
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Churn Probability", f"{m['churn_prob']:.1%}")
                k2.metric("CLV",               f"${m['CLV']:,.0f}")
                k3.metric("Flights (2017)",    safe_get(m,"total_flights"))
                k4.metric("Active Months",     f"{safe_get(m,'active_months')}/12")

                st.divider()
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown('<div class="section-hdr">📈 Behavioral Profile</div>',
                                unsafe_allow_html=True)
                    profile = {
                        "Flight Consistency": safe_get(m, "flight_consistency", "pct"),
                        "Redemption Rate":    safe_get(m, "redemption_rate",    "pct"),
                        "Points Accumulated": safe_get(m, "points_accumulated", "int"),
                        "Points Redeemed":    safe_get(m, "points_redeemed",    "int"),
                        "Q4 Flights":         safe_get(m, "q4_flights",         "int"),
                        "Summer Flights":     safe_get(m, "summer_flights",     "int"),
                        "Winter Flights":     safe_get(m, "winter_flights",     "int"),
                    }
                    for k, v in profile.items():
                        st.markdown(f"""
                        <div class="profile-row">
                            <span class="profile-key">{k}</span>
                            <span class="profile-value" style="color:{c['border']}">{v}</span>
                        </div>""", unsafe_allow_html=True)

                with col_r:
                    st.markdown('<div class="section-hdr">👤 Demographics</div>',
                                unsafe_allow_html=True)
                    demo = {
                        "Loyalty Card":    m.get("Loyalty Card","N/A"),
                        "Province":        m.get("Province","N/A"),
                        "City":            m.get("City","N/A"),
                        "Gender":          m.get("Gender","N/A"),
                        "Education":       m.get("Education","N/A"),
                        "Marital Status":  m.get("Marital Status","N/A"),
                        "Enrollment Type": m.get("Enrollment Type","N/A"),
                        "Tenure":          f"{int(m['tenure_months'])} months",
                        "Salary":          safe_get(m,"Salary","usd"),
                    }
                    for k, v in demo.items():
                        st.markdown(f"""
                        <div class="profile-row">
                            <span class="profile-key">{k}</span>
                            <span class="profile-value">{v}</span>
                        </div>""", unsafe_allow_html=True)

                # ── Recommended action ────────────────────────────────────────
                st.divider()
                st.markdown('<div class="section-hdr">📋 Recommended Action</div>',
                            unsafe_allow_html=True)
                iv = INTERVENTIONS.get(seg, {})
                if iv:
                    a1, a2 = st.columns(2)
                    items = list(iv.items())
                    for i, (k, v) in enumerate(items):
                        bg, border = IV_BOX_COLORS.get(k, ("#1e293b","#94a3b8"))
                        block = iv_block(k.upper(), v, bg, border)
                        (a1 if i % 2 == 0 else a2).markdown(block, unsafe_allow_html=True)

        except ValueError:
            st.error("Please enter a valid numeric loyalty number.")