# =========================================================
# DHL Strategic Intelligence Dashboard
# Run with: streamlit run dashboard.py
#
# Covers brief Sections 1-7:
#   1. Company Overview        4. Risk Monitor
#   2. Market Intelligence      5. Sentiment Analysis
#   3. Opportunity Monitor      6. Strategic Recommendations
#                                7. CEO Briefing (LLM-generated)
#
# Requires: pip install streamlit chromadb sentence-transformers ollama pandas plotly
# Requires: ollama pull qwen2.5:7b-instruct  (or whichever model you set below)
# =========================================================

import streamlit as st
import pandas as pd
import chromadb
import ollama
import datetime
import json
import re
import plotly.graph_objects as go

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
CHROMA_PATH = r"d:\NLP\dhl_chromadb"
COLLECTION_NAME = "dhl_intelligence"
OLLAMA_MODEL = "llama3.1:8b"  # change to exactly match `ollama list` output

st.set_page_config(
    page_title="DHL Strategic Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME — indigo/blue admin-dashboard style (ThemePixels-
# category inspired: sidebar nav, card-grid KPIs, soft shadows,
# tight 2-3 color accent system — not a copy of any specific
# paid template's exact assets, just the shared style language
# of that admin-dashboard genre)
# Indigo = Opportunity · Red = Risk · Teal = Trend
# =========================================================
PALETTE = {
    "Opportunity": "#306FDE",
    "Risk": "#E14B4B",
    "Trend": "#1EABA3",
    "Positive": "#1EABA3",
    "Negative": "#E14B4B",
    "Neutral": "#9AA3AF",
    "Unscored": "#D9DCE2",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #F4F6FB;
    --card: #FFFFFF;
    --border: #E6E9F2;
    --ink: #1B1F2B;
    --ink-muted: #6B7280;
    --indigo: #306FDE;
    --indigo-deep: #1F4FB0;
    --sidebar: #1A2138;
    --sidebar-active: #243154;
    --red: #E14B4B;
    --teal: #1EABA3;
}

.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }

h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; color: var(--ink) !important; }

/* ---- Sidebar: dark indigo admin-panel style ---- */
section[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #C7CDDE !important; }
section[data-testid="stSidebar"] .stRadio label { color: #C7CDDE !important; font-weight: 500; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #C7CDDE !important; }
.sidebar-brand {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: #FFFFFF !important;
    padding: 0.5rem 0 1.2rem 0;
    border-bottom: 1px solid #2C3553;
    margin-bottom: 1rem;
}
.sidebar-brand .accent { color: var(--indigo) !important; }

.dash-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: var(--ink);
    margin-bottom: 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.dash-title .chip {
    background: var(--indigo);
    color: #FFFFFF;
    font-size: 0.9rem;
    font-weight: 700;
    padding: 0.15rem 0.65rem;
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
}

.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ink);
    border-left: 5px solid var(--indigo);
    padding-left: 0.6rem;
    margin: 1.2rem 0 1rem 0;
}

p, div, span, label { font-family: 'Inter', sans-serif; }
[data-testid="stCaptionContainer"] { color: var(--ink-muted) !important; font-size: 0.82rem !important; }

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 6px rgba(31,79,176,0.06);
    border-top: 3px solid var(--indigo);
}
[data-testid="stMetricLabel"] {
    color: var(--ink-muted) !important;
    font-size: 0.74rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
}

[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

.stButton button {
    background: var(--indigo) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.3rem !important;
}
.stButton button:hover { background: var(--indigo-deep) !important; }

.stTextInput input { border: 1px solid var(--border) !important; color: var(--ink) !important; border-radius: 8px !important; }

[data-baseweb="tab-list"] { gap: 0.3rem; }
[data-baseweb="tab"] {
    background: var(--card);
    border-radius: 8px 8px 0 0;
    color: var(--ink-muted);
    font-weight: 600;
}
[aria-selected="true"][data-baseweb="tab"] { color: var(--ink) !important; border-bottom: 3px solid var(--indigo) !important; }

hr { border-color: var(--border) !important; }

.intel-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 6px rgba(31,79,176,0.05);
}
.intel-card.opportunity { border-left: 4px solid var(--indigo); }
.intel-card.risk { border-left: 4px solid var(--red); }
.intel-card.trend { border-left: 4px solid var(--teal); }

.intel-title { font-weight: 700; font-size: 1.02rem; color: var(--ink); margin-bottom: 0.5rem; }
.intel-meta { font-size: 0.78rem; color: var(--ink-muted); display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center; }

.badge {
    display: inline-block; padding: 0.18rem 0.6rem; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
}
.badge-high { background: #FCE9E9; color: #B82E2E; }
.badge-medium { background: #FFF1D6; color: #9A6B00; }
.badge-low { background: #E4F4F2; color: #117A71; }
.badge-conf-high { background: #E4F4F2; color: #117A71; }
.badge-conf-medium { background: #FFF1D6; color: #9A6B00; }
.badge-conf-low { background: #EEF0F3; color: #6B7280; }

.eyebrow {
    font-size: 0.72rem; color: var(--ink-muted); text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 700; margin-bottom: 0.4rem; display: block;
}

.kpi-card {
    background: var(--card);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 6px rgba(31,79,176,0.06);
    border-left: 4px solid var(--indigo);
}
</style>
""", unsafe_allow_html=True)


def _badge(label, kind):
    return f'<span class="badge badge-{kind}">{label}</span>'


def render_intel_card(frame, title, meta_html, extra_html=""):
    st.markdown(f"""
    <div class="intel-card {frame.lower()}">
        <div class="intel-title">{title}</div>
        <div class="intel-meta">{meta_html}</div>
        {extra_html}
    </div>
    """, unsafe_allow_html=True)


def plotly_theme(fig, height=300, showlegend=True):
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1A1D24", family="Inter"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="#EEF0F3", zerolinecolor="#EEF0F3")
    fig.update_yaxes(gridcolor="#EEF0F3", zerolinecolor="#EEF0F3")
    return fig


# -----------------------------------------------------------
# Connect to ChromaDB
# -----------------------------------------------------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)

collection = get_collection()


# =========================================================
# STAGE 1 LOGIC — category scoring, confidence, severity
# =========================================================
CATEGORY_KEYWORDS = {
    "automation": {
        "keywords": ["automation", "robotics", "robotic", "autostore", "warehouse"],
        "recommendation": "Accelerate warehouse automation and robotics deployment.",
        "impact": ["Cost Reduction", "Faster Deliveries", "Scalability"],
        "frame": ["Opportunity", "Trend"],
    },
    "energy": {
        "keywords": ["battery", "renewable", "energy", "\\bev\\b", "electric"],
        "recommendation": "Increase investment in New Energy Logistics.",
        "impact": ["Revenue Growth", "Market Differentiation",
                   "Stronger Position in Renewable Energy Supply Chains"],
        "frame": ["Opportunity"],
    },
    "sustainability": {
        "keywords": ["sustainability", "gogreen", "carbon", "emission"],
        "recommendation": "Expand sustainable logistics initiatives.",
        "impact": ["Brand Reputation", "Regulatory Readiness", "ESG Positioning"],
        "frame": ["Opportunity", "Trend"],
    },
    "ecommerce": {
        "keywords": ["fulfillment", "e commerce", "ecommerce", "online retail", "parcel"],
        "recommendation": "Expand e-commerce fulfillment capabilities.",
        "impact": ["Revenue Growth", "Customer Acquisition", "Market Share"],
        "frame": ["Opportunity"],
    },
    "partnership": {
        "keywords": ["partnership", "collaboration", "alliance", "acquisition", "agreement"],
        "recommendation": "Pursue strategic partnerships to expand market reach.",
        "impact": ["Market Expansion", "Capability Access", "Risk Sharing"],
        "frame": ["Opportunity"],
    },
    "risk_tariff": {
        "keywords": ["tariff", "customs", "trade war", "regulation", "regulatory"],
        "recommendation": "Diversify supply chains to reduce tariff and regulatory exposure.",
        "impact": ["Reduced Regulatory Risk", "Business Continuity", "Supply Chain Resilience"],
        "frame": ["Risk"],
    },
    "ai": {
        "keywords": ["artificial intelligence", "\\bai\\b", "machine learning",
                     "ai agent", "ai-powered", "ai-driven", "predictive analytics"],
        "recommendation": "Scale AI-driven operational and customer-facing capabilities.",
        "impact": ["Operational Efficiency", "Customer Experience", "Competitive Differentiation"],
        "frame": ["Opportunity", "Trend"],
    },
}

RISK_SIGNAL_KEYWORDS = {
    # Tuned against a direct corpus-wide keyword count (not semantic search)
    # run 2026-06-20: tariff=17 docs, customs=6, regulation+regulatory=3,
    # layoff=2, strike=2, disrupt/shortage/delay/rival=1 each. competitor,
    # lawsuit, fine, penalty, backlash, criticism = 0 hits — dropped, since
    # keeping zero-hit words adds noise without ever contributing signal.
    "regulatory": ["tariff", "customs", "regulation", "regulatory", "trade war"],
    "negative_sentiment": ["layoff", "job cut", "strike", "disrupt"],
    "supply_chain": ["shortage", "delay", "bottleneck", "port congestion"],
}

MIN_SCORE_THRESHOLD = 2


def _count_occurrences(keyword, text):
    if "\\b" in keyword:
        return len(re.findall(keyword, text))
    return text.count(keyword)


def _display_label(keyword):
    return keyword.replace("\\b", "").strip()


def score_categories(docs_text):
    scores = {}
    for category, info in CATEGORY_KEYWORDS.items():
        counts = {kw: _count_occurrences(kw, docs_text) for kw in info["keywords"]}
        matched = {_display_label(kw): c for kw, c in counts.items() if c > 0}
        scores[category] = {
            "score": sum(matched.values()),
            "matched_keywords": list(matched.keys()),
            "keyword_counts": matched,
        }
    return scores


def _confidence_score(best_score, runner_up_score):
    if best_score == 0:
        return "low", 0.0
    ratio = best_score / (best_score + runner_up_score) if (best_score + runner_up_score) > 0 else 1.0
    if ratio >= 0.75:
        label = "high"
    elif ratio >= 0.55:
        label = "medium"
    else:
        label = "low"
    return label, round(ratio, 2)


def _severity_level(score):
    if score >= 10:
        return "High"
    elif score >= 4:
        return "Medium"
    return "Low"


def strategic_analysis_dashboard(query, n_results=8):
    """
    NOTE: per-query risk detection was REMOVED from this function. A direct
    corpus check showed 17 documents contain "tariff", but semantic search
    for a query like "DHL tariffs trade regulation" was retrieving almost
    none of them in its top-8 — the embedding model matches generic DHL/
    logistics vocabulary more strongly than specific risk terms. Risk
    detection now runs separately via run_corpus_wide_risk_scan(), which
    checks every document directly instead of depending on retrieval luck.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    docs_text = " ".join(docs).lower()

    scores = score_categories(docs_text)
    sorted_cats = sorted(scores.items(), key=lambda x: -x[1]["score"])
    best_cat, best_info = sorted_cats[0]
    runner_up_score = sorted_cats[1][1]["score"] if len(sorted_cats) > 1 else 0

    confidence_label, confidence_ratio = _confidence_score(best_info["score"], runner_up_score)
    severity = _severity_level(best_info["score"])

    entries = []
    if best_info["score"] >= MIN_SCORE_THRESHOLD:
        cat_def = CATEGORY_KEYWORDS[best_cat]
        for frame in cat_def["frame"]:
            entries.append({
                "frame": frame, "title": cat_def["recommendation"], "category": best_cat,
                "impact_level": severity, "confidence": confidence_label,
                "confidence_score": confidence_ratio, "evidence": docs,
                "matched_keywords": best_info["keyword_counts"],
            })

    return {"query": query, "entries": entries, "raw_scores": scores}


def _corpus_wide_confidence(matched_doc_count, distinct_keyword_hits, total_keywords):
    """
    Confidence for CORPUS-WIDE risk scans, distinct from _confidence_score()
    (which compares a winning category against its runner-up within one
    query's top-k — wrong denominator here). For a corpus-wide scan,
    confidence should reflect how many of a risk category's keyword
    variants actually appear (breadth) and how many documents carry them
    (depth), not "matched docs vs. entire 302-doc corpus" — that
    denominator made a genuine 17-document tariff signal score as
    near-zero confidence, which misrepresents real evidence as noise.
    """
    keyword_breadth = distinct_keyword_hits / total_keywords if total_keywords else 0
    if matched_doc_count >= 10 and keyword_breadth >= 0.4:
        return "high", round(keyword_breadth, 2)
    elif matched_doc_count >= 4 and keyword_breadth >= 0.2:
        return "medium", round(keyword_breadth, 2)
    else:
        return "low", round(keyword_breadth, 2)


@st.cache_data(ttl=600, show_spinner=False)
def run_corpus_wide_risk_scan():
    """
    Scans EVERY document in the collection directly for risk keywords,
    instead of only checking whatever docs a semantic query happens to
    retrieve. A direct keyword count confirmed real risk content exists
    (17 docs mention "tariff", 6 mention "customs") but was being missed
    by per-query semantic retrieval. This checks the whole corpus, every
    time, so risk monitoring doesn't depend on query phrasing luck.
    """
    all_docs = collection.get()["documents"]
    full_text = " ".join(all_docs).lower()
    doc_count = max(len(all_docs), 1)

    entries = []
    for risk_type, keywords in RISK_SIGNAL_KEYWORDS.items():
        matched_docs = [d for d in all_docs if any(kw in d.lower() for kw in keywords)]
        if not matched_docs:
            continue
        occurrence_score = sum(full_text.count(kw) for kw in keywords)
        distinct_hits = sum(1 for kw in keywords if kw in full_text)
        coverage_pct = round((len(matched_docs) / doc_count) * 100, 1)
        conf_label, conf_score = _corpus_wide_confidence(len(matched_docs), distinct_hits, len(keywords))
        entries.append({
            "frame": "Risk",
            "title": f"Potential risk signal: {risk_type.replace('_', ' ')}",
            "category": risk_type,
            "impact_level": _severity_level(occurrence_score),
            "confidence": conf_label,
            "confidence_score": conf_score,
            "evidence": matched_docs[:5],
            "matched_keywords": [kw for kw in keywords if kw in full_text],
            "doc_coverage": f"{len(matched_docs)}/{doc_count} docs ({coverage_pct}%)",
            "source_query": "corpus-wide scan",
        })
    return entries


TOPIC_QUERIES = [
    "DHL e commerce fulfillment growth",
    "DHL warehouse automation opportunities",
    "DHL renewable energy logistics opportunities",
    "DHL sustainability strategy",
    "DHL strategic partnerships",
    "DHL AI analytics machine learning",
    "DHL tariffs trade regulation",
    "DHL supply chain disruption",
]

@st.cache_data(ttl=600, show_spinner=False)
def run_full_scan():
    all_entries = []
    all_scores_by_query = {}
    for q in TOPIC_QUERIES:
        result = strategic_analysis_dashboard(q)
        all_scores_by_query[q] = result["raw_scores"]
        for e in result["entries"]:
            e["source_query"] = q
            all_entries.append(e)
    return all_entries, all_scores_by_query


# =========================================================
# LLM — Sentiment scoring + CEO Briefing
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def classify_sentiment_batch(doc_tuple, source_label):
    docs = list(doc_tuple)
    numbered = "\n".join(f"{i+1}. {d[:200]}" for i, d in enumerate(docs))
    prompt = f"""Classify the sentiment of each numbered article snippet below
as exactly one of: Positive, Negative, Neutral.

Articles:
{numbered}

Respond ONLY with valid JSON, no markdown fences, no preamble, as a list:
[{{"index": 1, "sentiment": "Positive"}}, {{"index": 2, "sentiment": "Neutral"}}, ...]
One entry per article, in order."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        raw = response["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        results = []
        for i, doc in enumerate(docs):
            match = next((p for p in parsed if p.get("index") == i + 1), None)
            sentiment = match["sentiment"] if match else "Neutral"
            results.append({"text": doc, "sentiment": sentiment, "source": source_label})
        return results
    except Exception:
        return [{"text": d, "sentiment": "Unscored", "source": source_label} for d in docs]


def run_sentiment_scan():
    sample = collection.query(query_texts=["DHL news announcement"], n_results=10)
    docs = sample["documents"][0]
    return classify_sentiment_batch(tuple(docs), "news_sample")


@st.cache_data(ttl=600, show_spinner=False)
def generate_ceo_briefing(opportunity_titles, risk_titles, trend_titles):
    prompt = f"""You are briefing the CEO of DHL. Based on the following detected
signals from live monitoring, write a short executive briefing.

Opportunities detected: {opportunity_titles}
Risks detected: {risk_titles}
Trends detected: {trend_titles}

Structure your answer in exactly three short sections with these headers:
**What happened?**
**Why does it matter?**
**What should management do next?**

Keep each section to 2-3 sentences. Be direct and specific to DHL's logistics
business. Do not use markdown bullet lists, just prose paragraphs."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        return response["message"]["content"]
    except Exception as e:
        return (f"_(LLM briefing unavailable: {e})_\n\n"
                f"Fallback summary: {len(opportunity_titles)} opportunities, "
                f"{len(risk_titles)} risks, and {len(trend_titles)} trends were "
                f"detected in the latest scan.")


def _render_doc_cards(docs, empty_msg):
    if not docs:
        st.caption(empty_msg)
        return
    for i, doc in enumerate(docs, 1):
        st.markdown(f"""
        <div class="intel-card" style="border-top: 4px solid var(--border);">
            <span class="eyebrow">RESULT {i:02d}</span>
            <div style="color:var(--ink); font-size:0.92rem;">{doc[:220]}...</div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================
with st.spinner("Running strategic scan across topic areas..."):
    all_entries, scores_by_query = run_full_scan()

with st.spinner("Scanning full corpus for risk signals..."):
    risks = run_corpus_wide_risk_scan()

opportunities = [e for e in all_entries if e["frame"] == "Opportunity"]
trends = [e for e in all_entries if e["frame"] == "Trend"]

# =========================================================
# SIDEBAR NAVIGATION (admin-dashboard style: dark sidebar,
# section list instead of one long scroll)
# =========================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">📦 DHL <span class="accent">Intel</span></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        [
            "1 · Company Overview",
            "2 · Market Intelligence",
            "3 · Opportunity Monitor",
            "4 · Risk Monitor",
            "5 · Trend Monitor",
            "6 · Sentiment Analysis",
            "7 · Strategic Recommendations",
            "8 · CEO Briefing",
        ],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#2C3553;'>", unsafe_allow_html=True)
    st.caption(f"📄 {collection.count()} documents indexed")
    st.caption(f"🕐 Last scan: {datetime.datetime.now().strftime('%H:%M, %d %b')}")

# =========================================================
# HEADER (shown on every page)
# =========================================================
st.markdown("""
<div class="dash-title">DHL Strategic Intelligence <span class="chip">LIVE</span></div>
""", unsafe_allow_html=True)
st.caption("AI-powered executive monitoring across logistics, e-commerce, automation, energy, AI & sustainability")
st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# SECTION 1: Company Overview
# =========================================================
if page == "1 · Company Overview":
    st.markdown('<div class="section-title">1 &middot; Company Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Company", "DHL Group")
    col2.metric("Industry", "Logistics")
    col3.metric("Documents", collection.count())
    col4.metric("Data Sources", "3+")
    col5.metric("Last Scan", datetime.datetime.now().strftime("%H:%M, %d %b"))

    agg_scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for q_scores in scores_by_query.values():
        for cat, info in q_scores.items():
            agg_scores[cat] += info["score"]

    cov_fig = go.Figure(data=[go.Bar(
        x=list(agg_scores.values()),
        y=[c.replace("_", " ").title() for c in agg_scores.keys()],
        orientation="h",
        marker_color=PALETTE["Trend"],
    )])
    cov_fig.update_layout(title="Corpus Coverage by Topic (total signal strength)")
    st.plotly_chart(plotly_theme(cov_fig, height=280, showlegend=False), use_container_width=True)

# =========================================================
# SECTION 2: Market Intelligence
# =========================================================
if page == "2 · Market Intelligence":
    st.markdown('<div class="section-title">2 &middot; Market Intelligence</div>', unsafe_allow_html=True)
    mi_tab1, mi_tab2, mi_tab3, mi_tab4 = st.tabs(
        ["Recent News", "Competitor Activity", "Emerging Tech", "Announcements"]
    )

    with mi_tab1:
        r = collection.query(query_texts=["DHL latest news update"], n_results=5)
        _render_doc_cards(r["documents"][0], "No recent news found.")

    with mi_tab2:
        r = collection.query(query_texts=["FedEx UPS competitor logistics rival market share"], n_results=5)
        _render_doc_cards(
            r["documents"][0],
            "No competitor-specific coverage found — corpus is DHL-press-skewed. "
            "Add FedEx/UPS/Maersk queries to your collection notebooks for real coverage."
        )

    with mi_tab3:
        r = collection.query(query_texts=["DHL emerging technology AI automation innovation"], n_results=5)
        _render_doc_cards(r["documents"][0], "No emerging technology coverage found.")

    with mi_tab4:
        r = collection.query(query_texts=["DHL announces launches unveils new"], n_results=5)
        _render_doc_cards(r["documents"][0], "No announcement coverage found.")

# =========================================================
# SECTION 3: Opportunity Monitor
# =========================================================
if page == "3 · Opportunity Monitor":
    st.markdown('<div class="section-title">3 &middot; Opportunity Monitor</div>', unsafe_allow_html=True)

    if opportunities:
        opp_col1, opp_col2 = st.columns([1.3, 1])

        with opp_col1:
            for o in sorted(opportunities, key=lambda x: -x["confidence_score"]):
                meta = (
                    f"{_badge(o['impact_level'] + ' IMPACT', o['impact_level'].lower())} "
                    f"{_badge(o['confidence'].upper() + ' CONF &middot; ' + str(o['confidence_score']), 'conf-' + o['confidence'])}"
                )
                render_intel_card("Opportunity", o["title"], meta)

        with opp_col2:
            opp_fig = go.Figure(data=[go.Bar(
                x=[o["confidence_score"] for o in opportunities],
                y=[o["title"][:35] + "..." if len(o["title"]) > 35 else o["title"] for o in opportunities],
                orientation="h",
                marker_color=PALETTE["Opportunity"],
            )])
            opp_fig.update_layout(title="Opportunity Confidence", xaxis_title="Confidence ratio")
            st.plotly_chart(plotly_theme(opp_fig, height=320, showlegend=False), use_container_width=True)

        with st.expander("View supporting evidence"):
            for o in opportunities:
                st.markdown(f"**{o['title']}** — matched: `{o['matched_keywords']}`")
                for ev in o["evidence"][:3]:
                    st.caption(f"- {ev[:150]}...")
    else:
        st.info("No opportunities met the confidence threshold in this scan.")

# =========================================================
# SECTION 4: Risk Monitor
# =========================================================
if page == "4 · Risk Monitor":
    st.markdown('<div class="section-title">4 &middot; Risk Monitor</div>', unsafe_allow_html=True)

    if risks:
        risk_col1, risk_col2 = st.columns([1.3, 1])
        with risk_col1:
            for r in sorted(risks, key=lambda x: -x["confidence_score"]):
                meta = (
                    f"{_badge(r['impact_level'] + ' SEVERITY', r['impact_level'].lower())} "
                    f"{_badge(r['confidence'].upper() + ' CONF', 'conf-' + r['confidence'])} "
                    f"<span>category: {r['category']}</span> "
                    f"<span>&middot; {r.get('doc_coverage', '')}</span>"
                )
                render_intel_card("Risk", r["title"], meta)
                with st.expander(f"Evidence: {r['title'][:50]}..."):
                    for ev in r["evidence"][:3]:
                        st.caption(f"- {ev[:180]}...")
                    st.caption(f"Matched terms: {r['matched_keywords']}")

        with risk_col2:
            risk_severity_counts = pd.Series([r["impact_level"] for r in risks]).value_counts()
            rfig = go.Figure(data=[go.Pie(
                labels=risk_severity_counts.index, values=risk_severity_counts.values,
                marker=dict(colors=["#E14B4B", "#306FDE", "#1EABA3"]), hole=0.6,
            )])
            rfig.update_layout(title="Risk Severity Split")
            st.plotly_chart(plotly_theme(rfig, height=300), use_container_width=True)
    else:
        st.warning(
            "No risks detected in this scan — corpus may genuinely skew positive "
            "(DHL's own press releases dominate), or risk keyword coverage needs "
            "broadening. Flagged here honestly as a known limitation."
        )

# =========================================================
# SECTION 5: Trend Monitor
# =========================================================
if page == "5 · Trend Monitor":

    st.markdown(
        '<div class="section-title">5 · Trend Monitor</div>',
        unsafe_allow_html=True
    )

    if trends:

        trend_col1, trend_col2 = st.columns([1.3,1])

        with trend_col1:

            for t in sorted(
                trends,
                key=lambda x: -x["confidence_score"]
            ):

                meta = (
                    f"{_badge(t['impact_level'] + ' IMPACT', t['impact_level'].lower())} "
                    f"{_badge(t['confidence'].upper() + ' CONF', 'conf-' + t['confidence'])}"
                )

                render_intel_card(
                    "Trend",
                    t["title"],
                    meta
                )

        with trend_col2:

            trend_names = [
                t["category"].replace("_"," ").title()
                for t in trends
            ]

            trend_scores = [
                t["confidence_score"]
                for t in trends
            ]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=trend_names,
                        y=trend_scores,
                        marker_color="#1EABA3"
                    )
                ]
            )

            fig.update_layout(
                title="Emerging Trend Strength"
            )

            st.plotly_chart(
                plotly_theme(
                    fig,
                    height=320,
                    showlegend=False
                ),
                use_container_width=True
            )

        with st.expander("View Trend Evidence"):

            for t in trends:

                st.markdown(
                    f"**{t['title']}**"
                )

                for ev in t["evidence"][:3]:

                    st.caption(
                        f"- {ev[:150]}..."
                    )

    else:

        st.info(
            "No strong trends detected."
        )

# =========================================================
# SECTION 6: Sentiment Analysis
# =========================================================
if page == "6 · Sentiment Analysis":
    st.markdown('<div class="section-title">5 &middot; Sentiment Analysis</div>', unsafe_allow_html=True)
    st.caption(f"Sentiment scored live by {OLLAMA_MODEL} across a sample of recent articles.")

    with st.spinner(f"Scoring sentiment via {OLLAMA_MODEL}..."):
        sentiment_results = run_sentiment_scan()

    sentiment_counts = pd.Series([s["sentiment"] for s in sentiment_results]).value_counts()

    sent_col1, sent_col2, sent_col3 = st.columns([1, 1, 1])

    with sent_col1:
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index, values=sentiment_counts.values,
            marker=dict(colors=[PALETTE.get(s, "#9AA3AF") for s in sentiment_counts.index]),
            hole=0.55,
        )])
        fig.update_layout(title="News Sentiment Mix")
        st.plotly_chart(plotly_theme(fig, height=270), use_container_width=True)

    with sent_col2:
        pos = sentiment_counts.get("Positive", 0)
        neg = sentiment_counts.get("Negative", 0)
        total = sentiment_counts.sum()
        net_score = round(((pos - neg) / total) * 100, 1) if total else 0
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=net_score,
            title={"text": "Net Sentiment Index"},
            gauge={
                "axis": {"range": [-100, 100]},
                "bar": {"color": PALETTE["Trend"]},
                "steps": [
                    {"range": [-100, -20], "color": "#FCE8E8"},
                    {"range": [-20, 20], "color": "#F4F1E5"},
                    {"range": [20, 100], "color": "#E3F6EC"},
                ],
            },
        ))
        st.plotly_chart(plotly_theme(gauge, height=270, showlegend=False), use_container_width=True)

    with sent_col3:
        st.markdown('<span class="eyebrow">SAMPLE CLASSIFICATIONS</span>', unsafe_allow_html=True)
        for s in sentiment_results[:4]:
            kind = {"Positive": "low", "Negative": "high", "Neutral": "medium"}.get(s["sentiment"], "medium")
            st.markdown(f"""
            <div class="intel-card" style="padding:0.5rem 0.8rem; margin-bottom:0.4rem;">
                <span class="badge badge-{kind}">{s['sentiment'].upper()}</span>
                <div style="font-size:0.78rem; color:var(--ink-muted); margin-top:0.3rem;">{s['text'][:90]}...</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<span class="eyebrow">PUBLIC SENTIMENT (PROXY)</span>', unsafe_allow_html=True)
    st.caption(
        "No direct public/social source (Reddit, reviews, forums) is in the collection "
        "pipeline yet — only press/news sources. The chart above reflects news sentiment "
        "only. Add a community source to populate true public sentiment separately."
    )
    st.markdown('<span class="eyebrow">SENTIMENT TREND</span>', unsafe_allow_html=True)
    st.caption(
        "Trend-over-time needs normalized published_at timestamps across all three "
        "sources — currently only NewsAPI provides them consistently."
    )

# =========================================================
# SECTION 7: Strategic Recommendations
# =========================================================
if page == "7 · Strategic Recommendations":
    st.markdown('<div class="section-title">6 &middot; Strategic Recommendations</div>', unsafe_allow_html=True)

    if opportunities:
        overall_risk_note = (
            f"{len(risks)} active risk signal(s) — see Risk Monitor" if risks
            else "No active risk signals this scan"
        )
        rec_col1, rec_col2 = st.columns([1.3, 1])

        with rec_col1:
            for o in sorted(opportunities, key=lambda x: -x["confidence_score"]):
                meta = (
                    f"{_badge('PRIORITY: ' + o['impact_level'].upper(), o['impact_level'].lower())} "
                    f"{_badge(o['confidence'].upper() + ' CONFIDENCE', 'conf-' + o['confidence'])}"
                )
                expected_impact = CATEGORY_KEYWORDS.get(o["category"], {}).get("impact", [])
                chips = " ".join(f"<span class='badge badge-low'>{i}</span>" for i in expected_impact)
                extra = f"""
                <div style="margin-top:0.6rem;">
                    <span class="eyebrow" style="margin-bottom:0.2rem;">EXPECTED IMPACT</span>
                    {chips}
                </div>
                <div style="font-size:0.78rem; color:var(--ink-muted); margin-top:0.5rem;">
                    <strong>Risk level:</strong> {overall_risk_note}
                </div>
                """
                render_intel_card("Opportunity", o["title"], meta, extra)

        with rec_col2:
            priority_counts = pd.Series([o["impact_level"] for o in opportunities]).value_counts()
            pfig = go.Figure(data=[go.Bar(
                x=priority_counts.index, y=priority_counts.values,
                marker_color=["#E14B4B" if p == "High" else "#306FDE" if p == "Medium" else "#1EABA3"
                              for p in priority_counts.index],
            )])
            pfig.update_layout(title="Recommendation Priority Mix")
            st.plotly_chart(plotly_theme(pfig, height=300, showlegend=False), use_container_width=True)
    else:
        st.write("No recommendations generated yet.")

# =========================================================
# SECTION 8: CEO Briefing
# =========================================================
if page == "8 · CEO Briefing":
    st.markdown('<div class="section-title">7 &middot; CEO Briefing</div>', unsafe_allow_html=True)

    if st.button("Generate Executive Briefing"):
        with st.spinner(f"Asking {OLLAMA_MODEL} to draft the briefing..."):
            briefing = generate_ceo_briefing(
                opportunity_titles=[o["title"] for o in opportunities],
                risk_titles=[r["title"] for r in risks],
                trend_titles=[t["title"] for t in trends],
            )
        st.markdown(f"""
        <div style="background:var(--card); border:1px solid var(--border); border-top:4px solid var(--indigo);
                    border-radius:10px; padding:1.5rem 1.8rem; box-shadow:0 1px 3px rgba(16,24,40,0.05);">
        """, unsafe_allow_html=True)
        st.markdown(briefing)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("Click to generate a fresh LLM-written briefing based on the current scan.")