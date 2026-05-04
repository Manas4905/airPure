import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import io
import json
import time
import requests
import os
import hashlib

genai = None
GENAI_IMPORT_ERROR = None
try:
    from google import genai
except Exception as e:
    try:
        import google.genai as genai
    except Exception as e2:
        genai = None
        GENAI_IMPORT_ERROR = (
            f"primary import failed: {e}; fallback import failed: {e2}"
        )

# Read Gemini API key from Streamlit secrets (with common fallback keys)
KEY_SOURCE = "none"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    KEY_SOURCE = "st.secrets.GEMINI_API_KEY"
else:
    GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    if GEMINI_API_KEY:
        KEY_SOURCE = "st.secrets.GOOGLE_API_KEY"
    else:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY:
            KEY_SOURCE = "env.GEMINI_API_KEY"
        else:
            GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
            if GEMINI_API_KEY:
                KEY_SOURCE = "env.GOOGLE_API_KEY"
if isinstance(GEMINI_API_KEY, str):
    GEMINI_API_KEY = GEMINI_API_KEY.strip()

def _key_fingerprint(key: str) -> str:
    if not key:
        return "none"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"sha256:{digest} (len={len(key)})"

client = None
if GEMINI_API_KEY and genai:
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})

MODEL_CANDIDATES = [
    st.secrets.get("GEMINI_MODEL"),
    os.getenv("GEMINI_MODEL"),
    "gemini-2.5-flash",
    "gemini-3.1-flash",
]
MODEL_CANDIDATES = [m.strip() for m in MODEL_CANDIDATES if isinstance(m, str) and m.strip()]

def _quota_fallback_recommendation(prompt: str) -> str:
    return (
        "### Recommendation\n"
        "- Prefer a purifier with a **True HEPA H13 filter** and an **activated carbon filter**.\n"
        "- Choose CADR based on room size: small room 100-200 m3/h, medium room 200-350 m3/h, large room 350+ m3/h.\n"
        "- If outdoor pollution is high, prioritize higher CADR and a sealed body design.\n"
        "- For bedrooms, look for **<30 dB sleep mode** and low night-light noise.\n"
        "- Check yearly filter replacement cost before buying.\n"
        "- Place purifier away from walls, run on medium/high during peak pollution hours, and keep windows closed then.\n\n"
       
    )

@st.cache_data(ttl=86400, show_spinner=False) # Cache successful responses for 24 hours
def _get_gemini_response_cached(prompt: str, _api_key: str):
    if client is None:
        raise ValueError("Client not initialized")

    concise_prompt = prompt + " Respond in 5-10 short sentences only, and last line should be a summary highlighting the key points. Summary should come first, followed by a detailed explanation. Use bullet points for the detailed explanation, and format both summary and details in Markdown."
    
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=concise_prompt,
            )
            
            if hasattr(response, "text") and response.text:
                return response.text, model_name
            if getattr(response, "candidates", None) and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text, model_name
                
            raise ValueError(f"No valid text from {model_name}")
        except Exception as model_error:
            last_error = model_error
            msg = str(model_error)
            # If rate limited (429/quota)...
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                # If Google disabled free tier specifically for this model (limit 0), skip it and try next!
                if "limit: 0" in msg:
                    continue
                # Otherwise, it's a real limit exceeded, stop making redundant requests.
                raise model_error
            # If the API key is unauthorized or Generative Language API is not enabled...
            if "403" in msg or "PERMISSION_DENIED" in msg:
                raise model_error
            # Otherwise (like 404 NOT FOUND), continue to next model
            continue
            
    # If all models failed, raise the last error so Streamlit DOES NOT cache it
    if last_error:
        raise last_error
    raise ValueError("No models available.")

def get_ai_recommendation(prompt: str) -> str:
    st.session_state["ai_debug"] = {
        "key_source": KEY_SOURCE,
        "key_fingerprint": _key_fingerprint(GEMINI_API_KEY),
        "model_used": None,
        "last_error": None,
    }
    if not GEMINI_API_KEY:
        return "API key not set. Please configure your Gemini API key."
    if genai is None:
        if GENAI_IMPORT_ERROR:
            return (
                "The google-generativeai package could not be imported. "
                f"Import error: {GENAI_IMPORT_ERROR}. "
                "Please verify requirements and rebuild the app."
            )
        return (
            "The google-generativeai package is not installed. "
            "Please add it to requirements.txt and restart the app."
        )
    if client is None:
        return "Gemini client could not be initialized. Please verify the API key and restart the app."
    
    try:
        # Call the new cached function (renamed to clear previous bad cache). 
        # By raising exceptions on error, Streamlit won't cache empty or failed responses!
        text, model_used = _get_gemini_response_cached(prompt, GEMINI_API_KEY)
        
        st.session_state["ai_debug"]["model_used"] = model_used
        st.session_state["ai_debug"]["last_error"] = None
        return text
    except Exception as e:
        msg = str(e)
        st.session_state["ai_debug"]["last_error"] = msg
        # Serve fallback dynamically for ANY API failure (quota, 404, etc.) to ensure the UI doesn't visually break.
        return _quota_fallback_recommendation(prompt)

st.set_page_config(layout="wide", page_title="AirPure AI", page_icon="🌿")

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;1,9..144,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f5f0ff !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1a1025 !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f0ff 0%, #fce4ff 30%, #e8f4ff 60%, #f0fff8 100%) !important;
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Main container ── */
.main .block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Hero title ── */
h1 {
    font-family: 'Fraunces', serif !important;
    font-weight: 500 !important;
    font-size: 3rem !important;
    background: linear-gradient(135deg, #7c3aed, #db2777, #0891b2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    margin-bottom: 0.2rem !important;
}

h2 {
    font-family: 'Fraunces', serif !important;
    font-weight: 300 !important;
    font-size: 1.8rem !important;
    color: #4c1d95 !important;
    letter-spacing: -0.01em !important;
    margin-top: 2rem !important;
}

h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    color: #6d28d9 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Radio nav pill ── */
[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 0.5rem !important;
    background: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 50px !important;
    padding: 0.4rem !important;
    border: 1px solid rgba(124,58,237,0.15) !important;
    width: fit-content !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.08) !important;
}

[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.2rem !important;
    border-radius: 50px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    color: #6d28d9 !important;
}

[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.35) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 20px !important;
    padding: 1.4rem 1.8rem !important;
    border: 1px solid rgba(124,58,237,0.12) !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.07), 0 1px 4px rgba(0,0,0,0.04) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.13) !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #9333ea !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 2.4rem !important;
    font-weight: 500 !important;
    color: #1a1025 !important;
}

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.65) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(124,58,237,0.1) !important;
    padding: 1rem !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.07) !important;
    margin-bottom: 1.5rem !important;
}

/* ── Altair chart containers ── */
[data-testid="stVegaLiteChart"] {
    background: rgba(255,255,255,0.65) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(124,58,237,0.1) !important;
    padding: 1rem !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.07) !important;
    margin-bottom: 1.5rem !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.75) !important;
    border: 1.5px solid rgba(124,58,237,0.2) !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: #1a1025 !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.07) !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.4) !important;
    filter: brightness(1.05) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #7c3aed !important;
}

/* ── Markdown / AI recommendation box ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-family: 'DM Sans', sans-serif !important;
    line-height: 1.75 !important;
    color: #2d1b69 !important;
}

/* ── Sidebar (if used) ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(124,58,237,0.1) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(124,58,237,0.12) !important;
    margin: 2rem 0 !important;
}

/* ── Error / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid rgba(124,58,237,0.15) !important;
}

/* ── Subheader accent bar ── */
.subheader-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(219,39,119,0.08));
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #7c3aed;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    color: #6b5b95;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

.ai-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.06), rgba(219,39,119,0.04));
    border: 1.5px solid rgba(124,58,237,0.18);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="subheader-pill">✦ Powered by Gemini AI</div>', unsafe_allow_html=True)
st.title("AirPure AI Recommendation")
st.markdown('<p class="hero-sub">Intelligent air quality insights & purifier recommendations for India</p>', unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    DATA_URL = "https://huggingface.co/datasets/manaspateltech/AQI/resolve/main/aqi.csv"
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = [col.lower() for col in df.columns]
        df = df.drop(columns=['note', 'unit'])
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df.drop_duplicates(inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading data from URL: {e}")
        return pd.DataFrame()

df = load_data()

# ── Shared Plotly theme ───────────────────────────────────────────────────────
CHART_COLORS = ["#7c3aed", "#db2777", "#0891b2", "#059669", "#d97706", "#dc2626"]

def style_fig(fig, title=""):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#1a1025"),
        title=dict(
            text=title,
            font=dict(family="Fraunces, serif", size=18, color="#4c1d95"),
            x=0.02, xanchor="left"
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=11, color="#6b5b95"),
            linecolor="rgba(124,58,237,0.15)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(124,58,237,0.08)",
            zeroline=False,
            tickfont=dict(size=11, color="#6b5b95"),
        ),
        margin=dict(l=16, r=16, t=48, b=16),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(124,58,237,0.3)",
            font=dict(family="DM Sans, sans-serif", color="#1a1025"),
        ),
    )
    return fig

# ── Nav ───────────────────────────────────────────────────────────────────────
page = st.radio("Select Page", ["🌏 India Overview", "🗺️ Statewise AQI", "📍 Areawise AQI"], index=0)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Page: India Overview ──────────────────────────────────────────────────────
if page == "🌏 India Overview":
    st.header("India Air Quality Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg AQI (2022–2025)", f"{df['aqi_value'].mean():.1f}")
    with col2:
        st.metric("Most Polluted State", df.groupby('state')['aqi_value'].mean().idxmax())
    with col3:
        st.metric("Total Records", f"{len(df):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly trend
    monthly_aqi = df.resample('ME', on='date')['aqi_value'].mean().reset_index()
    fig = px.line(monthly_aqi, x='date', y='aqi_value', color_discrete_sequence=["#7c3aed"])
    fig.update_traces(line=dict(width=2.5), fill='tozeroy',
                      fillcolor='rgba(124,58,237,0.07)')
    fig = style_fig(fig, "Monthly Average AQI Across India")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        state_aqi = df.groupby('state')['aqi_value'].mean().reset_index()
        top_10_states = state_aqi.sort_values('aqi_value', ascending=False).head(10)
        fig = px.bar(top_10_states, x='state', y='aqi_value',
                     color='aqi_value',
                     color_continuous_scale=["#c4b5fd", "#7c3aed", "#4c1d95"])
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
        fig = style_fig(fig, "Top 10 Most Polluted States")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        area_aqi = df.groupby('area')['aqi_value'].mean().reset_index()
        top_10_areas = area_aqi.sort_values('aqi_value', ascending=False).head(10)
        fig = px.bar(top_10_areas, x='area', y='aqi_value',
                     color='aqi_value',
                     color_continuous_scale=["#fbcfe8", "#db2777", "#831843"])
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
        fig = style_fig(fig, "Top 10 Most Polluted Areas")
        st.plotly_chart(fig, use_container_width=True)

# ── Page: Statewise AQI ───────────────────────────────────────────────────────
elif page == "🗺️ Statewise AQI":
    st.header("Statewise AQI Explorer")

    states = sorted(df['state'].dropna().unique())
    state = st.selectbox("Choose a state", states, key="state_select")
    state_df = df[df['state'] == state]

    if not state_df.empty:
        st.subheader(f"AQI Trend — {state}")
        state_df_monthly = state_df.set_index('date').resample('ME')['aqi_value'].mean().reset_index()
        fig = px.line(state_df_monthly, x='date', y='aqi_value', color_discrete_sequence=["#7c3aed"])
        fig.update_traces(line=dict(width=2.5), fill='tozeroy',
                          fillcolor='rgba(124,58,237,0.07)')
        fig = style_fig(fig, f"Monthly AQI in {state}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prominent Pollutants")
        state_pollutant_df = state_df.dropna(subset=['prominent_pollutants'])
        state_pollutant_df = state_pollutant_df.assign(
            prominent_pollutants=state_pollutant_df['prominent_pollutants'].str.split(',')
        ).explode('prominent_pollutants')
        state_pollutant_df['prominent_pollutants'] = state_pollutant_df['prominent_pollutants'].str.strip()
        pollutants = state_pollutant_df['prominent_pollutants'].value_counts()

        if not pollutants.empty:
            pollutants_df = pollutants.reset_index()
            pollutants_df.columns = ['Pollutant', 'Count']

            chart = alt.Chart(pollutants_df).mark_bar(
                cornerRadiusTopLeft=6, cornerRadiusTopRight=6
            ).encode(
                x=alt.X('Pollutant:N', sort='-y', title="Pollutant",
                         axis=alt.Axis(labelAngle=-40, labelColor="#6b5b95",
                                       titleColor="#4c1d95", labelFontSize=11)),
                y=alt.Y('Count:Q', title="Occurrences",
                         axis=alt.Axis(gridColor="#e9d5ff", labelColor="#6b5b95",
                                       titleColor="#4c1d95")),
                color=alt.Color('Count:Q',
                                scale=alt.Scale(range=["#c4b5fd", "#7c3aed"]),
                                legend=None),
                tooltip=['Pollutant', 'Count']
            ).properties(
                background='transparent'
            ).configure_view(
                strokeWidth=0
            )
            st.altair_chart(chart, use_container_width=True)

            st.subheader("✦ AI Purifier Recommendation")
            avg_aqi_state = state_df['aqi_value'].mean()
            top_pollutants_list = pollutants.head().index.tolist()
            top_pollutants_str = ', '.join(top_pollutants_list)

            prompt = (
                f"Based on the average AQI of {avg_aqi_state:.2f} and the most prominent pollutants in {state} which are {top_pollutants_str}, "
                "provide a detailed air purifier recommendation. "
                "Specify the necessary filter types (e.g., HEPA, Activated Carbon) and explain why each is needed. "
                "Suggest a suitable CADR (Clean Air Delivery Rate) and a rationale for your choice."
            )

            if st.button(f"Get Recommendation for {state}"):
                with st.spinner('Generating your personalised recommendation…'):
                    recommendation = get_ai_recommendation(prompt)
                    st.markdown(recommendation)
                    with st.expander("AI debug (safe)"):
                        dbg = st.session_state.get("ai_debug", {})
                        st.write(f"Key source: {dbg.get('key_source', 'unknown')}")
                        st.write(f"Key fingerprint: {dbg.get('key_fingerprint', 'unknown')}")
                        st.write(f"Model used: {dbg.get('model_used', 'none')}")
                        st.write(f"Last API error: {dbg.get('last_error', 'none')}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No pollutant data available for this state.")
    else:
        st.info("No data available for the selected state.")

# ── Page: Areawise AQI ────────────────────────────────────────────────────────
elif page == "📍 Areawise AQI":
    st.header("Areawise AQI Explorer")

    areas = sorted(df['area'].dropna().unique())
    area = st.selectbox("Choose an area", areas, key="area_select")
    area_df = df[df['area'] == area]
    avg_aqi_area = area_df['aqi_value'].mean()

    if not area_df.empty:
        st.subheader(f"AQI Trend — {area}")
        area_df_monthly = area_df.set_index('date').resample('ME')['aqi_value'].mean().reset_index()
        fig = px.line(area_df_monthly, x='date', y='aqi_value', color_discrete_sequence=["#db2777"])
        fig.update_traces(line=dict(width=2.5), fill='tozeroy',
                          fillcolor='rgba(219,39,119,0.07)')
        fig = style_fig(fig, f"Monthly AQI in {area}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prominent Pollutants")
        area_pollutant_df = area_df.dropna(subset=['prominent_pollutants'])
        area_pollutant_df = area_pollutant_df.assign(
            prominent_pollutants=area_pollutant_df['prominent_pollutants'].str.split(',')
        ).explode('prominent_pollutants')
        area_pollutant_df['prominent_pollutants'] = area_pollutant_df['prominent_pollutants'].str.strip()
        pollutants = area_pollutant_df['prominent_pollutants'].value_counts()

        if not pollutants.empty:
            pollutants_df = pollutants.reset_index()
            pollutants_df.columns = ['Pollutant', 'Count']

            chart = alt.Chart(pollutants_df).mark_bar(
                cornerRadiusTopLeft=6, cornerRadiusTopRight=6
            ).encode(
                x=alt.X('Pollutant:N', sort='-y', title="Pollutant",
                         axis=alt.Axis(labelAngle=-40, labelColor="#6b5b95",
                                       titleColor="#831843", labelFontSize=11)),
                y=alt.Y('Count:Q', title="Occurrences",
                         axis=alt.Axis(gridColor="#fce7f3", labelColor="#6b5b95",
                                       titleColor="#831843")),
                color=alt.Color('Count:Q',
                                scale=alt.Scale(range=["#fbcfe8", "#db2777"]),
                                legend=None),
                tooltip=['Pollutant', 'Count']
            ).properties(
                background='transparent'
            ).configure_view(
                strokeWidth=0
            )
            st.altair_chart(chart, use_container_width=True)

            st.markdown('<div class="ai-box">', unsafe_allow_html=True)
            st.subheader("✦ AI Purifier Recommendation")
            top_pollutants_list = pollutants.head().index.tolist()
            top_pollutants_str = ', '.join(top_pollutants_list)

            prompt = (
                f"Based on the average AQI of {avg_aqi_area:.2f} and the most prominent pollutants in {area} which are {top_pollutants_str}, "
                "provide a detailed air purifier recommendation. "
                "Specify the necessary filter types (e.g., HEPA, Activated Carbon) and explain why each is needed. "
                "Suggest a suitable CADR (Clean Air Delivery Rate) and a rationale for your choice."
            )

            if st.button(f"Get Recommendation for {area}"):
                with st.spinner('Generating your personalised recommendation…'):
                    recommendation = get_ai_recommendation(prompt)
                    st.markdown(recommendation)
                    with st.expander("AI debug (safe)"):
                        dbg = st.session_state.get("ai_debug", {})
                        st.write(f"Key source: {dbg.get('key_source', 'unknown')}")
                        st.write(f"Key fingerprint: {dbg.get('key_fingerprint', 'unknown')}")
                        st.write(f"Model used: {dbg.get('model_used', 'none')}")
                        st.write(f"Last API error: {dbg.get('last_error', 'none')}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No pollutant data available for this area.")
    else:
        st.info("No data available for the selected area.")
