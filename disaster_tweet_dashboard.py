"""
================================================================================
 DISASTER_TWEET_DASHBOARD.PY

 Project      : NHIS Project 7 — Disaster Tweet Classification
 Author       : Yousuf S. R. Sakkaf
 Description  : Streamlit dashboard for the selected model (DistilBERT). A user
                types or pastes an English message (up to 280 characters) and
                the app says whether it reads like a real disaster report, an
                everyday message, or whether the model is not sure. The user can
                also pick one of eight ready-made examples from a dropdown.
 Run          : streamlit run disaster_tweet_dashboard.py
================================================================================
"""

import base64
import glob
import html
import os
import re
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ASSETS_DIR = "deploy_assets"
HF_REPO_ID = "ysakkaf/nhis-project7-distilbert"
LOCAL_MODEL_DIRS = [os.path.join(ASSETS_DIR, "distilbert_model"), "Assets/models/distilbert_model"]
MAX_LENGTH = 128                      # same limit the model was trained with
MAX_CHARS = 280                       # classic tweet length
UNSURE_LOW, UNSURE_HIGH = 0.45, 0.55  # disaster chance inside this band -> "Not sure"
SHORT_MESSAGE_WORDS = 3               # fewer cleaned words than this -> reliability note

BLUE, NAVY, TEAL, CORAL, AMBER = "#1D6FD8", "#12355B", "#14B8A6", "#F2545B", "#F5A623"
INK, MUTED, LINE = "#1B2B3A", "#5B6B7B", "#E3ECF6"

# Eight examples shown in the dropdown (3 disaster, 3 not a disaster, 2 "Not sure"), mixed in order.
# The dropdown shows only the text, so the learner is not told the answer in advance.
# Every example was checked with the deployed model: chances in the comments (saved model).
EXAMPLES = [
    "Wildfire spreading fast near the highway, evacuations ordered",   # 99% -> disaster
    "The students dance performance rocked the atmosphere",             # 4%  -> not a disaster
    "Ready for this storm",                                             # 49% -> not sure
    "Flood waters rising in the city, rescue boats are on the way",     # 99% -> disaster
    "She danced like a fire",                                           # 18% -> not a disaster
    "FATALITY!",                                                        # 54% -> not sure
    "Earthquake of magnitude 6.1 strikes off the coast, buildings collapsed",  # 99% -> disaster
    "This new song is fire, I have it on repeat",                       # 9%  -> not a disaster
]
EXAMPLE_PROMPT = "Select an example…"

LOGO_FILE = os.path.join(ASSETS_DIR, "logo_canary.png")
st.set_page_config(
    page_title="Disaster Tweet Checker",
    page_icon=LOGO_FILE if os.path.exists(LOGO_FILE) else "🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], button, textarea {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #EEF6FF 0%, #FFFFFF 55%);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1180px; }
footer { visibility: hidden; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, %%NAVY%% 0%, #1B4F8F 100%); }
[data-testid="stSidebar"] * { color: #EAF2FB; }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.sb-brand { display:flex; align-items:center; gap:12px; margin-bottom: 6px; }
.sb-brand img { width:46px; height:46px; border-radius:12px; }
.sb-brand b { font-size:1.12rem; color:#FFFFFF; line-height:1.15; }
.sb-sec { margin-top: 18px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,.16); }
.sb-sec h4 { margin: 0 0 8px 0; font-size: .95rem; letter-spacing: .02em; color:#FFFFFF; }
.sb-sec p, .sb-sec li { font-size: .9rem; line-height: 1.5; margin: 0 0 6px 0; color:#D6E4F4; }
.sb-sec ol { padding-left: 18px; margin: 0; }
.chip { display:inline-block; padding: 3px 11px; border-radius: 999px; font-size:.8rem; font-weight:700; margin: 2px 6px 4px 0; color:#fff !important; }
.chip.d { background: %%CORAL%%; } .chip.n { background: %%TEAL%%; } .chip.u { background: %%AMBER%%; color:#3A2A00 !important; }
.sb-stat { background: rgba(255,255,255,.10); border-radius: 12px; padding: 10px 12px; font-size:.9rem; color:#FFFFFF; }
.sb-stat b { color:#FFFFFF; }

/* ---------- Header ---------- */
.hero { background: linear-gradient(120deg, %%BLUE%% 0%, #1594C9 55%, %%TEAL%% 100%); border-radius: 20px;
        padding: 26px 30px; display:flex; align-items:center; gap:20px; color:#fff; margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(29,111,216,.22); }
.hero img { width:76px; height:76px; border-radius:18px; box-shadow: 0 0 0 4px rgba(255,255,255,.35); }
.hero h1 { margin:0; font-size: 2.15rem; font-weight:800; color:#fff; letter-spacing:-.01em; line-height:1.15; }
.hero p { margin:6px 0 0 0; font-size:1.05rem; color:#EAF6FF; }

/* ---------- Cards ---------- */
.card-title { font-size:1.15rem; font-weight:700; color:%%NAVY%%; margin: 0 0 4px 0; }
.card-sub { font-size:.92rem; color:%%MUTED%%; margin: 0 0 10px 0; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; border-color: %%LINE%% !important; }

[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; box-shadow: none !important; }
textarea { border-radius: 12px !important; font-size: 1.02rem !important; }
textarea:focus { border-color: %%BLUE%% !important; box-shadow: 0 0 0 2px rgba(29,111,216,.2) !important; }

/* Primary button */
[data-testid="stBaseButton-primary"], button[kind="primary"],
[data-testid="stBaseButton-primaryFormSubmit"], button[kind="primaryFormSubmit"] {
    background: linear-gradient(90deg, %%BLUE%%, %%TEAL%%); border: none; color:#fff; font-weight:700;
    border-radius: 12px; padding: .65rem 1rem; font-size: 1.02rem; }
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover, button[kind="primaryFormSubmit"]:hover { filter: brightness(1.06); }
/* Example chips */
[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    border-radius: 999px; border:1px solid #BBD3F0; background:#F3F8FF; color:%%BLUE%%; font-size:.82rem; font-weight:600; padding:.25rem .7rem; }
[data-testid="stBaseButton-secondary"]:hover { border-color:%%BLUE%%; background:#E6F0FF; color:%%NAVY%%; }

/* ---------- Result ---------- */
.result { border-radius:16px; padding:18px 20px; }
.result .tag { font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform: uppercase; }
.result .head { font-size:1.55rem; font-weight:800; margin: 2px 0 6px 0; }
.result .body { font-size:1rem; color:%%INK%%; line-height:1.5; }
.quote { background:#F6F9FD; border-radius:12px; padding:10px 14px; font-size:.95rem; color:%%MUTED%%; margin: 12px 0 4px 0; font-style: italic; }
.gauge-label { display:flex; justify-content:space-between; font-size:.85rem; color:%%MUTED%%; margin-top:16px; }
.gauge { position:relative; height:16px; border-radius:999px; background:#E8EEF5; margin-top:6px; overflow:visible; }
.gauge .fill { position:absolute; left:0; top:0; bottom:0; border-radius:999px; opacity:.9; }
.gauge .band { position:absolute; top:-3px; bottom:-3px; background: rgba(245,166,35,.28); border:1px dashed %%AMBER%%; border-radius:6px; }
.gauge .dot { position:absolute; top:-4px; width:24px; height:24px; margin-left:-12px; border-radius:50%; background:#fff; box-shadow: 0 2px 6px rgba(0,0,0,.25); }
.gauge-scale { display:flex; justify-content:space-between; font-size:.75rem; color:#8A99A8; margin-top:8px; }
.pct { font-size: 2.2rem; font-weight:800; line-height:1; }
.note { font-size:.9rem; color:#7A5600; background:#FFF6E0; border-radius:10px; padding:8px 12px; margin-top:12px; }
.placeholder { text-align:center; padding: 38px 12px; color:%%MUTED%%; }
.placeholder img { width:84px; height:84px; opacity:.9; border-radius:20px; margin-bottom:10px; }
.recent { font-size:.88rem; color:%%INK%%; display:flex; align-items:center; gap:8px; padding:5px 0; border-bottom:1px solid #F0F4F9; }
.recent .dotc { width:10px; height:10px; border-radius:50%; flex:none; }
.recent span.t { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.recent span.p { color:%%MUTED%%; font-weight:600; }

/* ---------- Performance ---------- */
.kpi { background:#fff; border:1px solid %%LINE%%; border-radius:14px; padding:14px 14px 12px; text-align:center; height:100%; }
.kpi .v { font-size:1.9rem; font-weight:800; }
.kpi .l { font-size:.92rem; font-weight:700; color:%%NAVY%%; }
.kpi .s { font-size:.8rem; color:%%MUTED%%; margin-top:2px; }
table.cmp { width:100%; border-collapse: collapse; font-size:.92rem; }
table.cmp th { background:#EAF3FF; color:%%NAVY%%; text-align:left; padding:8px 10px; }
table.cmp td { padding:7px 10px; border-bottom:1px solid #EEF3F9; }
table.cmp tr.win td { background:#E9F9F6; font-weight:700; }
.foot { text-align:center; color:%%MUTED%%; font-size:.85rem; margin-top: 26px; }
</style>
"""
for token, value in {"BLUE": BLUE, "NAVY": NAVY, "TEAL": TEAL, "CORAL": CORAL, "AMBER": AMBER,
                     "INK": INK, "MUTED": MUTED, "LINE": LINE}.items():
    CSS = CSS.replace(f"%%{token}%%", value)
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def image_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except OSError:
        return ""


LOGO_B64 = image_b64(LOGO_FILE)
LOGO_TAG = f'<img src="data:image/png;base64,{LOGO_B64}" alt="logo">' if LOGO_B64 else ""


@st.cache_data(show_spinner=False)
def load_bundle():
    """Final evaluation results saved by the notebook (None if not available)."""
    paths = sorted(glob.glob(os.path.join(ASSETS_DIR, "final_evaluation_bundle*.pkl")))
    if not paths:
        return None
    try:
        return joblib.load(paths[-1])
    except Exception:
        return None


# --- Text cleaning: identical to the notebook's clean_and_tokenize ----------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
CONTRACTIONS = {"gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of", "outta": "out of"}


@st.cache_resource(show_spinner=False)
def load_text_tools():
    import nltk
    for package in ("stopwords", "wordnet", "punkt", "punkt_tab"):
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    return set(stopwords.words("english")), WordNetLemmatizer(), word_tokenize


def clean_text(text):
    """Return the cleaned text the model was trained on (same steps as the notebook)."""
    stop_words, lemmatizer, word_tokenize = load_text_tools()
    t = html.unescape(str(text))
    for informal, expanded in CONTRACTIONS.items():
        t = re.sub(rf"\b{informal}\b", expanded, t, flags=re.IGNORECASE)
    t = URL_PATTERN.sub("", t)
    t = MENTION_PATTERN.sub("", t)
    t = t.replace("#", " ")
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = t.lower()
    tokens = [w for w in word_tokenize(t) if w not in stop_words and len(w) > 1]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(w for w in tokens if len(w) > 1)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fine-tuned DistilBERT once per server (local folder first, then Hugging Face)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    source = next((d for d in LOCAL_MODEL_DIRS if os.path.isfile(os.path.join(d, "config.json"))), HF_REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.eval()
    return tokenizer, model


def disaster_chance(cleaned_text):
    """Model's estimated chance (0–1) that the message describes a real disaster."""
    import torch
    tokenizer, model = load_model()
    encoded = tokenizer(cleaned_text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**encoded).logits
    return float(torch.softmax(logits, dim=-1)[0, 1])


def looks_english(text):
    letters = re.findall(r"[^\W\d_]", text)
    if not letters:
        return True
    return sum(ch.isascii() for ch in letters) / len(letters) >= 0.8


def verdict_for(chance):
    if chance > UNSURE_HIGH:
        return dict(key="disaster", color=CORAL, bg="#FFF1F1", tag="Result", head="Looks like a disaster report",
                    body="The message reads like someone describing a real emergency or its effects.")
    if chance < UNSURE_LOW:
        return dict(key="normal", color=TEAL, bg="#EAFAF7", tag="Result", head="Not a disaster report",
                    body="The message reads like everyday talk, news or a figure of speech, not a real emergency.")
    return dict(key="unsure", color=AMBER, bg="#FFF7E6", tag="Result", head="Not sure",
                body="The AI is almost evenly split on this one. Treat it as a coin-flip and check the original source.")


def gauge_html(chance, color):
    pos = min(max(chance * 100, 2), 98)
    return f"""
    <div class="gauge-label"><span>Chance this describes a real disaster</span></div>
    <div class="gauge">
        <div class="band" style="left:{UNSURE_LOW*100}%;width:{(UNSURE_HIGH-UNSURE_LOW)*100}%;"></div>
        <div class="fill" style="width:{pos}%;background:{color};"></div>
        <div class="dot" style="left:{pos}%;border:5px solid {color};"></div>
    </div>
    <div class="gauge-scale"><span>0% — very unlikely</span><span>50%</span><span>100% — very likely</span></div>
    """


def flat(markup):
    """Remove line indentation so Markdown never mistakes indented HTML for a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def set_example(text):
    st.session_state["tweet_text"] = text
    st.session_state["example_text"] = text
    try:
        run_check(text)
    except Exception as exc:
        st.session_state["result"] = {"error": str(exc)}


def pick_example():
    """Dropdown callback: run the check as soon as an example is selected."""
    chosen = st.session_state.get("example_pick")
    if chosen:
        set_example(chosen)


def on_check_click():
    """If the learner typed their own text, clear the dropdown so it does not look like an example is active."""
    if st.session_state.get("tweet_text", "").strip() != (st.session_state.get("example_pick") or ""):
        st.session_state["example_pick"] = None


def run_check(raw_text):
    """Clean, score and store the result of one check in session state."""
    text = raw_text.strip()
    cleaned = clean_text(text)
    result = {"text": text, "cleaned": cleaned, "english": looks_english(text), "chance": None}
    if cleaned.strip():
        result["chance"] = disaster_chance(cleaned)
        history = st.session_state.setdefault("history", [])
        v = verdict_for(result["chance"])
        history.insert(0, {"text": text, "color": v["color"], "chance": result["chance"]})
        del history[5:]
    st.session_state["result"] = result


# ------------------------------------------------------------------------------
# Sidebar — plain-language project information
# ------------------------------------------------------------------------------
bundle = load_bundle()
reliability_html = (
    "In our test on tweets the AI had never seen, the answer was correct about "
    f"<b>{bundle['test_metrics'].sort_values('Rank').iloc[0]['Test Accuracy'] * 100:.0f} times out of 100</b>."
    if bundle is not None
    else "The AI is right most of the time, but not always."
)
n_tested = f"{bundle['n_test']:,}" if bundle is not None else "1,498"

with st.sidebar:
    st.markdown(flat(f"""
    <div class="sb-brand">{LOGO_TAG}<b>Disaster Tweet<br>Checker</b></div>

    <div class="sb-sec">
      <h4>ABOUT THE PROJECT</h4>
      <p>During floods, fires and earthquakes, people post about them online within minutes. But many posts
      use dramatic words in everyday ways, like "this song is fire".</p>
      <p>This tool reads a short message and tells you whether it looks like a report of a real emergency,
      or just ordinary talk. It was built to show how computers can help sort useful emergency information
      from the noise.</p>
    </div>

    <div class="sb-sec">
      <h4>HOW TO USE IT</h4>
      <ol>
        <li>Select an example from the dropdown, or type your own message in English (up to {MAX_CHARS} characters).</li>
        <li>Press <b>Check message</b>. Examples are checked as soon as you select them.</li>
        <li>The answer appears on the right, with a bar showing the chance (from 0% to 100%) that the message describes a real disaster.</li>
      </ol>
    </div>

    <div class="sb-sec">
      <h4>WHAT THE ANSWERS MEAN</h4>
      <span class="chip d">Disaster report</span>
      <span class="chip n">Not a disaster</span>
      <span class="chip u">Not sure</span>
      <p style="margin-top:6px;">"Not sure" appears when the AI is almost evenly split, so you know when to be extra careful.</p>
    </div>

    <div class="sb-sec">
      <h4>HOW RELIABLE IS IT?</h4>
      <div class="sb-stat">{reliability_html}<br><span style="font-size:.8rem;">Tested on {n_tested} real tweets.</span></div>
      <p style="margin-top:8px;">That means roughly 1 answer in 5 can be wrong. Never rely on it alone in an emergency.</p>
    </div>

    <div class="sb-sec">
      <h4>GOOD TO KNOW</h4>
      <p>• English messages only.<br>• Full sentences work better than one or two words.<br>
      • It judges the words in the message, not whether the event is really happening.<br>
      • Figures of speech (like "on fire") can sometimes fool it.<br>
      </p>
    </div>

    <div class="sb-sec">
      <h4>PROJECT</h4>
      <p>NHIS Project 7 — Disaster Tweet Classification<br>Built by Yousuf S. R. Sakkaf<br>
      Made with Python and Streamlit.</p>
    </div>
    """), unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.markdown(flat(f"""
<div class="hero">
    {LOGO_TAG}
    <div>
        <h1>Disaster Tweet Checker</h1>
    </div>
</div>
"""), unsafe_allow_html=True)

@st.cache_resource(show_spinner="Starting up the AI. This happens once and can take a minute…")
def warm_up():
    load_text_tools()
    load_model()
    disaster_chance("warm up")
    return True

warm_up()

st.session_state.setdefault("tweet_text", "")

col_in, col_out = st.columns([1.1, 1], gap="large")

# ------------------------------------------------------------------------------
# Input
# ------------------------------------------------------------------------------
with col_in:
    with st.container(border=True):
        st.markdown('<p class="card-title">Enter a message</p>'
                    f'<p class="card-sub">English only, up to {MAX_CHARS} characters, like a tweet.</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="card-sub" style="margin-bottom:6px;">Select an example from the dropdown, '
                    'or type your own message below:</p>', unsafe_allow_html=True)
        st.selectbox("Examples", EXAMPLES, index=None, placeholder=EXAMPLE_PROMPT, key="example_pick",
                     on_change=pick_example, label_visibility="collapsed")

        # A form commits the text and the click together, so one press of the button (or Ctrl+Enter) always works
        with st.form("check_form", border=False):
            st.text_area(
                "Message", key="tweet_text", max_chars=MAX_CHARS, height=150,
                placeholder="e.g. Flood waters rising in the city, rescue boats are on the way…",
                label_visibility="collapsed",
            )
            check_clicked = st.form_submit_button("Check message", type="primary", use_container_width=True,
                                                  on_click=on_check_click)

# ------------------------------------------------------------------------------
# Run the check
# ------------------------------------------------------------------------------
if check_clicked:
    message = st.session_state["tweet_text"].strip()
    if message:
        st.session_state.pop("example_text", None)
    else:
        message = st.session_state.pop("example_text", "")
    if not message:
        st.session_state["result"] = {"empty": True}
    else:
        with st.spinner("Reading the message…"):
            try:
                run_check(message)
            except Exception as exc:  # keep the page alive and explain in plain words
                st.session_state["result"] = {"error": str(exc)}

# ------------------------------------------------------------------------------
# Result
# ------------------------------------------------------------------------------
with col_out:
    with st.container(border=True):
        st.markdown('<p class="card-title">Result</p>', unsafe_allow_html=True)
        result = st.session_state.get("result")

        if result is None:
            st.markdown(flat(f"""
            <div class="placeholder"><svg width="84" height="84" viewBox="0 0 84 84" fill="none"><rect x="8" y="10" width="68" height="46" rx="14" fill="#EAF3FF" stroke="#1D6FD8" stroke-width="3" stroke-dasharray="6 5"/><path d="M26 56 v14 l16 -14 z" fill="#EAF3FF" stroke="#1D6FD8" stroke-width="3" stroke-linejoin="round"/><circle cx="30" cy="33" r="4.5" fill="#14B8A6"/><circle cx="42" cy="33" r="4.5" fill="#1D6FD8"/><circle cx="54" cy="33" r="4.5" fill="#F2545B"/></svg><br><b>Your result will appear here</b><br>
            Select an example or enter your own message, then press <b>Check message</b>.</div>"""), unsafe_allow_html=True)

        elif result.get("empty"):
            st.info("Please select an example or type your own message first.")

        elif "error" in result:
            st.error("Something went wrong while loading the AI. Please try again in a moment.")
            with st.expander("Technical details"):
                st.code(result["error"])

        elif result["chance"] is None:
            st.markdown(flat("""
            <div class="result" style="border-left:8px solid #6B7B8C;background:#F3F6F9;">
              <div class="tag" style="color:#6B7B8C;">Result</div><div class="head" style="color:#6B7B8C;">Nothing to read</div>
              <div class="body">After removing links, @mentions, numbers and symbols, no English words were left.
              Please enter a message with some words in English.</div></div>"""), unsafe_allow_html=True)

        else:
            v = verdict_for(result["chance"])
            st.markdown(flat(f"""
            <div class="result" style="border-left:8px solid {v['color']};background:{v['bg']};">
              <div class="tag" style="color:{v['color']};">{v['tag']}</div>
              <div class="head" style="color:{v['color']};">{v['head']}</div>
              <div class="body">{v['body']}</div>
            </div>
            <div style="display:flex;align-items:baseline;gap:10px;margin-top:16px;">
              <span class="pct" style="color:{v['color']};">{result['chance'] * 100:.0f}%</span>
              <span style="color:{MUTED};">chance of a real disaster report</span>
            </div>
            {gauge_html(result['chance'], v['color'])}
            <div class="quote">“{html.escape(result['text'])}”</div>
            """), unsafe_allow_html=True)

            notes = []
            if not result["english"]:
                notes.append("This message may not be in English, so the answer may be unreliable.")
            if len(result["cleaned"].split()) < SHORT_MESSAGE_WORDS:
                notes.append("Very short messages give less reliable answers. Try a full sentence.")
            for n in notes:
                st.markdown(f'<div class="note">{n}</div>', unsafe_allow_html=True)

        history = st.session_state.get("history", [])
        if len(history) > 1:
            st.markdown('<p class="card-sub" style="margin-top:16px;"><b>Recent checks</b></p>', unsafe_allow_html=True)
            rows = "".join(
                f'<div class="recent"><span class="dotc" style="background:{h["color"]};"></span>'
                f'<span class="t">{html.escape(h["text"])}</span><span class="p">{h["chance"] * 100:.0f}%</span></div>'
                for h in history
            )
            st.markdown(rows, unsafe_allow_html=True)

st.write("")
st.markdown('<div class="foot">This is a learning project, not an official alert system. '
            'For emergencies, always follow your local authorities.</div>', unsafe_allow_html=True)
