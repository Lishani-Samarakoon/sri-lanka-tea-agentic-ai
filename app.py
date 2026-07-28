"""Professional Streamlit interface for the Sri Lankan Tea Agent."""

from __future__ import annotations

import base64
import html
import textwrap
from pathlib import Path
from typing import Any

from openai import api_key
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from src.retriever import get_retriever

from src.agents.orchestrator import TeaOrchestrator
from src.config import (
    ANNUAL_EXPORTS_CSV,
    ANNUAL_PRODUCTION_CSV,
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    MONTHLY_PRODUCTION_CSV,
    get_groq_api_key,
    load_settings,
)
from src.llm_client import LLMService
from src.rag.build_index import find_pdf_files


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sri Lankan Tea Intelligence Agent",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DIAGRAMS_DIR = BASE_DIR / "diagrams"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def find_named_image(stem: str) -> Path:
    """
    Return the first image matching the requested file stem.

    Example:
        diagrams/card_annual.jpg
        diagrams/card_annual.png
    """
    for extension in IMAGE_EXTENSIONS:
        candidate = DIAGRAMS_DIR / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    # Return the preferred location even when the image is missing.
    # The interface will use a gradient fallback.
    return DIAGRAMS_DIR / f"{stem}.jpg"


BANNER_IMAGE = find_named_image("tea_banner")

QUESTION_CARDS: dict[str, dict[str, Any]] = {
    "annual": {
        "icon": "🌱",
        "title": "Which year had the highest annual tea production?",
        "question": "Which year had the highest annual tea production?",
        "image": find_named_image("card_annual"),
    },
    "production": {
        "icon": "📈",
        "title": "Compare annual tea production in 2023 and 2024.",
        "question": "Compare annual tea production in 2023 and 2024.",
        "image": find_named_image("card_compare"),
    },
    "exports": {
        "icon": "🚢",
        "title": "Compare tea export volume in 2023 and 2024.",
        "question": "Compare tea export volume in 2023 and 2024.",
        "image": find_named_image("card_export"),
    },
    "reports": {
        "icon": "📖",
        "title": "What do the official documents say about Sri Lankan tea exports?",
        "question": "What do the official documents say about Sri Lankan tea exports?",
        "image": find_named_image("card_reports"),
    },
}


# =========================================================
# HTML, IMAGE AND SCROLL HELPERS
# =========================================================

def render_html(content: str) -> None:
    """Render interface HTML without displaying HTML tags as text."""
    cleaned = textwrap.dedent(content).strip()

    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(
            cleaned,
            unsafe_allow_html=True,
        )


def image_to_data_uri(path: Path) -> str:
    """Convert a local image into a browser-readable data URI."""
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    mime_type = mime_types.get(
        path.suffix.lower(),
        "image/jpeg",
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def create_card_background(path: Path) -> str:
    """Create an image background or a gradient fallback for a question card."""
    if path.exists():
        image_uri = image_to_data_uri(path)

        return (
            "linear-gradient("
            "180deg,"
            "rgba(3,18,13,.06) 0%,"
            "rgba(3,22,16,.38) 44%,"
            "rgba(2,15,11,.97) 100%"
            "),"
            f"url('{image_uri}')"
        )

    return (
        "linear-gradient("
        "145deg,"
        "#0f5138 0%,"
        "#083b2c 55%,"
        "#041d16 100%"
        ")"
    )


def compact_model_name(model_name: str) -> str:
    """Create a short model label for the sidebar."""
    value = model_name.strip().split("/")[-1]

    if len(value) <= 20:
        return value

    return value[:17] + "..."


def scroll_to_top_once() -> None:
    """
    Reset the Streamlit page to the top after the first page load.

    Streamlit can preserve an old scroll position after a rerun or refresh.
    """
    components.html(
        """
        <script>
        function resetStreamlitScroll() {
            const parentWindow = window.parent;
            const parentDocument = parentWindow.document;

            parentWindow.scrollTo(0, 0);

            const selectors = [
                '[data-testid="stAppViewContainer"]',
                'section.main',
                '.main',
                '.stApp'
            ];

            selectors.forEach((selector) => {
                const element = parentDocument.querySelector(selector);

                if (element) {
                    element.scrollTo({
                        top: 0,
                        left: 0,
                        behavior: "auto"
                    });
                }
            });
        }

        resetStreamlitScroll();
        setTimeout(resetStreamlitScroll, 80);
        setTimeout(resetStreamlitScroll, 250);
        setTimeout(resetStreamlitScroll, 700);
        </script>
        """,
        height=0,
        width=0,
    )


# =========================================================
# CUSTOM CSS
# =========================================================

def inject_custom_css() -> None:
    if "startup_notice" not in st.session_state:
        st.toast("🍃 Tea Intelligence Agent is ready.", icon="✅")
        st.session_state.startup_notice = True
    """Apply the complete professional dashboard style."""
    if BANNER_IMAGE.exists():
        banner_uri = image_to_data_uri(BANNER_IMAGE)

        hero_background = (
            "linear-gradient("
            "90deg,"
            "rgba(2,18,13,.98) 0%,"
            "rgba(3,47,31,.90) 48%,"
            "rgba(2,14,10,.42) 100%"
            "),"
            f"url('{banner_uri}')"
        )
    else:
        hero_background = (
            "linear-gradient("
            "135deg,"
            "#052e24 0%,"
            "#064e3b 50%,"
            "#0f766e 100%"
            ")"
        )

    css = """
    <style>
    :root {
        --tea-text: #ecfdf5;
        --tea-light: #a7f3d0;
        --tea-green: #34d399;
        --tea-red: #fda4af;
        --tea-yellow: #fde68a;
        --tea-gold: #facc15;
    }

    * {
        box-sizing: border-box;
    }

    html {
        scroll-behavior: smooth;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 78% 4%,
                rgba(16, 185, 129, .11),
                transparent 31%
            ),
            linear-gradient(
                180deg,
                #06100c 0%,
                #081612 55%,
                #06100c 100%
            );
    }

    .block-container {
        max-width: 1380px;
        padding-top: 4.65rem !important;
        padding-bottom: 7rem;
    }

    header[data-testid="stHeader"] {
        height: 3.55rem !important;
        background: rgba(6, 16, 12, .96) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(110, 231, 183, .10);
        z-index: 999999 !important;
    }

    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        min-width: 375px !important;
        max-width: 375px !important;
        background:
            linear-gradient(
                180deg,
                #041d15 0%,
                #08281d 55%,
                #061811 100%
            );
        border-right: 1px solid rgba(110, 231, 183, .16);
    }

    section[data-testid="stSidebar"] > div {
        width: 375px !important;
        padding-top: 1rem;
    }

    .sidebar-brand {
        width: 100%;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                rgba(16, 185, 129, .22),
                rgba(6, 95, 70, .19)
            );
        border: 1px solid rgba(52, 211, 153, .30);
        box-shadow: 0 12px 28px rgba(0, 0, 0, .20);
    }

    .sidebar-brand-row {
        display: flex;
        align-items: center;
        gap: .7rem;
    }

    .sidebar-brand-icon {
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        border-radius: 14px;
        font-size: 1.45rem;
        background: rgba(16, 185, 129, .14);
        border: 1px solid rgba(110, 231, 183, .22);
    }

    .sidebar-brand-title {
        color: var(--tea-text);
        font-size: 1.06rem;
        font-weight: 850;
    }

    .sidebar-brand-subtitle {
        color: #d1fae5;
        font-size: .69rem;
    }

    .sidebar-brand-text {
        color: var(--tea-light);
        font-size: .73rem;
        line-height: 1.48;
        margin-top: .65rem;
    }

    .sidebar-section-title {
        color: #86efac;
        font-size: .72rem;
        font-weight: 850;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin: 1.2rem 0 .62rem;
    }

    /* Readiness cards */

    .status-card {
        width: 100%;
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr) auto;
        align-items: center;
        gap: .68rem;
        padding: .72rem .74rem;
        margin-bottom: .46rem;
        border-radius: 13px;
        background: rgba(12, 44, 33, .77);
        border: 1px solid rgba(148, 163, 184, .13);
    }

    .status-icon {
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 11px;
        font-size: 1.1rem;
        background: rgba(16, 185, 129, .11);
        border: 1px solid rgba(110, 231, 183, .17);
    }

    .status-label {
        color: #e2e8f0;
        font-size: .73rem;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .status-ready,
    .status-missing {
        font-size: .70rem;
        font-weight: 850;
        white-space: nowrap;
    }

    .status-ready {
        color: #6ee7b7;
    }

    .status-missing {
        color: var(--tea-red);
    }

    .status-note {
        color: #94a3b8;
        font-size: .64rem;
        line-height: 1.4;
        padding: 0 .25rem .55rem;
    }

    /* Agent cards */

    .agent-card {
        position: relative;
        width: 100%;
        min-height: 96px;
        padding: .76rem 5.4rem .76rem .76rem;
        margin-bottom: .54rem;
        border-radius: 15px;
        background:
            linear-gradient(
                135deg,
                rgba(15, 118, 110, .15),
                rgba(6, 78, 59, .12)
            );
        border: 1px solid rgba(94, 234, 212, .18);
    }

    .agent-heading {
        display: flex;
        align-items: center;
        gap: .56rem;
        margin-bottom: .35rem;
    }

    .agent-icon {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        border-radius: 50%;
        font-size: 1.08rem;
        background: rgba(6, 78, 59, .70);
        border: 1px solid rgba(110, 231, 183, .25);
    }

    .agent-name {
        color: #ecfdf5;
        font-size: .80rem;
        font-weight: 850;
    }

    .agent-role {
        color: #a1a1aa;
        font-size: .66rem;
        line-height: 1.42;
    }

    .agent-model {
        position: absolute;
        right: .62rem;
        top: 50%;
        transform: translateY(-50%);
        max-width: 4.6rem;
        padding: .30rem .40rem;
        text-align: center;
        border-radius: 8px;
        color: #67e8f9;
        background: rgba(8, 47, 73, .46);
        border: 1px solid rgba(34, 211, 238, .26);
        font-family: monospace;
        font-size: .56rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .pattern-list {
        display: flex;
        flex-wrap: wrap;
        gap: .33rem;
    }

    .pattern-badge {
        color: #a7f3d0;
        background: rgba(16, 185, 129, .12);
        border: 1px solid rgba(52, 211, 153, .21);
        border-radius: 999px;
        font-size: .62rem;
        padding: .31rem .50rem;
    }

    section[data-testid="stSidebar"] .stButton > button {
        color: #facc15;
        background: rgba(113, 63, 18, .10);
        border: 1px solid rgba(250, 204, 21, .56);
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        color: #fef08a;
        background: rgba(161, 98, 7, .18);
        border-color: #facc15;
    }

    /* Hero */

    .hero-shell {
        position: relative;
        z-index: 1;
        background-image: __HERO_BACKGROUND__;
        background-size: cover;
        background-position: center right;
        min-height: 455px;
        padding: 1.8rem 2.8rem 2.3rem;
        margin-top: 0 !important;
        margin-bottom: 1.55rem;
        border-radius: 26px;
        border: 1px solid rgba(110, 231, 183, .25);
        box-shadow:
            0 26px 60px rgba(0, 0, 0, .35),
            inset 0 0 0 1px rgba(255, 255, 255, .03);
        overflow: hidden;
    }

    .hero-topline {
        position: relative;
        z-index: 4;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-top: 0;
        margin-bottom: 1.8rem;
    }

    .official-label,
    .security-label {
        display: inline-flex;
        align-items: center;
        gap: .44rem;
        border-radius: 999px;
        padding: .42rem .84rem;
        font-size: .71rem;
        font-weight: 850;
        backdrop-filter: blur(8px);
    }

    .official-label {
        color: #d1fae5;
        background: rgba(16, 185, 129, .14);
        border: 1px solid rgba(52, 211, 153, .28);
    }

    .security-label {
        color: #86efac;
        background: rgba(22, 101, 52, .18);
        border: 1px solid rgba(74, 222, 128, .24);
    }

    .hero-content {
        position: relative;
        z-index: 3;
        max-width: 760px;
    }

    .hero-title {
        color: #ffffff;
        font-size: clamp(2.6rem, 5vw, 5rem);
        line-height: .98;
        font-weight: 920;
        letter-spacing: -.052em;
        margin: 0 0 1rem;
        text-shadow: 0 8px 30px rgba(0, 0, 0, .32);
    }

    .hero-title-accent {
        background:
            linear-gradient(
                90deg,
                #ffffff 0%,
                #86efac 46%,
                #4ade80 100%
            );
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-description {
        color: #d1fae5;
        max-width: 730px;
        margin-bottom: 1.3rem;
        font-size: .98rem;
        line-height: 1.7;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .62rem;
        max-width: 760px;
    }

    .metric-tile {
        display: flex;
        align-items: center;
        gap: .66rem;
        min-height: 86px;
        padding: .78rem;
        border-radius: 14px;
        color: #ecfdf5;
        background:
            linear-gradient(
                135deg,
                rgba(16, 70, 48, .76),
                rgba(4, 28, 20, .80)
            );
        border: 1px solid rgba(110, 231, 183, .22);
        box-shadow: 0 10px 24px rgba(0, 0, 0, .15);
        backdrop-filter: blur(8px);
    }

    .metric-icon {
        font-size: 1.60rem;
        flex-shrink: 0;
    }

    .metric-value {
        color: #86efac;
        font-size: 1.15rem;
        font-weight: 900;
        line-height: 1.05;
    }

    .metric-label {
        color: #d1fae5;
        font-size: .66rem;
        font-weight: 750;
        line-height: 1.28;
    }

    /* Question cards */

    .section-heading {
        color: #f0fdf4;
        font-size: 1.28rem;
        font-weight: 850;
        margin: 1.3rem 0 .25rem;
    }

    .section-description {
        color: #94a3b8;
        font-size: .84rem;
        margin-bottom: .85rem;
    }

    .question-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .86rem;
        margin-top: 1rem;
        margin-bottom: 1.35rem;
    }

    .question-card-link {
        color: inherit !important;
        text-decoration: none !important;
    }

    .question-card {
        position: relative;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 1rem;
        background-size: cover;
        background-position: center;
        border-radius: 18px;
        border: 1px solid rgba(234, 179, 8, .38);
        box-shadow: 0 14px 30px rgba(0, 0, 0, .25);
        overflow: hidden;
        transition:
            transform .20s ease,
            border-color .20s ease,
            box-shadow .20s ease;
    }

    .question-card::after {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(
                130deg,
                rgba(134, 239, 172, .03),
                rgba(234, 179, 8, .04)
            );
        pointer-events: none;
    }

    .question-card:hover {
        transform: translateY(-5px);
        border-color: rgba(250, 204, 21, .76);
        box-shadow: 0 20px 40px rgba(0, 0, 0, .34);
    }

    .question-icon {
        position: relative;
        z-index: 1;
        width: 47px;
        height: 47px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        font-size: 1.36rem;
        color: #ffffff;
        background: rgba(5, 46, 32, .82);
        border: 1px solid rgba(110, 231, 183, .30);
        backdrop-filter: blur(8px);
    }

    .question-card-text {
        position: relative;
        z-index: 1;
        color: #f0fdf4;
        font-size: .90rem;
        font-weight: 850;
        line-height: 1.5;
        text-shadow: 0 3px 12px rgba(0, 0, 0, .88);
    }

    .question-card-footer {
        position: relative;
        z-index: 1;
        margin-top: .64rem;
        color: #86efac;
        font-size: .67rem;
        font-weight: 850;
    }

    /* Chat and answers */

    .stButton > button {
        min-height: 44px;
        color: #d1fae5;
        background: rgba(6, 78, 59, .38);
        border: 1px solid rgba(52, 211, 153, .24);
        border-radius: 12px;
        transition: all .18s ease;
    }

    .stButton > button:hover {
        color: #ffffff;
        background: rgba(5, 150, 105, .36);
        border-color: #34d399;
        transform: translateY(-1px);
    }

    div[data-testid="stChatMessage"] {
        background: rgba(10, 35, 26, .64);
        border: 1px solid rgba(148, 163, 184, .13);
        border-radius: 18px;
        margin-bottom: .8rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, .13);
    }

    div[data-testid="stChatInput"] {
        border-color: rgba(52, 211, 153, .30);
    }

    .answer-approved,
    .answer-warning {
        display: inline-block;
        border-radius: 999px;
        padding: .32rem .65rem;
        margin-bottom: .7rem;
        font-size: .68rem;
        font-weight: 850;
    }

    .answer-approved {
        color: #6ee7b7;
        background: rgba(16, 185, 129, .13);
        border: 1px solid rgba(52, 211, 153, .25);
    }

    .answer-warning {
        color: var(--tea-yellow);
        background: rgba(245, 158, 11, .12);
        border: 1px solid rgba(251, 191, 36, .24);
    }

    .source-card {
        width: 100%;
        padding: .77rem .88rem;
        margin-bottom: .5rem;
        border-radius: 13px;
        background:
            linear-gradient(
                135deg,
                rgba(6, 78, 59, .31),
                rgba(15, 23, 42, .36)
            );
        border: 1px solid rgba(94, 234, 212, .17);
    }

    .source-title {
        color: #ccfbf1;
        font-size: .79rem;
        font-weight: 850;
        margin-bottom: .18rem;
        overflow-wrap: anywhere;
    }

    .source-meta {
        color: #94a3b8;
        font-size: .69rem;
        line-height: 1.4;
    }

    .app-footer {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: .5rem;
        margin-top: 1.4rem;
        padding: .9rem 0;
        color: #86efac;
        font-size: .70rem;
        border-top: 1px solid rgba(110, 231, 183, .12);
    }

    div[data-testid="stExpander"] {
        background: rgba(6, 18, 14, .40);
        border: 1px solid rgba(148, 163, 184, .13);
        border-radius: 13px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, .13);
        border-radius: 13px;
        overflow: hidden;
    }

    footer,
    #MainMenu {
        visibility: hidden;
    }

    @media (max-width: 1100px) {
        .metric-grid,
        .question-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 760px) {
        section[data-testid="stSidebar"] {
            min-width: 300px !important;
            max-width: 300px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 300px !important;
        }

        .block-container {
            padding-top: 4.2rem !important;
        }

        .hero-shell {
            min-height: 420px;
            padding: 1.5rem 1.35rem 1.8rem;
        }

        .hero-topline {
            align-items: flex-start;
            flex-direction: column;
            margin-bottom: 1.4rem;
        }

        .hero-description {
            font-size: .88rem;
        }

        .metric-grid,
        .question-grid {
            grid-template-columns: 1fr;
        }

        .question-card {
            min-height: 210px;
        }
    }
    </style>
    """

    css = css.replace(
        "__HERO_BACKGROUND__",
        hero_background,
    )

    render_html(css)


inject_custom_css()

if "initial_scroll_completed" not in st.session_state:
    scroll_to_top_once()
    st.session_state.initial_scroll_completed = True


# =========================================================
# BACKEND
# =========================================================

@st.cache_resource(show_spinner=False)
def get_orchestrator(api_key: str) -> TeaOrchestrator:
    """Create and cache the multi-agent system."""
    llm = LLMService(api_key=api_key)
    return TeaOrchestrator(llm=llm)


# =========================================================
# DATASET VALIDATION
# =========================================================

def clean_column_name(column_name: Any) -> str:
    """Clean one CSV column name."""
    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
    )


def validate_standard_csv(
    path: Path,
    required_columns: set[str],
) -> tuple[bool, str]:
    """Validate a normal CSV file."""
    try:
        if not path.exists():
            return False, f"{path.name} was not found."

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            return False, "The CSV contains no data rows."

        available = {
            clean_column_name(column)
            for column in dataframe.columns
        }

        missing = sorted(
            required_columns - available
        )

        if missing:
            return (
                False,
                "Missing: " + ", ".join(missing),
            )

        return True, f"{len(dataframe)} records"

    except Exception as error:
        return False, str(error)


def validate_monthly_csv(path: Path) -> tuple[bool, str]:
    """Validate monthly columns, numeric values and totals."""
    required = {
        "year",
        "month",
        "high_kg",
        "medium_kg",
        "low_kg",
        "total_kg",
        "source",
    }

    ready, note = validate_standard_csv(
        path,
        required,
    )

    if not ready:
        return ready, note

    try:
        dataframe = pd.read_csv(path)

        dataframe.columns = [
            clean_column_name(column)
            for column in dataframe.columns
        ]

        numeric_columns = [
            "year",
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
        ]

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )

        calculated = (
            dataframe["high_kg"]
            + dataframe["medium_kg"]
            + dataframe["low_kg"]
        )

        incorrect = dataframe[
            calculated != dataframe["total_kg"]
        ]

        if not incorrect.empty:
            rows = (
                incorrect.index + 2
            ).tolist()

            return (
                False,
                "Incorrect total in CSV row(s): "
                + ", ".join(map(str, rows)),
            )

        return True, f"{len(dataframe)} monthly records"

    except Exception as error:
        return False, str(error)


# =========================================================
# SIDEBAR COMPONENTS
# =========================================================

def render_status(
    icon: str,
    label: str,
    ready: bool,
    status_text: str,
    note: str = "",
) -> None:
    """Render one readiness status card."""
    status_class = (
        "status-ready"
        if ready
        else "status-missing"
    )

    render_html(
        f"""
        <div class="status-card">
            <div class="status-icon">
                {icon}
            </div>

            <div class="status-label">
                {html.escape(label)}
            </div>

            <div class="{status_class}">
                ● {html.escape(status_text)}
            </div>
        </div>
        """
    )

    if not ready and note:
        render_html(
            f"""
            <div class="status-note">
                {html.escape(note)}
            </div>
            """
        )


def render_agent_card(
    icon: str,
    name: str,
    description: str,
    model: str,
) -> None:
    """Render one AI agent card."""
    render_html(
        f"""
        <div class="agent-card">
            <div class="agent-heading">
                <div class="agent-icon">
                    {icon}
                </div>

                <div class="agent-name">
                    {html.escape(name)}
                </div>
            </div>

            <div class="agent-role">
                {html.escape(description)}
            </div>

            <div
                class="agent-model"
                title="{html.escape(model)}"
            >
                {html.escape(compact_model_name(model))}
            </div>
        </div>
        """
    )


# =========================================================
# QUESTION CARDS
# =========================================================

def render_question_cards() -> None:
    """Display four clickable visual question cards."""
    cards: list[str] = []

    for key, card in QUESTION_CARDS.items():
        background = create_card_background(
            card["image"]
        )

        cards.append(
            f"""
            <a
                class="question-card-link"
                href="?sample={html.escape(key)}"
                target="_self"
            >
                <div
                    class="question-card"
                    style="background-image:{background};"
                >
                    <div class="question-icon">
                        {card["icon"]}
                    </div>

                    <div>
                        <div class="question-card-text">
                            {html.escape(card["title"])}
                        </div>

                        <div class="question-card-footer">
                            Ask this question →
                        </div>
                    </div>
                </div>
            </a>
            """
        )

    render_html(
        '<div class="question-grid">'
        + "".join(cards)
        + "</div>"
    )


# =========================================================
# ANSWER DISPLAY
# =========================================================

def render_chart(chart: dict[str, Any] | None) -> None:
    """Render chart information returned by the Data Agent."""
    if not chart or not chart.get("data"):
        return

    dataframe = pd.DataFrame(chart["data"])

    x_column = chart.get("x")
    y_column = chart.get("y")

    if (
        x_column not in dataframe.columns
        or y_column not in dataframe.columns
    ):
        return

    chart_data = (
        dataframe[[x_column, y_column]]
        .copy()
        .set_index(x_column)
    )

    st.markdown("#### Visual analysis")

    if chart.get("type") == "line":
        st.line_chart(
            chart_data,
            use_container_width=True,
        )
    else:
        st.bar_chart(
            chart_data,
            use_container_width=True,
        )


def render_sources(
    sources: list[dict[str, Any]],
) -> None:
    """Render dataset and document sources."""
    if not sources:
        return

    st.markdown("#### Sources used")

    for source in sources:
        if source.get("type") == "dataset":
            title = source.get(
                "source",
                "Dataset",
            )
            metadata = "Verified numerical dataset"
        else:
            title = source.get(
                "document",
                "Official document",
            )
            metadata = (
                f"{source.get('source_id', 'Source')}"
                f" · Page {source.get('page', 'N/A')}"
                f" · Similarity "
                f"{source.get('similarity_score', 'N/A')}"
            )

        render_html(
            f"""
            <div class="source-card">
                <div class="source-title">
                    {html.escape(str(title))}
                </div>

                <div class="source-meta">
                    {html.escape(str(metadata))}
                </div>
            </div>
            """
        )


def render_evidence(
    evidence: list[dict[str, Any]],
) -> None:
    """Render retrieved PDF evidence."""
    if not evidence:
        return

    with st.expander(
        "View retrieved official-document evidence"
    ):
        for item in evidence:
            source_id = item.get(
                "source_id",
                "Source",
            )
            document = item.get(
                "document",
                "Official document",
            )
            page = item.get("page", "N/A")
            score = item.get(
                "similarity_score",
                "N/A",
            )

            st.markdown(
                f"**{source_id} — {document}**"
            )
            st.caption(
                f"Page {page} · Similarity {score}"
            )
            st.write(
                item.get(
                    "text",
                    "No text was returned.",
                )
            )
            st.divider()


def render_technical_details(
    result: dict[str, Any],
) -> None:
    """Display technical agent evidence for marking."""
    with st.expander(
        "How the AI agents worked"
    ):
        (
            communication_tab,
            model_tab,
            review_tab,
        ) = st.tabs(
            [
                "Agent communication",
                "Models and tools",
                "Router and review",
            ]
        )

        with communication_tab:
            st.json(result.get("trace", []))

        with model_tab:
            st.json(result.get("models", {}))

        with review_tab:
            st.write(
                "**Internal route:** "
                f"`{result.get('route', 'Unknown')}`"
            )

            review_result = (
                "Approved"
                if result.get("approved")
                else "Needs attention"
            )

            st.write(
                "**Review result:** "
                f"`{review_result}`"
            )
            st.json(result.get("router", {}))


def render_assistant_result(
    result: dict[str, Any],
) -> None:
    """Render a complete assistant response."""
    if result.get("approved"):
        badge = """
        <span class="answer-approved">
            ✓ Evidence checked and grounded
        </span>
        """
    else:
        badge = """
        <span class="answer-warning">
            ⚠ Evidence review needs attention
        </span>
        """

    render_html(badge)

    st.markdown(
        result.get(
            "answer",
            "No answer was returned.",
        )
    )

    issues = result.get("issues", [])

    if issues:
        with st.expander("System notes"):
            for issue in issues:
                st.write(f"• {issue}")

    data_result = result.get("data_result")

    if data_result:
        st.markdown("#### Data analysis")

        if data_result.get("summary"):
            st.write(data_result["summary"])

        if data_result.get("records"):
            st.dataframe(
                pd.DataFrame(
                    data_result["records"]
                ),
                use_container_width=True,
                hide_index=True,
            )

        render_chart(
            data_result.get("chart")
        )

    render_sources(
        result.get("sources", [])
    )
    render_evidence(
        result.get("evidence", [])
    )
    render_technical_details(result)


# =========================================================
# CHAT SESSION
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. I am your Sri Lankan Tea "
                "Intelligence Agent. Ask me about "
                "annual tea production, monthly "
                "production, tea exports, sales "
                "information, or official "
                "tea-industry documents."
            ),
            "result": None,
        }
    ]


def process_question(question: str) -> None:
    
    api_key = get_groq_api_key()

    if not api_key:
        st.warning(
            "⚠ LLM service is not configured. "
            "The app is running in limited mode."
    )
        return
    """Process one question through all agents."""
    cleaned = question.strip()

    if not cleaned:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": cleaned,
            "result": None,
        }
    )

    try: 
        api_key = get_groq_api_key()

        if not api_key:
            st.error("❌ GROQ API key is not configured.")
            return

        orchestrator = get_orchestrator(api_key)

        with st.spinner(
            "The agents are analysing data, "
            "retrieving evidence and reviewing "
            "the answer..."
        ):
            result = orchestrator.run(cleaned)

        answer = result.get(
            "answer",
            "The system did not return an answer.",
        )

    except Exception as error:
        result = None
        answer = (
            "The agent system encountered "
            f"an error: {error}"
        )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "result": result,
        }
    )


# =========================================================
# PROCESS QUESTION CARD CLICKS
# =========================================================

sample_parameter = st.query_params.get(
    "sample"
)

if isinstance(sample_parameter, list):
    sample_parameter = (
        sample_parameter[0]
        if sample_parameter
        else None
    )

if (
    sample_parameter
    and sample_parameter in QUESTION_CARDS
):
    selected_question = QUESTION_CARDS[
        sample_parameter
    ]["question"]

    # Clear before processing to prevent duplicate execution.
    st.query_params.clear()

    process_question(selected_question)
    st.rerun()


# =========================================================
# SYSTEM READINESS
# =========================================================

settings = load_settings()

try:
    pdf_count = len(find_pdf_files())
except Exception:
    pdf_count = 0

annual_ready, annual_note = validate_standard_csv(
    ANNUAL_PRODUCTION_CSV,
    {
        "year",
        "production_mn_kg",
        "source",
    },
)

monthly_ready, monthly_note = validate_monthly_csv(
    MONTHLY_PRODUCTION_CSV
)

export_ready, export_note = validate_standard_csv(
    ANNUAL_EXPORTS_CSV,
    {
        "year",
        "export_volume_mn_kg",
        "export_revenue_lkr_bn",
        "source",
    },
)

# FAISS now builds automatically in memory
rag_ready = True


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-row">
                <div class="sidebar-brand-icon">
                    🍃
                </div>

                <div>
                    <div class="sidebar-brand-title">
                        Tea Intelligence AI
                    </div>

                    <div class="sidebar-brand-subtitle">
                        Sri Lanka Tea Industry Agent
                    </div>
                </div>
            </div>

            <div class="sidebar-brand-text">
                Multi-agent analysis grounded in verified
                datasets and official Sri Lankan tea-industry
                publications.
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="sidebar-section-title">
            ⚡ System readiness
        </div>
        """
    )

    render_status(
        "📁",
        "Official document library",
        pdf_count >= 20,
        f"{pdf_count} documents",
        (
            ""
            if pdf_count >= 20
            else "At least 20 documents are required."
        ),
    )

    render_status(
        "📊",
        "Annual production data",
        annual_ready,
        "Ready" if annual_ready else "Problem",
        annual_note,
    )

    render_status(
        "🗓️",
        "Monthly production data",
        monthly_ready,
        "Ready" if monthly_ready else "Problem",
        monthly_note,
    )

    render_status(
        "🚚",
        "Tea export data",
        export_ready,
        "Ready" if export_ready else "Problem",
        export_note,
    )

    render_status(
        "🗄️",
        "FAISS knowledge index",
        True,
        "Builds automatically",
        "Index is generated in memory at runtime."
)

    render_html(
        """
        <div class="sidebar-section-title">
            👥 AI agent team
        </div>
        """
    )

    render_agent_card(
        "🧭",
        "Router Agent",
        (
            "Classifies the question, rejects "
            "unrelated requests and selects "
            "the correct specialist."
        ),
        settings.router_model,
    )

    render_agent_card(
        "📊",
        "Data Analysis Agent",
        (
            "Performs accurate analysis using "
            "pandas tools and official datasets."
        ),
        "pandas",
    )

    render_agent_card(
        "📚",
        "Document Retrieval Agent",
        (
            "Retrieves relevant passages from "
            "official tea-industry reports "
            "using FAISS RAG."
        ),
        settings.embedding_model,
    )

    render_agent_card(
        "🛡️",
        "Review Agent",
        (
            "Reviews answers for accuracy, "
            "grounding and source validation "
            "before delivery."
        ),
        settings.reasoning_model,
    )

    render_html(
        """
        <div class="sidebar-section-title">
            🧩 Architecture patterns
        </div>

        <div class="pattern-list">
            <span class="pattern-badge">
                🧭 Router
            </span>

            <span class="pattern-badge">
                🛠 Tool Use
            </span>

            <span class="pattern-badge">
                🛡 Reflection
            </span>

            <span class="pattern-badge">
                ⚙ Orchestrator
            </span>
        </div>
        """
    )

    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversation cleared. "
                    "What would you like to analyse?"
                ),
                "result": None,
            }
        ]
        st.rerun()


# =========================================================
# HERO
# =========================================================

render_html(
    """
    <div class="hero-shell">
        <div class="hero-topline">
            <div class="official-label">
                🇱🇰 Official Sri Lankan Tea
                Industry Intelligence
            </div>

            <div class="security-label">
                🛡️ Secure • Verified • Reliable
            </div>
        </div>

        <div class="hero-content">
            <div class="hero-title">
                Tea
                <span class="hero-title-accent">
                    Intelligence
                </span>
                <br>
                Agent
            </div>

            <div class="hero-description">
                A multi-agent AI system for analysing
                Sri Lankan tea production and exports,
                comparing official statistics, and
                retrieving evidence from official
                industry reports.
            </div>

            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="metric-icon">
                        🤖
                    </div>

                    <div>
                        <div class="metric-value">
                            4
                        </div>

                        <div class="metric-label">
                            Specialist<br>
                            AI Agents
                        </div>
                    </div>
                </div>

                <div class="metric-tile">
                    <div class="metric-icon">
                        📑
                    </div>

                    <div>
                        <div class="metric-value">
                            20+
                        </div>

                        <div class="metric-label">
                            Official<br>
                            Documents
                        </div>
                    </div>
                </div>

                <div class="metric-tile">
                    <div class="metric-icon">
                        🔍
                    </div>

                    <div>
                        <div class="metric-value">
                            RAG
                        </div>

                        <div class="metric-label">
                            Evidence-Based<br>
                            Retrieval
                        </div>
                    </div>
                </div>

                <div class="metric-tile">
                    <div class="metric-icon">
                        🛡️
                    </div>

                    <div>
                        <div class="metric-value">
                            ✓
                        </div>

                        <div class="metric-label">
                            Reviewed and<br>
                            Verified Answers
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
)


# =========================================================
# VISUAL QUESTION CARDS
# =========================================================

render_html(
    """
    <div class="section-heading">
        🌿 Ask the Tea Intelligence Agent
    </div>

    <div class="section-description">
        Select a visual question card or write
        your own message in the chat box below.
    </div>
    """
)

render_question_cards()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:
    role = message["role"]
    avatar = "👤" if role == "user" else "🍃"

    with st.chat_message(
        role,
        avatar=avatar,
    ):
        result = message.get("result")

        if role == "assistant" and result:
            render_assistant_result(result)
        else:
            st.markdown(message["content"])


# =========================================================
# FOOTER
# =========================================================

render_html(
    """
    <div class="app-footer">
        🍃 Your AI assistant for Sri Lankan
        tea-industry intelligence 🍃
    </div>
    """
)


# =========================================================
# CHAT INPUT
# =========================================================

typed_question = st.chat_input(
    "Ask about tea production, exports or official reports..."
)

if typed_question:
    process_question(typed_question)
    st.rerun()