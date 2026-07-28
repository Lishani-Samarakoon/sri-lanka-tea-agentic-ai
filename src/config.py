"""Project paths, model settings, and API-key configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]

# Load a local .env file when one exists.
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DOCUMENTS_DIR = BASE_DIR / "documents"

# You already have this folder at the project root.
ROOT_ANNUAL_REPORTS_DIR = BASE_DIR / "annual_reports"

# The RAG builder searches both existing locations.
RAG_SOURCE_DIRS = (
    ROOT_ANNUAL_REPORTS_DIR,
    DOCUMENTS_DIR,
)

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
EVALUATION_DIR = BASE_DIR / "evaluation"


def first_available_path(*paths: Path) -> Path:
    """
    Return the first path that already exists.

    When none exists, return the first path so error messages show the
    preferred file name.
    """
    for path in paths:
        if path.exists():
            return path

    return paths[0]


ANNUAL_PRODUCTION_CSV = PROCESSED_DATA_DIR / "tea_annual_production.csv"

# Supports both names in case you previously created one of them.
MONTHLY_PRODUCTION_CSV = first_available_path(
    PROCESSED_DATA_DIR / "tea_monthly_production_2025.csv",
    PROCESSED_DATA_DIR / "tea_monthly_production.csv",
)

ANNUAL_EXPORTS_CSV = PROCESSED_DATA_DIR / "tea_annual_exports.csv"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "tea_index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "tea_chunks.json"
INDEX_METADATA_PATH = VECTOR_STORE_DIR / "index_metadata.json"


@dataclass(frozen=True)
class Settings:
    """Application settings."""

    router_model: str = "openai/gpt-oss-20b"
    reasoning_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    minimum_similarity: float = 0.20


def streamlit_secret(name: str) -> str | None:
    """Read a Streamlit secret without breaking command-line scripts."""
    try:
        import streamlit as st

        value = st.secrets.get(name)

        if value is None:
            return None

        value = str(value).strip()
        return value or None

    except Exception:
        return None


def get_setting(name: str, default: str) -> str:
    """Read an environment variable, Streamlit secret, or default value."""
    environment_value = os.getenv(name)

    if environment_value and environment_value.strip():
        return environment_value.strip()

    secret_value = streamlit_secret(name)

    if secret_value:
        return secret_value

    return default


def get_groq_api_key() -> str:
    """Return the Groq API key or raise a helpful error."""
    key = os.getenv("GROQ_API_KEY") or streamlit_secret("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to "
            ".streamlit/secrets.toml for local Streamlit use."
        )

    return key.strip()


def load_settings() -> Settings:
    """Create the application settings object."""
    return Settings(
        router_model=get_setting(
            "ROUTER_MODEL",
            "openai/gpt-oss-20b",
        ),
        reasoning_model=get_setting(
            "REASONING_MODEL",
            "openai/gpt-oss-120b",
        ),
        embedding_model=get_setting(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        top_k=int(get_setting("TOP_K", "5")),
        minimum_similarity=float(
            get_setting("MINIMUM_SIMILARITY", "0.20")
        ),
    )