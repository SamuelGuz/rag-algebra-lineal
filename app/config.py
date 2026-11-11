"""Configuración central: rutas y variables de entorno (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
MENU_PATH = DATA_DIR / "menu.json"
CHROMA_DIR = DATA_DIR / "chroma"
ORDERS_DB = DATA_DIR / "pedidos.sqlite3"
STATIC_DIR = BASE_DIR / "static"

# Modelo de embeddings: multilingüe (entiende español), entrenado para búsqueda; vectores de 768 dimensiones.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
COLLECTION_NAME = "menu"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Nivel de "razonamiento" del modelo: low responde más rápido (vacío = el del modelo)
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "low")

RESTAURANT_NAME = "La Fonda de la Abuela"
