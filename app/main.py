"""Servidor web: API del chat, panel vectorial y cocina."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, vector_store
from app.agent import Session, build_trace, make_client
from app.menu import load_menu
from app.orders import STATES, OrderStore

log = logging.getLogger("rag")
orders = OrderStore()
sessions: dict[str, Session] = {}
_client = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    n = vector_store.index_menu()
    log.warning("Menú indexado en ChromaDB: %d platos", n)
    vector_store.menu_pca()
    vector_store.menu_hnsw()
    yield


app = FastAPI(title=config.RESTAURANT_NAME, lifespan=lifespan)


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Evita que el navegador use versiones viejas del HTML/JS/CSS."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(("/static", "/cocina")):
        response.headers["Cache-Control"] = "no-cache"
    return response


def get_client():
    global _client
    if _client is None:
        _client = make_client()
    return _client


def get_session(session_id: str) -> Session:
    if session_id not in sessions:
        sessions[session_id] = Session(orders)
    return sessions[session_id]


def friendly_error(e: Exception) -> str:
    text = str(e)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        if "per day" in text or "PerDay" in text:
            return (f"Se agotó la cuota diaria gratuita del modelo {config.GEMINI_MODEL}. "
                    "Cambia GEMINI_MODEL en el archivo .env por otro modelo o espera a mañana. "
                    "Mientras tanto, el buscador del laboratorio funciona sin IA.")
        return "La IA alcanzó su límite de mensajes por minuto (plan gratuito). Espera unos segundos y vuelve a escribir."
    if "API key" in text or "GEMINI_API_KEY" in text:
        return "Falta o no es válida la clave GEMINI_API_KEY en el archivo .env."
    return f"No pude conectarme con la IA: {text[:200]}"


class SearchIn(BaseModel):
    consulta: str
    categoria: str | None = None
    vegetariano: bool | None = None
    precio_max: int | None = None


class ChatIn(BaseModel):
    session_id: str
    message: str


class StatusIn(BaseModel):
    estado: str


@app.post("/api/chat")
def chat(body: ChatIn):
    session = get_session(body.session_id)
    try:
        reply = session.chat(body.message, get_client())
        error = None
    except Exception as e:  # sin clave, sin red, cuota agotada...
        log.exception("Error hablando con Gemini")
        reply, error = None, friendly_error(e)
    return {
        "reply": reply,
        "error": error,
        "cart": session.cart.to_dict(),
        "vector_trace": session.traces,
        "pedido": session.last_order,
    }


@app.post("/api/reset/{session_id}")
def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/api/menu/pca")
def menu_pca():
    ids, vectors = vector_store.menu_vectors()
    pca = vector_store.menu_pca()
    coords = pca.project(vectors)
    menu = load_menu()
    return {
        "dimensiones": int(vectors.shape[1]),
        "varianza_explicada": [round(float(x), 4) for x in pca.explained_variance_ratio],
        "puntos": [
            {"id": i, "nombre": menu[i]["nombre"], "categoria": menu[i]["categoria"],
             "xy": [round(float(c[0]), 4), round(float(c[1]), 4)],
             "xyz": [round(float(x), 4) for x in c]}
            for i, c in zip(ids, coords)
        ],
    }


@app.post("/api/search")
def search(body: SearchIn):
    """Búsqueda semántica directa (sin la IA): para explorar el espacio vectorial."""
    result = vector_store.search(body.consulta, k=5, categoria=body.categoria,
                                 vegetariano=body.vegetariano, precio_max=body.precio_max)
    return build_trace(result)


@app.get("/api/menu/matrix")
def menu_matrix():
    """La base de datos como matriz X (una fila por plato, una columna por dimensión)."""
    ids, X = vector_store.menu_vectors()
    menu = load_menu()
    return {
        "filas": [{"id": i, "nombre": menu[i]["nombre"], "categoria": menu[i]["categoria"]} for i in ids],
        "dimensiones": int(X.shape[1]),
        "X": [[round(float(x), 4) for x in row] for row in X],
    }


@app.get("/api/hnsw/graph")
def hnsw_graph():
    return vector_store.menu_hnsw().to_dict()


@app.get("/api/info")
def info():
    return {"restaurante": config.RESTAURANT_NAME, "modelo_embeddings": config.EMBEDDING_MODEL,
            "modelo_llm": config.GEMINI_MODEL, "platos": len(load_menu())}


@app.get("/api/orders")
def list_orders():
    return {"estados": STATES, "pedidos": orders.list()}


@app.patch("/api/orders/{order_id}")
def update_order(order_id: int, body: StatusIn):
    try:
        order = orders.set_status(order_id, body.estado)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if order is None:
        raise HTTPException(404, "Pedido no encontrado")
    return order


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/cocina")
def cocina():
    return FileResponse(config.STATIC_DIR / "cocina.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
