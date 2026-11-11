"""Agente conversacional (Gemini) que toma pedidos usando RAG.

Flujo de un turno:
  1. El mensaje del cliente se envía a Gemini junto con el historial.
  2. Si Gemini necesita platos, llama a la herramienta `buscar_platos`:
     esa es la parte de Recuperación (búsqueda semántica en ChromaDB).
  3. Los platos encontrados vuelven a Gemini como contexto (Aumentación).
  4. Gemini redacta la respuesta o maneja el carrito (Generación).
"""
from google import genai
from google.genai import types

from app import config, vector_store
from app.cart import Cart, CartError
from app.menu import load_menu
from app.orders import OrderStore

MAX_TOOL_ROUNDS = 6

CATEGORIES = sorted({d["categoria"] for d in load_menu().values()})

SYSTEM_PROMPT = f"""Eres el asistente de pedidos por WhatsApp de "{config.RESTAURANT_NAME}", un restaurante de comida colombiana.
Hablas en español colombiano, cálido y breve, como en un chat de WhatsApp (mensajes cortos, algún emoji ocasional).

Reglas obligatorias:
- NO conoces el menú de memoria. Siempre que el cliente pida, pregunte o describa comida o bebida, usa la herramienta `buscar_platos` con una descripción clara de lo que busca.
- Nunca inventes platos, precios ni ingredientes. Solo ofrece platos devueltos por `buscar_platos` y usa exactamente sus `id`.
- Cada resultado trae una `similitud` (coseno, de -1 a 1). Si ningún resultado es realmente lo que el cliente pidió (por ejemplo pidió sushi y salen platos sin relación), dile con amabilidad que no lo tienen y sugiere lo más parecido.
- Para agregar al carrito usa `agregar_al_carrito` con el id. Si el cliente nombra un plato, búscalo primero para obtener su id.
- Categorías disponibles para filtrar: {", ".join(CATEGORIES)}.
- Antes de confirmar, muestra el resumen del carrito con el total y pide el nombre del cliente. Solo llama a `confirmar_pedido` cuando el cliente confirme explícitamente y tengas su nombre.
- Precios en pesos colombianos con punto de miles, por ejemplo $32.000.
"""

S = types.Schema
T = types.Type

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="buscar_platos",
        description="Búsqueda semántica en el menú: devuelve los platos cuyo significado es más parecido a la consulta.",
        parameters=S(type=T.OBJECT, properties={
            "consulta": S(type=T.STRING, description="Qué busca el cliente, en lenguaje natural. Ej: 'sopa caliente con pollo'"),
            "categoria": S(type=T.STRING, enum=CATEGORIES, description="Filtrar por categoría (opcional)"),
            "vegetariano": S(type=T.BOOLEAN, description="true para solo platos sin carne (opcional)"),
            "precio_max": S(type=T.INTEGER, description="Precio máximo en pesos (opcional)"),
        }, required=["consulta"]),
    ),
    types.FunctionDeclaration(
        name="agregar_al_carrito",
        description="Agrega un plato del menú al carrito del cliente.",
        parameters=S(type=T.OBJECT, properties={
            "id": S(type=T.STRING, description="id del plato, tal como lo devolvió buscar_platos"),
            "cantidad": S(type=T.INTEGER, description="Unidades a agregar (por defecto 1)"),
            "nota": S(type=T.STRING, description="Instrucción especial, ej: 'sin cebolla' (opcional)"),
        }, required=["id"]),
    ),
    types.FunctionDeclaration(
        name="quitar_del_carrito",
        description="Quita un plato del carrito. Sin cantidad, lo quita por completo.",
        parameters=S(type=T.OBJECT, properties={
            "id": S(type=T.STRING),
            "cantidad": S(type=T.INTEGER, description="Unidades a quitar (opcional)"),
        }, required=["id"]),
    ),
    types.FunctionDeclaration(
        name="ver_carrito",
        description="Devuelve el contenido actual del carrito y el total.",
    ),
    types.FunctionDeclaration(
        name="confirmar_pedido",
        description="Confirma el pedido y lo envía a la cocina. Solo después de que el cliente lo confirme.",
        parameters=S(type=T.OBJECT, properties={
            "nombre_cliente": S(type=T.STRING, description="Nombre de quien hace el pedido"),
        }, required=["nombre_cliente"]),
    ),
])


def build_trace(result: dict) -> dict:
    """Arma la 'traza vectorial' de una búsqueda para el laboratorio de álgebra."""
    pca = vector_store.menu_pca()
    q = result["vector"]
    ids, X = vector_store.menu_vectors()
    index = {dish_id: i for i, dish_id in enumerate(ids)}
    coords = pca.project(X)
    scores = X @ q  # búsqueda exacta: s = X·q (un producto punto por fila)
    r6 = lambda x: round(float(x), 6)
    return {
        "consulta": result["consulta"],
        "filtros": result["filtros"],
        "dimensiones": int(q.shape[0]),
        "norma": r6(result["norma"]),
        "norma_cruda": r6(result["norma_cruda"]),
        "vector": [round(float(x), 5) for x in q],
        "consulta_xy": [round(float(x), 5) for x in pca.project(q)[0][:2]],
        "consulta_xyz": [round(float(x), 5) for x in pca.project(q)[0]],
        "scores": {dish_id: r6(scores[i]) for i, dish_id in enumerate(ids)},
        "candidatos": result["candidatos"],
        "resultados": [
            {
                "id": r["id"],
                "nombre": r["plato"]["nombre"],
                "categoria": r["plato"]["categoria"],
                "precio": r["plato"]["precio"],
                "distancia_chroma": r6(r["distancia_chroma"]),
                "similitud": r6(1 - r["distancia_chroma"]),
                "coseno_manual": r6(r["coseno_manual"]),
                "producto_punto": r6(r["producto_punto"]),
                "xy": [round(float(x), 5) for x in coords[index[r["id"]]][:2]],
            }
            for r in result["resultados"]
        ],
        "vecinos": [r["id"] for r in result["resultados"]],
        "hnsw": vector_store.menu_hnsw().search(q, k=len(result["resultados"]) or 5),
    }


class Session:
    """Estado de una conversación: historial para Gemini, carrito y trazas del turno."""

    def __init__(self, orders: OrderStore):
        self.orders = orders
        self.cart = Cart()
        self.history: list[types.Content] = []
        self.traces: list[dict] = []
        self.last_order: dict | None = None

    # ---- Herramientas que Gemini puede invocar ----
    def run_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "buscar_platos":
                result = vector_store.search(
                    args["consulta"], k=5,
                    categoria=args.get("categoria"),
                    vegetariano=args.get("vegetariano"),
                    precio_max=args.get("precio_max"),
                )
                self.traces.append(build_trace(result))
                return {"resultados": [
                    {"id": r["plato"]["id"], "nombre": r["plato"]["nombre"], "categoria": r["plato"]["categoria"],
                     "precio": r["plato"]["precio"], "vegetariano": r["plato"]["vegetariano"],
                     "descripcion": r["plato"]["descripcion"], "similitud": round(1 - r["distancia_chroma"], 3)}
                    for r in result["resultados"]
                ]}
            if name == "agregar_al_carrito":
                self.cart.add(args["id"], int(args.get("cantidad") or 1), args.get("nota") or "")
                return {"ok": True, "carrito": self.cart.to_dict()}
            if name == "quitar_del_carrito":
                cantidad = args.get("cantidad")
                self.cart.remove(args["id"], int(cantidad) if cantidad else None)
                return {"ok": True, "carrito": self.cart.to_dict()}
            if name == "ver_carrito":
                return self.cart.to_dict()
            if name == "confirmar_pedido":
                if not self.cart.items:
                    return {"error": "El carrito está vacío; no hay nada que confirmar."}
                order = self.orders.create(args["nombre_cliente"], self.cart.to_dict())
                self.cart.clear()
                self.last_order = order
                return {"ok": True, "pedido_id": order["id"], "total": order["total"], "estado": order["estado"]}
            return {"error": f"Herramienta desconocida: {name}"}
        except CartError as e:
            return {"error": str(e)}

    # ---- Bucle de conversación ----
    def chat(self, message: str, client) -> str:
        start = len(self.history)
        try:
            return self._chat(message, client)
        except Exception:
            del self.history[start:]  # si Gemini falla, no dejar el historial a medias
            raise

    def _chat(self, message: str, client) -> str:
        self.traces = []
        self.last_order = None
        self.history.append(types.Content(role="user", parts=[types.Part(text=message)]))
        gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[TOOLS],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.4,
            # thinking_level solo existe en la familia Gemini 3
            thinking_config=(types.ThinkingConfig(thinking_level=config.GEMINI_THINKING)
                             if config.GEMINI_THINKING and config.GEMINI_MODEL.startswith("gemini-3") else None),
        )
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.models.generate_content(
                model=config.GEMINI_MODEL, contents=self.history, config=gen_config
            )
            self.history.append(response.candidates[0].content)
            calls = response.function_calls or []
            if not calls:
                return response.text or ""
            parts = [
                types.Part.from_function_response(name=call.name, response=self.run_tool(call.name, dict(call.args or {})))
                for call in calls
            ]
            self.history.append(types.Content(role="user", parts=parts))
        return "Perdón, me enredé un poco. ¿Me repites qué te gustaría pedir?"


def make_client():
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "tu_clave_aqui":
        raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")
    # function calling solo está disponible en la API v1beta; timeout en ms para no congelar la demo
    return genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1beta", timeout=45_000),
    )
