"""Carrito de compras por sesión (en memoria)."""
from app.menu import get_dish


class CartError(ValueError):
    pass


class Cart:
    def __init__(self):
        self.items: dict[str, dict] = {}  # id -> {id, nombre, precio, cantidad, nota}

    def add(self, dish_id: str, cantidad: int = 1, nota: str = "") -> dict:
        dish = get_dish(dish_id)
        if dish is None:
            raise CartError(f"El plato con id '{dish_id}' no existe en el menú.")
        if cantidad < 1:
            raise CartError("La cantidad debe ser al menos 1.")
        item = self.items.setdefault(
            dish["id"], {"id": dish["id"], "nombre": dish["nombre"], "precio": dish["precio"], "cantidad": 0, "nota": ""}
        )
        item["cantidad"] += cantidad
        if nota:
            item["nota"] = nota
        return item

    def remove(self, dish_id: str, cantidad: int | None = None) -> None:
        key = dish_id.strip().upper()
        if key not in self.items:
            raise CartError(f"El plato '{dish_id}' no está en el carrito.")
        if cantidad is None or cantidad >= self.items[key]["cantidad"]:
            del self.items[key]
        else:
            self.items[key]["cantidad"] -= cantidad

    def clear(self) -> None:
        self.items.clear()

    @property
    def total(self) -> int:
        return sum(i["precio"] * i["cantidad"] for i in self.items.values())

    def to_dict(self) -> dict:
        items = [dict(i, subtotal=i["precio"] * i["cantidad"]) for i in self.items.values()]
        return {"items": items, "total": self.total}
