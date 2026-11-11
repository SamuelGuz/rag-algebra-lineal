"""Carga del menú del restaurante desde data/menu.json."""
import json
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def load_menu() -> dict[str, dict]:
    with open(config.MENU_PATH, encoding="utf-8") as f:
        return {dish["id"]: dish for dish in json.load(f)}


def get_dish(dish_id: str) -> dict | None:
    return load_menu().get(dish_id.strip().upper())


def dish_to_document(dish: dict) -> str:
    """Texto que se convierte en vector: todo lo que describe al plato."""
    return (
        f"{dish['nombre']}. Categoría: {dish['categoria']}. {dish['descripcion']} "
        f"Ingredientes: {', '.join(dish['ingredientes'])}. "
        f"Características: {', '.join(dish['etiquetas'])}."
    )
