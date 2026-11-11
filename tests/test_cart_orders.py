import pytest

from app.cart import Cart, CartError
from app.orders import OrderStore


def test_carrito_agregar_quitar_total():
    c = Cart()
    c.add("S01", 2)
    c.add("b02")
    assert c.total == 2 * 32000 + 9000
    c.add("S01", 1, "sin alcaparras")
    assert c.items["S01"]["cantidad"] == 3 and c.items["S01"]["nota"] == "sin alcaparras"
    c.remove("S01", 1)
    assert c.items["S01"]["cantidad"] == 2
    c.remove("S01")
    assert "S01" not in c.items and c.total == 9000


def test_carrito_rechaza_id_inexistente():
    with pytest.raises(CartError):
        Cart().add("ZZZ")


def test_pedidos_sqlite(tmp_path):
    store = OrderStore(tmp_path / "p.sqlite3")
    c = Cart()
    c.add("P01")
    order = store.create("Ana", c.to_dict())
    assert order["estado"] == "recibido" and order["total"] == 42000
    assert order["items"][0]["nombre"] == "Bandeja paisa"
    assert store.set_status(order["id"], "listo")["estado"] == "listo"
    assert len(OrderStore(tmp_path / "p.sqlite3").list()) == 1  # persiste
    with pytest.raises(ValueError):
        store.set_status(order["id"], "volando")
