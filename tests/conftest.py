import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import vector_store  # noqa: E402


@pytest.fixture(scope="session")
def col(tmp_path_factory):
    """Colección de Chroma temporal con el menú indexado."""
    c = vector_store.get_collection(tmp_path_factory.mktemp("chroma"))
    vector_store.index_menu(c)
    return c
