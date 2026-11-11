"""Arranca la aplicación: python run.py  ->  http://localhost:8000"""
import uvicorn

if __name__ == "__main__":
    print("Chat:   http://localhost:8000")
    print("Cocina: http://localhost:8000/cocina")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
