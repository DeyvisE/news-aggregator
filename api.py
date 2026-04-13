from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
app = FastAPI(title="Clasificador de Noticias")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # El asterisco significa que aceptamos órdenes de CUALQUIER página web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
modelo = joblib.load("modelo_noticias.pkl")
vectorizador = joblib.load("vectorizador.pkl")
@app.get("/")
def bienvenida():
    return {"mensaje": "¡Hola! La API del News Aggregator está funcionando."}
@app.get("/predecir")
def predecir_noticia(titular: str):
    texto_vectorizado = vectorizador.transform([titular])
    categoria_predicha = modelo.predict(texto_vectorizado)[0]
    return {"texto": titular, "resultado": categoria_predicha}
