#df.columns = ["Titular"
#df = pd.read_json("mis_noticias.json")
#df = pd.read_csv("titulares_deportes_peru.csv")
import joblib 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
def clasificar_noticia(titular):
    titular = titular.lower()
    if "ganó" in titular or "victoria" in titular or "triunfo" in titular or "remontó" in titular or "venció" in titular:
        return "Victoria"
    elif "perdió" in titular or "derrota" in titular or "caída" in titular or "remontado" in titular:
        return "Derrota"
    else:
        return "Neutral"

import pandas as pd
df = pd.read_json("mis_noticias.json")
noticias_u = df[df["Titular"].str.contains("Universitario")].copy()
datos_extra = pd.DataFrame({
    "Titular":[
        "Universitario ganó su partido",
        "Alianza lima perdió contra Universitario",
        "Universitario venció a Alianza lima",
        "Universitario le remontó a Alianza lima"
        ]
})
noticias_u = pd.concat([noticias_u, datos_extra], ignore_index=True)
noticias_u["resultado"] = noticias_u["Titular"].apply(clasificar_noticia)
noticias_neutrales = noticias_u[noticias_u["resultado"] == "Neutral"].sample(3)
noticias_relevantes = noticias_u[noticias_u["resultado"] != "Neutral"]
noticias_u = pd.concat([noticias_relevantes, noticias_neutrales], ignore_index=True)
noticias_u.to_csv("noticias_universitario.csv", index=False)
print(noticias_u)
print(df.head())
noticias = [
    "Universitario ganó el clásico",
    "Dura derrota de Universitario",
    "El clásico termino en empate",
]
vectorizador = TfidfVectorizer(stop_words=["vivo", "video", "vóley", "2026", "en", "vs", "la"])
matriz_tfidf = vectorizador.fit_transform(noticias_u["Titular"])
df_tfidf = pd.DataFrame(
    matriz_tfidf.toarray(),
    columns=vectorizador.get_feature_names_out()
)
print(df_tfidf.head())
X = df_tfidf
y = noticias_u["resultado"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = MultinomialNB()
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
calificacion = accuracy_score(y_test, predicciones)
resultados = confusion_matrix(y_test, predicciones)
print(f"La precisión del modelo es: {calificacion}")
print(f"Los resultados fueron: {resultados}")
print(f"El atributo classes_ es: {modelo.classes_}")
comparacion = pd.DataFrame({
    "Titular": noticias_u.loc[y_test.index,"Titular"],
    "Realidad": y_test,
    "Prediccion_IA": predicciones
})
print("\n--- REVISIÓN DEL EXAMEN ---")
print(comparacion)
nueva_noticia = ["Universitario venció en el Monumental"]
matriz_nueva = vectorizador.transform(nueva_noticia)
df_nueva = pd.DataFrame(
    matriz_nueva.toarray(),
    columns=vectorizador.get_feature_names_out()
)
print(f"La predicción es {modelo.predict(df_nueva)}")
joblib.dump(modelo, "modelo_noticias.pkl")
joblib.dump(vectorizador, "vectorizador.pkl")
