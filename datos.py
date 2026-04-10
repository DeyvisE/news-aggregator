#url = "http://rpp.pe"
#print(f"https://rpp.pe/deportes?page={pagina}")
#print(f"El titulo de la web es: {titulo}")
#import json
#with open("mis_noticias.json", "w") as archivo:
    #json.dump(mis_noticias, archivo)
import pandas as pd
import requests
import joblib
modelo = joblib.load("modelo_noticias.pkl")
vectorizador = joblib.load("vectorizador.pkl")
from bs4 import BeautifulSoup
mis_noticias = []
diarios = [
        "https://rpp.pe/deportes?page=",
        "https://elcomercio.pe/deportes?page=",
        "https://libero.pe/deportes?page="
]
for pagina in range(1, 6):
    for base_url in diarios:
        url_completa = base_url + str(pagina)
        print(f"Viajando a: {url_completa}")
        respuesta = requests.get(url_completa)
        sopa = BeautifulSoup(respuesta.text, "html.parser")
        titulo = sopa.title.text
        if "rpp.pe" in base_url:
            titulares = sopa.find_all("h2")
        elif "elcomercio.pe" in base_url:
            titulares = sopa.find_all("h3")
        elif "libero.pe" in base_url:
            titulares = sopa.find_all("h2", class_="title")
        for titular in titulares:
            texto_limpio = titular.text.strip()
            enlace_tag = titular.find("a")
            if enlace_tag:
                link = enlace_tag["href"]
                matriz = vectorizador.transform([texto_limpio])
                prediccion = modelo.predict(matriz)[0]
                mis_noticias.append({"Titular": texto_limpio, "Link": link, "Categoria": prediccion})
                print(f"Extraído: {texto_limpio} | Clasificación: {prediccion}")
df_final = pd.DataFrame(mis_noticias)
df_final = df_final.drop_duplicates(subset="Link")
df_final.to_csv("noticias_clasificadas.csv", index=False)
  
  
