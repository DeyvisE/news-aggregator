# 📰 The News Aggregator (Scraping Estático)

## 📋 Descripción
Este proyecto es un agregador de noticias deportivas que extrae automáticamente los titulares de los principales diarios de Perú (RPP, El Comercio y Líbero). 

Se implementó un modelo de **Inteligencia Artificial** (Machine Learning) para automatizar la clasificación de cada noticia en distintas categorías. Esto permite filtrar rápidamente la información más relevante, ahorrando tiempo de lectura y sentando las bases para procesar grandes volúmenes de datos a mayor escala.

## ✨ Características principales
* **Multi-fuente:** Recolecta datos de 3 sitios web distintos simultáneamente.
* **Clasificación con IA:** Usa un modelo de *Machine Learning* (Naive Bayes) para etiquetar noticias en tiempo real.
* **Limpieza de datos:** Elimina duplicados automáticamente basándose en los enlaces.
* **Exportación:** Genera un archivo CSV limpio y listo para su análisis.

## 🛠️ Tecnologías utilizadas
* **Python** 🐍
* **Librerías:** pandas, requests, beautifulsoup4, joblib y scikit-learn.

## 🚀 Instalación
1. Clona este repositorio.
2. Instala las dependencias necesarias abriendo tu terminal y ejecutando:
   ```bash
   pip install -r requirements.txt
```