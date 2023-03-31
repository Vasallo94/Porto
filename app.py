# --------------------LIBRERÍAS----------------------------#
import plotly.io as pio
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import cufflinks
from plotly.offline import iplot, init_notebook_mode
import plotly.subplots as sp
import chart_studio.plotly as py
import plotly_express as px
import plotly.graph_objs as go
from folium.plugins import HeatMap, MarkerCluster
from branca.colormap import LinearColormap
import geopandas as gpd
from folium.plugins import FastMarkerCluster, FloatImage
import folium
from janitor import clean_names
import warnings
import json
import os
from utils.funciones import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)

# mapas interactivos
# to make the plotly graphs
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
# text mining


def main():
    # --------------------CONFIGURACIÓN DE LA PÁGINA----------------------------#
    # layout="centered" or "wide"
    st.set_page_config(page_title="Porto y Airbnb ",
                       layout="wide", page_icon="🚢")
    st.set_option("deprecation.showPyplotGlobalUse", False)

    # --------------------LOGO+CREACIÓN DE COLUMNA----------------------------#
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.title("Titanic: Análisis de datos")
    # with col2:
    #     st.subheader("")
    # with col3:
    #     st.markdown(
    #         "![Titanic Leonardo Dicaprio GIF](https://media.giphy.com/media/ZiHgApcM5ij1S/giphy.gif)")

    # ----------------------LECTURA DE DATOS Y PREPROCESAMIENTO------------------#
    # Leemos los csv con los datos del Oporto y creamos los dataframe
    df_calendar = pd.read_csv('http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/data/calendar.csv.gz',
                              parse_dates=['date'], index_col=['listing_id'])
    # df_listing_detailed = pd.read_csv('http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/data/listings.csv.gz', index_col = ["id"])
    df_reviews = pd.read_csv(
        'http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/data/reviews.csv.gz', parse_dates=['date'])
    # df_listing = pd.read_csv('http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/visualisations/listings.csv', index_col=["id"])
    # df_listing_detailed['price'] = df_listing_detailed['price'].replace('[$,]', '', regex=True)
    # df_listing_detailed['price'] = df_listing_detailed['price'].astype(float).round().astype(int)

    # ? PREPROCESAMIENTO NO CREO QUE EN LA APP SEA NECESARIO PONER TODOS LOS DFs Y EL PREPROCESAMIENTO SINO SIMPLEMENTE YA USAMOS LOS CSV DE OUTPUT

    oporto = pd.read_csv('output/oporto.csv')

    # --------------------TITLE----------------------------#
    st.title(
        "Análisis de la situación de Airbnb en la ciudad de Oporto y sus alrededores")

    # ----------------------------SLIDEBAR---------------------------------#
    # st.sidebar.markdown("![BORRAR O PONER OTRA COSA](https://media.giphy.com/media/ZiHgApcM5ij1S/giphy.gif)")
    # Create the slidebar to select the distance to filter
    distancia_km = st.slider("Selecciona la distancia en km:", 0.0, 55.0)

    # Filtrar el dataframe según la distancia seleccionada
    df_distancia = filtrar_por_distancia(oporto, distancia_km)

    # Mostrar el dataframe filtrado

    # --------------------------------MAIN PAGE---------------------------#
    # Show the selected dataframe on the main page
    st.write(df_distancia)
    st.markdown("""---""")
    st.markdown(
        "<center><h2><l style='color:white; font-size: 30px;'>Visualización y estudio de los datos</h2></l></center>",
        unsafe_allow_html=True,
    )

    st.title("Selección de temas a estudiar")

    menu = ["Agrupaciones parroquiales: freguesias",
            "Tipos de propiedades y alojamientos", "Número de alojados", "Consejos al turismo", "Más!!!"]
    choice = st.sidebar.selectbox("Seleccione una pestaña", menu)

    if choice == "Agrupaciones parroquiales: freguesias":
        # Contenido de la Correlaciones entre los datos
        st.subheader("Distribuciones de las viviendas y precios")
        st.write(
            "RELLENAR")
        # Matriz de comparación de los datos
        # Creamos un diccionario de colores para asignar a cada valor de la columna "SURVIVED"

    elif choice == "Tipos de propiedades y alojamientos":
        # Tipos de propiedades y alojamientos
        st.subheader("RELLENAR")
        st.write(
            "RELLENAR")

    elif choice == "Número de alojados":
        # Contenido de la Sobre el precio de los pasajes
        st.subheader(
            "RELLENAR")
        st.write(
            "RELLENAR")

    elif choice == "Consejos al turismo":
        # Contenido de la Ruta y ciudades de embarque
        st.subheader("RELLENAR")
        st.write(
            "RELLENAR")

    elif choice == "Más!!!":
        # Contenido de la Sobre el precio de los pasajes a los pasajeros
        st.subheader("Más!!!")
        # st.write(
        #     "Aquí se puede agregar el contenido de la Sobre el precio de los pasajes a los pasajeros.")
        # Títulos y nubes de palabras
        st.image('img/wordcloud.png', use_column_width=True)


if __name__ == '__main__':
    main()
