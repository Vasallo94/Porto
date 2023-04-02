# --------------------LIBRERÍAS----------------------------#
import json
import os
import warnings
from webbrowser import get

import chart_studio.plotly as py
import cufflinks
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly_express as px
import requests
from branca.colormap import LinearColormap
from folium.plugins import (
    FastMarkerCluster, FloatImage, HeatMap, MarkerCluster)
from plotly.offline import init_notebook_mode, iplot
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
from utils.funciones import *
import json
from PIL import Image
import streamlit as st
from plotly.subplots import make_subplots

sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)

# mapas interactivos
# to make the plotly graphs
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
# text mining
# This i will use to print Lottie file ( Lottie is a JSON-based animation file format that can be used in webapp applications)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def main():

    # --------------------------------------------------------CONFIGURACIÓN DE LA PÁGINA---------------------------------------------------#
    # layout="centered" or "wide"
    st.set_page_config(page_title="Porto y Airbnb ",
                       layout="wide", page_icon="🐓", initial_sidebar_state="expanded")
    st.set_option("deprecation.showPyplotGlobalUse", False)

# -----------------------------------------------------------------HEADER----------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    # first column, this is the lottie file
    with col1:
        lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_7D0uqz.json"
        lottie_hello = load_lottieurl(lottie_url_hello)
        st_lottie(lottie_hello, key="hello", height=150, width=150, loop=True)
    # second column, this is the title
    with col2:
        st.title(
            "Airbnb: Oporto")
    with col3:
        image = Image.open('img/porto.jpeg')

        st.image(image, caption='Porto by prettymaps')
# -----------------------------------------------LECTURA DE DATOS Y PREPROCESAMIENTO------------------------------------#

    df_calendar = pd.read_csv('http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/data/calendar.csv.gz',
                              parse_dates=['date'], index_col=['listing_id'])
    porto_geojson = "http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/visualisations/neighbourhoods.geojson"
    porto_gdf = gpd.read_file(porto_geojson)
    df_reviews = pd.read_csv(
        'http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/data/reviews.csv.gz', parse_dates=['date'])

    df_55 = pd.read_csv('output/df_55.csv')
    st.subheader(
        "Análisis de la situación del alquiler de alojamientos en Airbnb en la ciudad de Oporto y sus alrededores.")
# -----------------------------------------------------------SLIDER--------------------------------------------#
    # Filtrar el dataframe según la distancia seleccionada
    # Crear el slider para seleccionar la distancia máxima
    distancia = st.slider(
        "Seleccione la distancia máxima a la que mostrar resultados (km):", 1, 60, 25)

    # Crear un nuevo dataframe filtrando los valores de la columna de distancias
    df_slider = df_55[df_55['distancia'] < distancia]


# -----------------------------------------------------------MAIN PAGE----------------------------------------#
    # Show the selected dataframe on the main page
    st.write(df_slider)
    st.markdown("""---""")
    st.markdown(
        "<center><h2><l style='color:white; font-size: 30px;'>Visualización y estudio de los datos</h2></l></center>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------TABS---------------------------------------------------#
    st.title("Selecciona lo que te interese")
    tabs = st.tabs(["Agrupaciones parroquiales: freguesias", "Tipos de propiedades y alojamientos",
                   "Número de alojados", "Consejos al turismo", 'Reseñas de los huéspedes', 'Disponibilidad y precios para el 2023'])

    # -------------------------------------------------------TAB 1-----------------------------------------------------#
    tab_plots = tabs[0]  # this is the first tab
    with tab_plots:

        st.title('Lorem ipsum dolor sit amet...')
        st.subheader(
            'Lorem ipsum dolor sit amet...')

        st.write('Lorem ipsum dolor sit amet...')

        cols = st.columns(2)
        with cols[0]:
            st.write("Freguesias con más de 500 alojamientos:")
            feq = df_slider['neighbourhood'].value_counts(
            ).sort_values(ascending=True)
            feq = feq[feq > 500]

            fig1 = px.bar(feq, x=feq.values, y=feq.index,
                          orientation='h', template='plotly_dark')
            fig1.update_layout(
                title="Number of listings by neighbourhood",
                xaxis_title="Number of listings",
                yaxis_title="Neighbourhood",
                font=dict(size=12)
            )
            st.plotly_chart(fig1,  use_container_width=True)

        with cols[1]:
            # TODO hacer que el mapa se centre bien
            lats = df_slider['latitude'].tolist()
            lons = df_slider['longitude'].tolist()
            # Guardamos latitudes y longitudes, hacemos una tupla y las cambiamos a una lista.
            locations = list(zip(lats, lons))

            # Le das una lat y lon inicial y un zoom inicial para representar el mapa
            map1 = folium.Map(location=[41.1496, -8.6109], zoom_start=15)
            # Te añade las localizaciones al mapa generado anteriormente
            FastMarkerCluster(data=locations).add_to(map1)
            folium.Marker(location=[41.1496, -8.6109]).add_to(map1)
            st_folium(map1, width=2000, height=600, returned_objects=[])

        cols = st.columns(2)
        with cols[0]:
            st.write("PRUEBA")
            # ? Poner cuando esté mejorado el código
            # Calculate mean price by neighborhood for listings that accommodate 2 people
            mean_prices = df_slider.loc[df_slider['accommodates'] == 2].groupby(
                'neighbourhood')['price'].mean()

            # Join the mean prices to the geojson
            porto_gdf = porto_gdf.join(mean_prices, on='neighbourhood')

            # Drop neighborhoods without mean prices
            porto_gdf.dropna(subset=['price'], inplace=True)

            # Round the mean prices and create a dictionary for the color map
            price_dict = porto_gdf.set_index('neighbourhood')[
                'price'].round().to_dict()

            # Define color map
            color_scale = LinearColormap(['green', 'yellow', 'red'], vmin=min(
                price_dict.values()), vmax=max(price_dict.values()), caption='Average price')

            # Define style and highlight functions
            def style_function(feature):
                return {
                    'fillColor': color_scale(price_dict.get(feature['properties']['neighbourhood'], 0)),
                    'color': 'black',
                    'weight': 1,
                    'dashArray': '5, 5',
                    'fillOpacity': 0.5
                }

            def highlight_function(feature):
                return {
                    'weight': 3,
                    'fillColor': color_scale(price_dict.get(feature['properties']['neighbourhood'], 0)),
                    'fillOpacity': 0.8
                }

            # Create map
            map3 = folium.Map(location=[41.1496, -8.6109], zoom_start=15)

            # Add geojson layer to map with tooltip and style and highlight functions
            folium.GeoJson(
                data=porto_gdf,
                name='Oporto',
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['neighbourhood', 'price'], labels=True, sticky=False),
                style_function=style_function,
                highlight_function=highlight_function
            ).add_to(map3)

            # Add marker to map
            folium.Marker(location=[41.1496, -8.6109]).add_to(map3)

            # Add color scale to map
            map3.add_child(color_scale)
            st_folium(map3,  width=2000, height=600, returned_objects=[])

        with cols[1]:
            # TODO hacer que el mapa se centre bien
            # Mapa de calor basándome en uno de Demetrio
            # Get the minimum and maximum price values
            min_price = df_slider['price'].min()
            max_price = df_slider['price'].max()
            # Define the color scale for the legend
            color_scale = LinearColormap(
                ['green', 'yellow', 'red'], vmin=min_price, vmax=max_price, caption='Precio')
            # Create the map
            calorsita = folium.Map(
                location=[41.1496, -8.6109], tiles='cartodbpositron', zoom_start=15)

            # Add a heatmap to the base map
            HeatMap(data=df_slider[['latitude', 'longitude', 'price']],
                    radius=20,
                    gradient={0.2: 'green', 0.5: 'yellow', 1: 'red'},
                    min_opacity=0.2).add_to(calorsita)

            # Add the color scale legend
            calorsita.add_child(color_scale)

            # Display the map
            st_folium(calorsita, width=2000, height=600, returned_objects=[])


# -------------------------------------------------------TAB 2-----------------------------------------------------#

    tab_plots = tabs[1]  # this is the second tab
    with tab_plots:

        st.title('Lorem ipsum dolor sit amet...')
        st.subheader(
            'Lorem ipsum dolor sit amet...')

        st.write('Lorem ipsum dolor sit amet...')

        cols = st.columns(2)
        with cols[0]:
            freq = df_slider['room_type'].value_counts(
            ).sort_values(ascending=True)

            room_by_type = px.bar(freq, orientation='h', color=freq.index,
                                  labels={'y': 'Room Type', 'x': 'Number of Listings'}, template='plotly_dark')
            room_by_type.update_layout(title="Number of Listings by Room Type",
                                       xaxis_title="Number of Listings",
                                       yaxis_title="Room Type",
                                       height=400, width=800)
            st.plotly_chart(room_by_type, use_container_width=True)

        with cols[1]:
            prop = df_slider.groupby(
                ['property_type', 'room_type']).room_type.count()
            prop = prop.unstack()
            prop['total'] = prop.iloc[:, 0:3].sum(axis=1)
            prop = prop.sort_values(by=['total'])
            prop = prop[prop['total'] >= 100]
            prop = prop.drop(columns=['total'])

            proper = px.bar(prop, barmode='stack', orientation='h',
                            color_discrete_sequence=[
                                "rgb(255, 102, 102)", "rgb(102, 178, 255)", "rgb(102, 255, 178)"],
                            width=1000, height=600)
            proper.update_layout(title='Property types in Oporto', xaxis_title='Number of listings', yaxis_title='',
                                 legend_title='', font=dict(size=14), template='plotly_dark')
            st.plotly_chart(proper, use_container_width=True)
        cols = st.columns(2)
        with cols[0]:
            feq = df_slider['accommodates'].value_counts(
            ).sort_index().reset_index()
            feq.columns = ['Accommodates', 'Number of listings']
            accomm = px.bar(feq, x='Accommodates', y='Number of listings',
                            color='Accommodates',
                            width=700, height=600, template='plotly_dark')
            accomm.update_layout(title={'text': "Accommodates (number of people)", 'x': 0.5},
                                 xaxis_title='Accommodates', yaxis_title='Number of listings',
                                 font=dict(size=14))
            st.plotly_chart(accomm, use_container_width=True)

        with cols[1]:
            st.write(
                "Aquí se podría poner el código que muestra los tipos de habitaciones pero no funciona aún.")
            # -------------------------------------------------------TAB 3-----------------------------------------------------#
    tab_plots = tabs[2]  # this is the third tab
    with tab_plots:

        st.title('Lorem ipsum dolor sit amet...')
        st.subheader('Lorem ipsum dolor sit amet...')

        st.write('Lorem ipsum dolor sit amet...')

        cols = st.columns(2)
        with cols[0]:
            st.write('Lorem ipsum dolor sit amet...')

        with cols[1]:
            st.write('Lorem ipsum dolor sit amet...')

            # -------------------------------------------------------TAB 4-----------------------------------------------------#
    tab_plots = tabs[3]  # this is the third tab
    with tab_plots:

        st.title('Lorem ipsum dolor sit amet...')
        st.subheader('Lorem ipsum dolor sit amet...')

        st.write('Lorem ipsum dolor sit amet...')

        cols = st.columns(2)
        with cols[0]:
            st.write('Lorem ipsum dolor sit amet...')

        with cols[1]:
            st.write('Lorem ipsum dolor sit amet...')

        # -------------------------------------------------------TAB 5-----------------------------------------------------#
    tab_plots = tabs[4]  # this is the third tab
    with tab_plots:
        st.title('Disponibilidad y precios de los alojamientos para el 2023')
        st.subheader('Lorem ipsum dolor sit amet...')

        st.write('Lorem ipsum dolor sit amet...')

        # Select listings with at least 10 reviews
        listings10 = df_slider[df_slider['number_of_reviews'] >= 10]

        # Create histogram figures for each review category
        histograms = []
        categories = ['location', 'cleanliness', 'value',
                      'communication', 'checkin', 'accuracy']
        for category in categories:
            scores_col = listings10[f'review_scores_{category}']
            if scores_col.dtype != 'float64':
                scores_col = pd.to_numeric(
                    scores_col, errors='coerce').fillna(0)
            scores = go.Figure(go.Histogram(
                x=scores_col, nbinsx=800, histfunc="count"))
            scores.update_layout(title=category.capitalize(),
                                 xaxis_title="Average review score",
                                 yaxis_title="Percentage of listings",
                                 font_size=14)
            histograms.append(scores)

        # Create subplot with 2 rows and 3 columns, with titles for each subplot
        scores = make_subplots(rows=2, cols=3, subplot_titles=(
            "Location", "Cleanliness", "Value", "Communication", "Arrival", "Accuracy"))

        # Add each bar chart to the subplot
        for i, fig in enumerate(histograms):
            row = i // 3 + 1
            col = i % 3 + 1
            scores.add_trace(fig['data'][0], row=row, col=col)

        # Update layout for the subplot
        scores.update_layout(
            title_text="Review Scores",
            height=800,
            width=1000,
            font_size=12,
            showlegend=False,
            template='plotly_dark'
        )

        # Show the plot
        st.plotly_chart(scores, use_container_width=True)
        # -------------------------------------------------------TAB 6-----------------------------------------------------#
    tab_plots = tabs[5]  # this is the third tab
    with tab_plots:

        st.title('Lorem ipsum dolor sit amet...')
        st.subheader('Lorem ipsum dolor sit amet...')

        st.write('Lorem ipsum dolor sit amet...')

        cols = st.columns(2)
        with cols[0]:
            st.write('Lorem ipsum dolor sit amet...')

        with cols[1]:
            st.write('Lorem ipsum dolor sit amet...')


if __name__ == '__main__':
    main()
