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
import pydeck as pdk


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

        st.image(image, caption='Porto by prettymaps',
                 use_column_width='auto')
# -----------------------------------------------LECTURA DE DATOS Y PREPROCESAMIENTO------------------------------------#

    df_cal = pd.read_csv('output/df_cal.csv.gz')
    porto_geojson = "http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/visualisations/neighbourhoods.geojson"
    porto_gdf = gpd.read_file(porto_geojson)
    # df_reviews = pd.read_csv('http://data.insideairbnb.com/portugal/norte/porto/2022-12-16/data/reviews.csv.gz', parse_dates=['date'])

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
    tabs = st.tabs(["Mapas", "Tipos de propiedades y alojamientos", "Consejos al turismo",
                   'Disponibilidad y precios para el 2023', 'Reseñas de los huéspedes'])

    # -------------------------------------------------------TAB 1-----------------------------------------------------#
    tab_plots = tabs[0]  # this is the first tab
    with tab_plots:

        st.title('Una representación visual de los datos sobre el terreno')

        cols = st.columns(2)
        with cols[0]:
            st.write("Mapa de las localizaciones de los alojamientos")
            # TODO hacer que el mapa se centre bien
            lats = df_slider['latitude'].tolist()
            lons = df_slider['longitude'].tolist()
            # Guardamos latitudes y longitudes, hacemos una tupla y las cambiamos a una lista.
            locations = list(zip(lats, lons))

            # Le das una lat y lon inicial y un zoom inicial para representar el mapa
            map1 = folium.Map(
                location=[41.1496, -8.6109], zoom_start=11, use_container_width=True)
            # Te añade las localizaciones al mapa generado anteriormente
            FastMarkerCluster(data=locations).add_to(map1)
            folium.Marker(location=[41.1496, -8.6109]).add_to(map1)
            st_folium(map1, returned_objects=[])

        with cols[1]:
            st.write("Mapa de la media de los precios por freguesias")
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
            map3 = folium.Map(
                location=[41.1496, -8.6109], zoom_start=11, use_container_width=True)

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
            color_scale.add_to(map3)
            st_folium(map3, returned_objects=[])

        cols = st.columns(2)
        with cols[0]:

            st.write('Mapa de calor de los precios del alojamiento:')
            # Mapa de calor basándome en uno de Demetrio
            # Get the minimum and maximum price values
            min_price = df_slider['price'].min()
            max_price = df_slider['price'].max()
            # Define the color scale for the legend
            color_scale = LinearColormap(
                ['green', 'yellow', 'red'], vmin=min_price, vmax=max_price, caption='Precio')
            # Create the map
            calorsita = folium.Map(
                location=[41.1496, -8.6109], tiles='cartodbpositron', zoom_start=15, use_container_width=True)
            # Add a heatmap to the base map
            HeatMap(data=df_slider[['latitude', 'longitude', 'price']],
                    radius=20,
                    gradient={0.2: 'green', 0.5: 'yellow', 1: 'red'},
                    min_opacity=0.2).add_to(calorsita)

            # Add the color scale legend
            # color_scale.add_to(calorsita)
            st_folium(calorsita, returned_objects=[])
        with cols[1]:
            # Display the map
            df_slider = df_slider.fillna(0)
            st.write(
                "Mapa de precios en 3D")
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=41.1496,
                    longitude=-8.6109,
                    zoom=11,
                    pitch=45,
                ),
                layers=[
                    pdk.Layer(
                        'HexagonLayer',
                        data=df_slider,
                        get_position='[longitude, latitude]',
                        radius=100,
                        elevation_scale=4,
                        elevation_range=[0, 1000],
                        pickable=True,
                        extruded=True,
                        get_fill_color='[255, (1 - (price / 300)) * 255, 0]',
                        get_line_color='[255, 255, 255]',
                    ),
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=df_slider,
                        get_position='[longitude, latitude]',
                        get_color='[200, 30, 0, 160]',
                        get_radius='price / 10',
                    ),
                ],
            ))

            # -------------------------------------------------------TAB 2-----------------------------------------------------#

    tab_plots = tabs[1]  # this is the second tab
    with tab_plots:
        st.title('Tipos de propiedades, alojamientos y número de huéspedes')

        cols = st.columns(2)
        with cols[0]:
            feq = df_slider['neighbourhood'].value_counts(
            ).sort_values(ascending=True)
            feq = feq[feq > 500]

            fig1 = px.bar(feq, x=feq.values, y=feq.index,
                          orientation='h', template='plotly_dark')
            fig1.update_layout(
                title="Freguesias con más de 500 alojamientos:",
                xaxis_title="Número",
                yaxis_title="Freguesia",
                font=dict(size=12)
            )
            st.plotly_chart(fig1,  use_container_width=True)

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
                                "rgb(255, 102, 102)", "rgb(102, 178, 255)", "rgb(102, 255, 178)", "rgb(12, 235, 738)"])
            proper.update_layout(title='Tipos de alojamientos en Oporto', xaxis_title='Número',
                                 yaxis_title='', legend_title='', font=dict(size=14), template='plotly_dark')
            st.plotly_chart(proper, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:

            freq = df_slider['room_type'].value_counts()

            room_by_type = px.bar(freq, barmode='stack', orientation='h', color=freq.index,
                                  labels={'y': 'Room Type', 'x': 'Number of Listings'}, template='plotly_dark', color_discrete_sequence=[
                                      "rgb(255, 102, 102)", "rgb(102, 178, 255)", "rgb(102, 255, 178)", "rgb(12, 235, 738)"])
            room_by_type.update_layout(title="Número de reservas por tipo de alojamiento",
                                       xaxis_title="Número",
                                       yaxis_title='')
            st.plotly_chart(room_by_type, use_container_width=True)

        with cols[1]:

            feq = df_slider['accommodates'].value_counts(
            ).sort_index().reset_index()
            feq.columns = ['Accommodates', 'Number of listings']
            accomm = px.bar(feq, x='Accommodates', y='Number of listings',
                            color='Accommodates',
                            width=700, height=600, template='plotly_dark')
            accomm.update_layout(title={'text': "Número de huéspedes", 'x': 0.5},
                                 xaxis_title='Huéspedes', yaxis_title='Número', font=dict(size=14))
            st.plotly_chart(accomm, use_container_width=True)

        # -------------------------------------------------------TAB 4-----------------------------------------------------#
    tab_plots = tabs[2]  # this is the third tab
    with tab_plots:
        st.title('Consejos al turismo')
        cols = st.columns(2)
        with cols[0]:
            # Carga de datos
            feq = df_slider[df_slider['accommodates'] == 2]
            feq = feq.groupby('neighbourhood')['price'].mean(
            ).sort_values(ascending=True).reset_index()

            # Crear gráfico
            fig = px.bar(feq, x='price', y='neighbourhood', orientation='h')
            fig.update_layout(
                title="Precio medio para dos huéspedes",
                xaxis_title="Precio (Euro)",
                yaxis_title="",
                font=dict(size=18)
            )
            st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            feq1 = df_slider[df_slider['number_of_reviews'] >= 10].groupby(
                'neighbourhood')['review_scores_location'].mean().sort_values(ascending=True)

            fig1 = px.bar(feq1, x='review_scores_location', y=feq1.index, orientation='h',
                          color='review_scores_location', color_continuous_scale='RdYlGn')
            fig1.update_layout(xaxis_title="Nota (1-5)",
                               yaxis_title="", title="Nota por localización",)

            st.plotly_chart(fig1, use_container_width=True)

        st.markdown('---')
        st.write('# Información sobre los host')
        # Muestra los gráficos en la interfaz
        figs = response_charts(df_slider)
        st.plotly_chart(figs, use_container_width=True)

        # calcular frecuencias
        df_frequencies = df_slider['host_is_superhost'].value_counts(
            normalize=True).reset_index()
        df_frequencies.columns = ['Superhost', 'Percentage']
        df_frequencies['Percentage'] = df_frequencies['Percentage'] * 100

        # crear gráfico de barras
        fig_super = px.bar(df_frequencies, x='Superhost', y='Percentage',
                           labels={'Superhost': 'Superhost',
                                   'Percentage': 'Percentage (%)'},
                           color='Superhost',
                           color_discrete_map={'f': 'rgb(255, 0, 0)', 't': 'rgb(0, 128, 0)'})

        # personalizar texto y leyenda
        fig_super.update_traces(
            texttemplate='%{y:.2f}%', textposition='inside')
        fig_super.update_layout(
            uniformtext_minsize=8, uniformtext_mode='hide')
        fig_super.update_layout(legend_title='Superhost', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # personalizar diseño
        fig_super.update_layout(
            title_text="Porcentaje de Superhost",
            height=400,
            width=1000,
            font_size=12,
            showlegend=False,
            template='plotly_dark'
        )
        # mostrar gráfico en Streamlit
        st.plotly_chart(fig_super, use_container_width=True)

        sh_v_h = px.scatter(df, x="Superhost", y="Price", color="Superhost",
                        color_discrete_map={"f": "red", "t": "green"},
                        labels={"Superhost": "Superhost", "Price": "Precio"}, template='plotly_dark')
        sh_v_h.update_layout(title="Precios de Superhost",
                        xaxis=dict(tickmode='linear'))
        st.plotly_chart(sh_v_h, use_container_width=True)

        # -------------------------------------------------------TAB 6-----------------------------------------------------#
    tab_plots = tabs[3]  # this is the third tab
    with tab_plots:

        st.title('Disponibilidad y precio para el 2023:')
        # Leer los datos y convertir la columna 'date' en tipo datetime
        # Leer los datos y convertir la columna 'date' en tipo datetime
        df_cal['date'] = pd.to_datetime(df_cal['date'])
        # Filtrar los datos para tener sólo los disponibles
        sum_available = df_cal[df_cal["available"] == "t"].groupby(
            ['date']).size().to_frame(name='available').reset_index()

        # Agregar la columna de día de la semana
        sum_available['weekday'] = sum_available['date'].dt.day_name()

        # Establecer 'date' como el índice del DataFrame
        sum_available = sum_available.set_index('date')

        # Crear la figura de Plotly Express
        disponibilidad = px.line(sum_available, y='available',
                                 title='Number of listings available by date', template='plotly_dark')
        st.plotly_chart(disponibilidad, use_container_width=True)
        # Gráfico del precio

        average_price = df_cal[(df_cal['available'] == "t") & (
            df_cal['accommodates'] == 2)].groupby(['date']).mean().astype(np.int64).reset_index()
        average_price['weekday'] = average_price['date'].dt.day_name()
        average_price = average_price.set_index('date')
        precio = px.line(average_price, x=average_price.index,
                         y='price_x', title='Precio medio para dos personas')
        precio.update_traces(text=average_price['weekday'])
        precio.update_layout(xaxis_title='Fecha',
                             yaxis_title='Precio', template='plotly_dark')
        st.plotly_chart(precio, use_container_width=True)

        # -------------------------------------------------------TAB 5-----------------------------------------------------#
    tab_plots = tabs[4]  # this is the third tab
    with tab_plots:
        st.title('Notas de los alojamientos')
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

        wordcloud = Image.open('img/wordcloud.png')

        st.image(wordcloud, caption='Nube de palabras hecha analizando las palabras más repetidas en los comentarios.',
                 use_column_width='auto')


if __name__ == '__main__':
    main()
