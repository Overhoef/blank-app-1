import streamlit as st
#import pandas as pd
#import matplotlib as plt
#import numpy as np
#import plotly.express as px
#import folium
st.set_page_config(layout="wide")

#homepage
st.title("🛫 Heb ik vertraging? 🛬")
st.write(
    "Eens kijken of je vertraging gaat hebben vandaag.."
)
st.subheader('Kaart van alle routes vanaf "vliegveld"')
#folium.Map(location=[41,29])


#Sidebar
st.sidebar.header('Info')
origin = st.sidebar.selectbox('vliegvelden', ['ICAO'])
#Tabs
airport, flights, delays, route = st.tabs(
    ["Vliegveld", "Vluchten", "Vertraging", "Route"]
)

#tab vliegveld
with airport:
    st.header('Vlieveld Info')

#tab vluchten
with flights:
    st.header(f'Vluchten vanaf {origin}')

#tab vertraging
with delays:
    st.header('Vertraging Info')

#tab route
with route:
    st.header('Route Info')
