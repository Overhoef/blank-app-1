import streamlit as st
import pandas as pd
import matplotlib as plt
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

zurich = pd.read_csv('schedule_airport.csv')
file_path = '/Users/olavverhoef/Desktop/Minor/Case3/blank-app-1/airports-extended-clean.csv'
airports = pd.read_csv(file_path, sep=';')

#Sidebar
st.sidebar.header('Info')
origin = st.sidebar.selectbox(f'airports', airports['ICAO'])
st.sidebar.subheader('')
st.sidebar.subheader('DATA NOTES')
#homepage
st.title("🛫 VLUCHTEN 🛬")


st.subheader('Kaart van alle routes van alle luchthavens')

airports_filtered = airports[airports['Type'] == 'airport']

airports_filtered['Latitude'] = airports_filtered['Latitude'].str.replace(',', '.', regex=False)
airports_filtered['Latitude'] = airports_filtered['Latitude'].astype(float)
airports_filtered['Longitude'] = airports_filtered['Longitude'].str.replace(',', '.', regex=False)
airports_filtered['Longitude'] = airports_filtered['Longitude'].astype(float)

airports_filtered['ICAO_First_Letter'] = airports_filtered['ICAO'].str[0]

# Count the occurrences of each first letter
icao_first_letter_counts = airports_filtered['ICAO_First_Letter'].value_counts()

# Get a list of Viridis colors (adjust the number as needed)
viridis_colors = px.colors.sequential.Viridis[:10]  # Example: Take the first 10 colors

# Create the colormap
colormap = {}
for letter, count in icao_first_letter_counts.items():
    if count > 3:
        # Calculate a color index within the available color list range
        color_index = count % len(viridis_colors)
        colormap[letter] = viridis_colors[color_index]
    else:
        colormap[letter] = 'black'  # Black for less frequent letters

# Create the scatter plot with color based on ICAO_First_Letter
fig = px.scatter_geo(
    airports_filtered,
    lat='Latitude',
    lon='Longitude',
    hover_name='Name',
    hover_data={'City': True, 'Country': True, 'IATA': True, 'ICAO': True},
    projection='natural earth',
    title='Wereldkaart van Luchthavens (Type: Airport)',
    color='ICAO_First_Letter',
    color_discrete_map=colormap,
)

fig.update_layout(
    height=800,
    margin={"r":0,"t":0,"l":0,"b":0},
    geo=dict(
        showland=True,
        landcolor="lightgray",
        coastlinecolor="black",
        projection_type='natural earth',
    )
)
st.plotly_chart(fig, use_container_width=True)

# #Tabs
# flights, delays, route = st.tabs(
#     "Vluchten", "Vertraging", "Route"
# )

# #tab vluchten
# with flights:
#     st.header(f'Vluchten vanaf {origin}')

# #tab vertraging
# with delays:
#     st.header('Vertraging Info')

# #tab route
# with route:
#     st.header('Route Info')

