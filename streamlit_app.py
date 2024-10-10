import streamlit as st
import pandas as pd
import matplotlib as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
df = pd.read_csv('./schedule_airport.csv')
airports = pd.read_csv('./airports-extended-clean.csv', sep=';')

# homepage
st.title("🛫 VLUCHTEN 🛬")

st.subheader('Kaart van alle luchthavens')

airports_filtered = airports[airports['Type'] == 'airport']

airports_filtered['Latitude'] = airports_filtered['Latitude'].str.replace(',', '.', regex=False)
airports_filtered['Latitude'] = airports_filtered['Latitude'].astype(float)
airports_filtered['Longitude'] = airports_filtered['Longitude'].str.replace(',', '.', regex=False)
airports_filtered['Longitude'] = airports_filtered['Longitude'].astype(float)

airports_filtered['ICAO_First_Letter'] = airports_filtered['ICAO'].str[0]

# Count the occurrences of each first letter
icao_first_letter_counts = airports_filtered['ICAO_First_Letter'].value_counts()

# Get a list of Viridis colors (adjust the number as needed)
inferno_colors = px.colors.sequential.Inferno[:10]  # Example: Take the first 10 colors

# Create the colormap
colormap = {}
for letter, count in icao_first_letter_counts.items():
    if count > 3:
        # Calculate a color index within the available color list range
        color_index = count % len(inferno_colors)
        colormap[letter] = inferno_colors[color_index]
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
    color='ICAO_First_Letter',
    color_discrete_map=colormap,
)

fig.update_layout(
    height=800,
    margin={"r":0,"t":0,"l":0,"b":0},
    geo=dict(
        showland=True,
        landcolor="rgb(240, 240, 240)",
        coastlinecolor="black",
        projection_type='natural earth',
        bgcolor='darkgrey'
    )
)

st.plotly_chart(fig, use_container_width=True)

#Tabs
delays, data_notes = st.tabs(["vertraging", "notities"])

#tab vertraging
with delays:
    st.header('Vertraging Info')
    # Formatting 
    df["STA_STD_ltc"] = pd.to_datetime(df["STA_STD_ltc"], errors='coerce')
    df["ATA_ATD_ltc"] = pd.to_datetime(df["ATA_ATD_ltc"], errors='coerce')

    # Verwijderen van de vluchten met missende tijden
    df = df.dropna(subset=["ATA_ATD_ltc"])

    airports = df["Org/Des"].unique()

    # Selectbox voor de luchthavens
    selected_airport = st.selectbox("Selecteer een luchthaven om vertraging percentages te gaan vergelijken:", airports)

    # Filteren van de luchthavens
    filtered_df = df[df["Org/Des"] == selected_airport]

    # checkbox show code
    show_outbound = st.checkbox("Toon vertrekkende vluchten (S)", value=True)
    show_inbound = st.checkbox("Toon aankomende vluchten (L)", value=True)

    # Filters van outbound of inbound vlucht gemaakt met checkboxen
    if show_outbound and not show_inbound:
        filtered_df = filtered_df[filtered_df["LSV"] == "S"]
    elif show_inbound and not show_outbound:
        filtered_df = filtered_df[filtered_df["LSV"] == "L"]
    elif not show_outbound and not show_inbound:
        filtered_df = filtered_df.iloc[0:0]

    # Slider code tot 1 uur te wijzigen
    delay_minutes = st.slider("Marge van optijd aanpassen in minuten", min_value=0, max_value=60, value=0)

    # Slider code 
    filtered_df["Adjusted_STA_STD_ltc"] = filtered_df["STA_STD_ltc"] + pd.to_timedelta(delay_minutes, unit='m')

    # Originele status balk code
    filtered_df["Original_Status"] = filtered_df.apply(
        lambda row: "Op tijd" if row["ATA_ATD_ltc"] <= row["STA_STD_ltc"] else "Te laat", axis=1
    )

    # Verandere status balk code
    filtered_df["Adjusted_Status"] = filtered_df.apply(
        lambda row: "Op tijd" if row["ATA_ATD_ltc"] <= row["Adjusted_STA_STD_ltc"] else "Te laat", axis=1
    )

    # Rekensom van de gefilterde vluchten
    on_time_original = len(filtered_df[filtered_df["Original_Status"] == "Op tijd"])
    on_time_adjusted = len(filtered_df[filtered_df["Adjusted_Status"] == "Op tijd"])
    total_flights = len(filtered_df)

    # Foutmelding van 0 waarde delen opgelost
    if total_flights > 0:
        on_time_percentage_original = (on_time_original / total_flights) * 100
        on_time_percentage_adjusted = (on_time_adjusted / total_flights) * 100
    else:
        st.write("Geen vluchten gevonden voor de geselecteerde luchthaven.")
        on_time_percentage_original = None
        on_time_percentage_adjusted = None

    if on_time_percentage_original is not None and on_time_percentage_adjusted is not None:
        # Barchart code
        fig2 = px.bar(
            x=["Originele op-tijd aankomsten", "Aangepaste op-tijd aankomsten"],
            y=[on_time_percentage_original, on_time_percentage_adjusted],
            labels={"x": "Aankomst Status", "y": "Percentage"},
            title=f"Op tijd aankomende/vertrekkende vluchten voor {selected_airport} (Voor en Na Aanpassing)",
            template="plotly_dark",
            width=800
        )

        # Layout verbeteren
        fig2.update_layout(barmode='group')

        # Grafiek showcasen
        st.plotly_chart(fig2, use_container_width=True)


    # Gebruik de Streamlit layout optie om de dropdown en grafiek naast elkaar te plaatsen
col1, col2 = st.columns([3, 1])  # De eerste kolom is breder (voor de grafiek) en de tweede is smaller (voor de dropdown)

# Zet de dropdown in de tweede kolom (rechts van de grafiek)
with col2:
    year_selection = st.selectbox("Selecteer een jaar", [2019, 2020])

# Stap 1: Converteer STD naar een datetime-formaat en filter op het geselecteerde jaar
df['STD'] = pd.to_datetime(df['STD'], format='%d/%m/%Y', errors='coerce')
filtered_data = df[df['STD'].dt.year == year_selection]

# Stap 2: Bereken de vertraging (ATA_ATD_ltc > STA_STD_ltc)
filtered_data['STA_STD_ltc'] = pd.to_datetime(filtered_data['STA_STD_ltc'], format='%H:%M:%S', errors='coerce').dt.time
filtered_data['ATA_ATD_ltc'] = pd.to_datetime(filtered_data['ATA_ATD_ltc'], format='%H:%M:%S', errors='coerce').dt.time
filtered_data['Vertraagd'] = filtered_data['ATA_ATD_ltc'] > filtered_data['STA_STD_ltc']

# Stap 3: Groepeer per maand en tel de vertraagde vluchten
filtered_data['Maand'] = filtered_data['STD'].dt.strftime('%B')
vertraagde_per_maand = filtered_data[filtered_data['Vertraagd']].groupby('Maand').size().reindex(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], fill_value=0).reset_index()

# Stap 4: Maak de interactieve Plotly grafiek voor het geselecteerde jaar
fig = px.line(vertraagde_per_maand, x='Maand', y=0, markers=True, 
              title=f"Aantal Vertraagde Vluchten per Maand in {year_selection}")

# Pas de y-as dynamisch aan op basis van het geselecteerde jaar
if year_selection == 2019:
    y_range = [10000, 18000]  # Voor 2019, stel de y-as in van 10.000 tot 18.000
    tickvals = list(range(10000, 18001, 2000))  # Sprongen van 2.000 voor 2019
else:
    y_range = [0, 12000]      # Voor 2020, stel de y-as in van 0 tot 12.000
    tickvals = list(range(0, 12001, 2000))  # Sprongen van 2.000 voor 2020

# Stap 5: Pas de x-as en y-as aan, met dynamische y-as limieten en sprongen van 2.000
fig.update_layout(
    xaxis_title="Maanden",
    yaxis_title="Aantal Vertraagde Vluchten",
    xaxis_tickangle=45,
    hovermode="x unified",
    yaxis=dict(
        range=y_range,  # Dynamische y-as limieten op basis van het jaar
        tickvals=tickvals  # Sprongen van 2.000
    ),
    height=600,  # Verhoog de hoogte van de grafiek
    width=2400   # Maak de grafiek twee keer zo breed (2400 pixels)
)

# Toon de grafiek in de eerste kolom (links)
with col1:
    st.plotly_chart(fig)
    
# data notes tab
with data_notes:
    col1, col2 = st.columns(2)
    with col1:

        # Laad de dataset
        airports = pd.read_csv('./airports-extended-clean.csv', sep=';')

        # Filter de dataset op de 'Type' kolom (voor de luchthaventypes)
        luchthaven_types = airports['Type'].unique()

        # Creëer een multiselect widget zodat gebruikers luchthaventypes kunnen kiezen
        selected_types = st.multiselect(
            'Selecteer luchthaventypes om te tonen',
            options=luchthaven_types,
            default=luchthaven_types  # Standaard worden alle types getoond
        )

        # Filter de dataset op de geselecteerde types
        filtered_data = airports[airports['Type'].isin(selected_types)]

        # Groepeer op type en tel het aantal luchthavens per type
        type_counts = filtered_data['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Aantal']

        # Maak een taartdiagram met Plotly Express
        fig = px.pie(type_counts, values='Aantal', names='Type', title='Dataset oorspronkelijk')

        # Toon de grafiek in Streamlit
        st.plotly_chart(fig)
       
        # Optioneel: Toon de verdeling ook als bar chart
        if st.checkbox("Toon als staafdiagram"):
            bar_fig = px.bar(type_counts, x='Type', y='Aantal', title='Verdeling van Luchthaventypes (Staafdiagram)')
            st.plotly_chart(bar_fig)

    with col2:
        st.subheader('')
        # Filter alleen de luchthavens met Type 'airport'
        airports_filtered = airports[airports['Type'] == 'airport']

        # Groepeer op type en tel het aantal luchthavens per type
        type_counts = airports_filtered['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Aantal']

        # Maak een taartdiagram met Plotly Express
        fig = px.pie(type_counts, values='Aantal', names='Type', title='Gebruikte data')

        # Toon de grafiek in Streamlit
        st.plotly_chart(fig)

        # Optioneel: Toon de verdeling ook als bar chart met een unieke key voor de checkbox
        if st.checkbox("Toon als staafdiagram", key="bar_chart_checkbox"):
            bar_fig = px.bar(type_counts, x='Type', y='Aantal', title='Verdeling van Luchthaventypes (Staafdiagram)')
            st.plotly_chart(bar_fig)
