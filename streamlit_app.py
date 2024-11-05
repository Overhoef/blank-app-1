import streamlit as st
import pandas as pd
import matplotlib as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.sidebar.header('Sidebar')

df = pd.read_csv('schedule_airport.csv')
airports = pd.read_csv('airports-extended-clean.csv', sep=';')

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
delays, data_notes, prediction = st.tabs(["vertraging", "notities", "voorspelling"])

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

# data notes tab
with data_notes:
    col1, col2 = st.columns(2)
    with col1:
        # Laad de dataset
        airports = pd.read_csv('airports-extended-clean.csv', sep=';')

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

with prediction:
    # Load data
    data = pd.read_csv('schedule_airport.csv')

    # Convert time columns to datetime (specify format if needed) 
    data['ATA_ATD_ltc'] = pd.to_datetime(data['ATA_ATD_ltc'], format='mixed')
    data['STA_STD_ltc'] = pd.to_datetime(data['STA_STD_ltc'], format='mixed')

    # Feature engineering
    # Convert 'STA_STD_ltc' to datetime
    data['STA_STD_ltc'] = pd.to_datetime(data['STA_STD_ltc'], format='%Y-%m-%d %H:%M:%S')
    # Extract hour from STA_STD_ltc
    data['Minute'] = data['STA_STD_ltc'].dt.minute
    data['Hour'] = data['STA_STD_ltc'].dt.hour
    data['Delay'] = (data['ATA_ATD_ltc'] - data['STA_STD_ltc']).dt.total_seconds() / 60
    data['Airline'] = data['Identifier'].str.slice(0, 2)  # Extract airline code
    data['Origin'] = data['Org/Des'].str.split('/').str[0]
    data['Destination'] = data['Org/Des'].str.split('/').str[1]

    # Define delay threshold (minutes)
    delay_threshold = 1.6

    # Create binary target variable (Delayed: 1, Not Delayed: 0)
    data['Delayed'] = (data['ATA_ATD_ltc'] - data['STA_STD_ltc']).dt.total_seconds() / 60 > delay_threshold

    # Extract relevant features (consider adding more)
    features = ['Minute', 'Hour', 'Airline', 'Origin', 'Destination']
    X = data[features]
    y = data['Delayed']

    # Encode categorical features
    categorical_features = ['Minute', 'Hour', 'Airline', 'Origin', 'Destination']
    X_encoded = pd.get_dummies(X, columns=categorical_features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Scale numerical features (optional)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1-score:", f1)

    # ... (optional: plot visualizations)
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # Define the ROC curve trace
    trace = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',  # Set mode to 'lines' for a line plot
        name='ROC Curve (AUC = {}'.format(roc_auc),  # Include AUC in legend
        line=dict(color='darkorange', width=2)  # Set line color and width
    )
    # Define the reference line (diagonal)
    diagonal = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='navy', width=2, dash='dash')  # Set line color, width, and dash style
    )
    # Create the layout
    layout = go.Layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
    )
    # Create the figure
    fig = go.Figure(data=[trace, diagonal], layout=layout)

    st.plotly_chart(fig)
