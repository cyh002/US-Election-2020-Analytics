import requests
import streamlit as st
import json

def get_geojson_state_names(geojson_url: str) -> list:
    """
    Fetches the GeoJSON file from the provided URL and extracts state names.

    Args:
        geojson_url (str): The URL to the GeoJSON file.

    Returns:
        list: A list of state names extracted from the GeoJSON.
    """
    try:
        response = requests.get(geojson_url)
        response.raise_for_status()  # Raise an error for bad status codes
        us_states_geojson = response.json()
        state_names = [feature['properties']['name'] for feature in us_states_geojson['features']]
        return state_names
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching GeoJSON data: {e}")
        return []


def compare_state_names(data_states: list, geojson_states: list) -> None:
    """
    Compares state names from the dataset with those from the GeoJSON and highlights mismatches.

    Args:
        data_states (list): List of state names from your dataset.
        geojson_states (list): List of state names from the GeoJSON.
    """
    # Normalize state names by stripping whitespace and converting to title case
    data_states_set = set(state.strip().title() for state in data_states)
    geojson_states_set = set(state.strip().title() for state in geojson_states)

    # States present in data but not in GeoJSON
    missing_in_geojson = data_states_set - geojson_states_set
    # States present in GeoJSON but not in data
    missing_in_data = geojson_states_set - data_states_set

    if missing_in_geojson:
        st.sidebar.warning("States in data not found in GeoJSON:")
        st.sidebar.write(sorted(list(missing_in_geojson)))
    else:
        st.sidebar.success("All data states are present in GeoJSON.")

    if missing_in_data:
        st.sidebar.info("States in GeoJSON not present in data:")
        st.sidebar.write(sorted(list(missing_in_data)))
    else:
        st.sidebar.info("All GeoJSON states are represented in data.")