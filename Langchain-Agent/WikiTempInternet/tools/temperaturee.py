
from dotenv import load_dotenv
load_dotenv()

import requests, datetime
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class TemperatureInput(BaseModel):
    """Fetch current temperature for give place"""
    query: str = Field(description="Fetch current temperature for given place")

class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


@tool
def temperature_tool(query:str):
    """Fetch current temperature for give place"""
    validated_input = TemperatureInput(query=query)
    weather = OpenWeatherMapAPIWrapper()
    return weather.run(validated_input.query)


@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {'latitude': latitude,'longitude': longitude,'hourly': 'temperature_2m','forecast_days': 1}
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in
                 results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}°C'
