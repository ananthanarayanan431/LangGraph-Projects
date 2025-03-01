
from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class TemperatureInput(BaseModel):
    """Fetch current temperature for give place"""
    query: str = Field(description="Fetch current temperature for given place")


def temperature_tool(query:str):
    """Fetch current temperature for give place"""
    validated_input = TemperatureInput(query=query)
    weather = OpenWeatherMapAPIWrapper()
    return weather.run(validated_input.query)

