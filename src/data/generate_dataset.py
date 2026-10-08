"""
Dataset Ingestion Entrypoint for Solar Flare Prediction.
Fetches real NOAA SWPC / NCEI GOES solar observation data.
"""

from src.data.noaa_fetcher import fetch_noaa_swpc_data

def generate_solar_dataset(output_path: str = "data/processed/solar_features.csv", **kwargs):
    """Fetch real NOAA SWPC GOES solar observation dataset."""
    return fetch_noaa_swpc_data(output_path=output_path)

if __name__ == "__main__":
    generate_solar_dataset()
