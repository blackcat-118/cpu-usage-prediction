import requests

def fetch_metrics(api_url: str) -> list:
    """
    Fetches the latest CPU usage metrics from the remote API.
    Assumes the remote API returns a JSON list of floats.
    """
    try:
        resp = requests.get(api_url, timeout=2)
        resp.raise_for_status()
        # Example: {"cpu_usage": [0.12, 0.13, ...]}
        data = resp.json()
        return data.get("cpu_usage", [])
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return []