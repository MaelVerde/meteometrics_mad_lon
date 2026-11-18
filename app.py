import pandas as pd
import requests

url_mad = "https://meteometrics.rlopez.org/historical-data/MAD"
url_lon = "https://meteometrics.rlopez.org/historical-data/LON"

try:
    response = requests.get(url_mad)
    response.raise_for_status()
    raw_data_madrid = response.json()

    response = requests.get(url_lon)
    response.raise_for_status()
    raw_data_londres = response.json()

    df_madrid = pd.json_normalize(raw_data_madrid)
    df_madrid["city"] = "Madrid"

    df_londres = pd.json_normalize(raw_data_londres)
    df_londres["city"] = "Londres"

    df = pd.concat([df_madrid, df_londres], ignore_index=True)
    print(df)

except requests.exceptions.RequestException as e:
    print(f"Error en la petición: {e}")