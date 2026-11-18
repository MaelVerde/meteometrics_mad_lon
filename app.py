from pathlib import Path

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

    registros_iniciales = len(df)
    columnas_unidad = [
        "metrics.temperature.unit",
        "metrics.precipitation.unit",
    ]

    # Normaliza el separador decimal solo en las métricas.
    columnas_metricas = df.columns[
        df.columns.str.startswith("metrics.")
        & ~df.columns.isin(columnas_unidad)
    ]
    df[columnas_metricas] = df[columnas_metricas].replace(",", ".", regex=True)

    # Convierte todas las columnas numéricas, excepto ciudad y unidades.
    columnas_numericas = df.columns[
        ~df.columns.isin(["city", *columnas_unidad])
    ]
    df[columnas_numericas] = df[columnas_numericas].apply(
        pd.to_numeric, errors="coerce"
    )

    # Trata el valor centinela como dato ausente.
    df[columnas_numericas] = df[columnas_numericas].mask(
        df[columnas_numericas].eq(-999.0)
    )

    # Elimina registros repetidos y ordena cada ciudad cronológicamente.
    df = df.drop_duplicates(subset=["city", "year", "month"])
    df = df.sort_values(["city", "year", "month"]).reset_index(drop=True)

    # Convierte las temperaturas Fahrenheit a Celsius de forma vectorizada.
    temperatura_en_fahrenheit = df["metrics.temperature.unit"].eq("F")
    columnas_temperatura = [
        "metrics.temperature.max",
        "metrics.temperature.avg",
        "metrics.temperature.min",
    ]
    df.loc[temperatura_en_fahrenheit, columnas_temperatura] = (
        df.loc[temperatura_en_fahrenheit, columnas_temperatura] - 32
    ) * 5 / 9
    df.loc[temperatura_en_fahrenheit, "metrics.temperature.unit"] = "C"

    # Convierte la precipitación en pulgadas a milímetros.
    precipitacion_en_pulgadas = df["metrics.precipitation.unit"].eq("inches")
    df.loc[precipitacion_en_pulgadas, "metrics.precipitation.avg"] *= 25.4
    df.loc[precipitacion_en_pulgadas, "metrics.precipitation.unit"] = "mm"

    registros_finales = len(df)
    informe = (
        f"Registros iniciales: {registros_iniciales}\n"
        f"Registros borrados: {registros_iniciales - registros_finales}\n"
        f"Registros finales: {registros_finales}\n"
    )
    Path(__file__).with_name("informe_limpieza.txt").write_text(
        informe, encoding="utf-8"
    )

    print(df)

except requests.exceptions.RequestException as e:
    print(f"Error en la petición: {e}")