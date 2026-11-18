from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).resolve().parent / "data"
CITY_COLORS = {"Madrid": "#d62728", "Londres": "#1f77b4"}
MONTHS = [
    "Ene",
    "Feb",
    "Mar",
    "Abr",
    "May",
    "Jun",
    "Jul",
    "Ago",
    "Sep",
    "Oct",
    "Nov",
    "Dic",
]


def load_dashboard_data():
    paths = {
        "df_total": DATA_DIR / "df_total.parquet",
        "resumen_ciudad": DATA_DIR / "resumen_ciudad.parquet",
        "resumen_anual": DATA_DIR / "resumen_anual.parquet",
    }
    missing_files = [path.name for path in paths.values() if not path.is_file()]
    if missing_files:
        st.error(
            "Faltan archivos Parquet en la carpeta data: "
            f"{', '.join(missing_files)}. Ejecuta primero app.py."
        )
        st.stop()

    try:
        return {
            name: pd.read_parquet(path, engine="pyarrow")
            for name, path in paths.items()
        }
    except ImportError as error:
        st.error(f"No se pudo cargar PyArrow para leer los Parquet: {error}")
        st.stop()


def add_month_date(df_total):
    data = df_total.copy()
    date_text = (
        data["year"].astype("Int64").astype("string")
        + "-"
        + data["month"].astype("Int64").astype("string")
        + "-01"
    )
    data["date"] = pd.to_datetime(date_text, errors="coerce")
    return data.dropna(subset=["date"])


def create_temperature_figure(df_total):
    required = [
        "city",
        "date",
        "metrics.temperature.max",
        "metrics.temperature.avg",
        "metrics.temperature.min",
    ]
    plot_data = df_total.dropna(subset=required).sort_values("date")
    if plot_data.empty:
        return None

    figure = go.Figure()
    for city, color in CITY_COLORS.items():
        city_data = plot_data[plot_data["city"].eq(city)]
        if city_data.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=city_data["date"],
                y=city_data["metrics.temperature.min"],
                mode="lines",
                line=dict(color=color, width=0.5),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=city,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=city_data["date"],
                y=city_data["metrics.temperature.max"],
                mode="lines",
                line=dict(color=color, width=0.5),
                fill="tonexty",
                fillcolor=(
                    "rgba(214,39,40,0.16)"
                    if city == "Madrid"
                    else "rgba(31,119,180,0.16)"
                ),
                name=f"{city}: rango máx.-mín.",
                legendgroup=city,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=city_data["date"],
                y=city_data["metrics.temperature.avg"],
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{city}: media",
                legendgroup=city,
            )
        )

    figure.update_layout(
        title="Evolución mensual de la temperatura",
        xaxis_title="Fecha",
        yaxis_title="Temperatura (°C)",
        hovermode="x unified",
        legend_title="Ciudad y medida",
    )
    return figure


def create_climate_radar(df_total):
    monthly = (
        df_total.groupby(["city", "month"], as_index=False)
        .agg(
            temperatura_media=("metrics.temperature.avg", "mean"),
            precipitacion_media=("metrics.precipitation.avg", "mean"),
        )
    )
    monthly = monthly[monthly["month"].between(1, 12)]
    if monthly.empty:
        return None, monthly

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=[
            "Temperatura media (°C)",
            "Precipitación media (mm)",
        ],
    )

    for city, color in CITY_COLORS.items():
        profile = (
            monthly[monthly["city"].eq(city)]
            .set_index("month")
            .reindex(range(1, 13))
        )
        figure.add_trace(
            go.Scatterpolar(
                r=profile["temperatura_media"],
                theta=MONTHS,
                mode="lines+markers",
                fill="toself",
                fillcolor=(
                    "rgba(214,39,40,0.12)"
                    if city == "Madrid"
                    else "rgba(31,119,180,0.12)"
                ),
                line=dict(color=color),
                name=city,
                legendgroup=city,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatterpolar(
                r=profile["precipitacion_media"],
                theta=MONTHS,
                mode="lines+markers",
                fill="toself",
                fillcolor=(
                    "rgba(214,39,40,0.12)"
                    if city == "Madrid"
                    else "rgba(31,119,180,0.12)"
                ),
                line=dict(color=color),
                name=city,
                legendgroup=city,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    figure.update_layout(
        title="Perfil climático mensual por ciudad",
        height=520,
        legend_title="Ciudad",
    )
    for polar_name in ("polar", "polar2"):
        figure.update_layout(
            **{
                polar_name: {
                    "angularaxis": {
                        "categoryorder": "array",
                        "categoryarray": MONTHS,
                    }
                }
            }
        )
    return figure, monthly


def format_evolution_interpretation(df_total):
    mean_temperature = df_total.groupby("city")[
        "metrics.temperature.avg"
    ].mean()
    mean_range = df_total.groupby("city")["thermal_range"].mean()
    warmest_city = mean_temperature.idxmax()
    widest_range_city = mean_range.idxmax()
    return (
        f"En el período disponible, **{warmest_city}** presenta la mayor "
        f"temperatura media histórica ({mean_temperature[warmest_city]:.1f} °C). "
        f"**{widest_range_city}** muestra la mayor oscilación térmica media "
        f"entre máximas y mínimas ({mean_range[widest_range_city]:.1f} °C)."
    )


def format_radar_interpretation(monthly):
    temperature_peaks = monthly.dropna(subset=["temperatura_media"])
    temperature_peaks = temperature_peaks.loc[
        temperature_peaks.groupby("city")["temperatura_media"].idxmax()
    ].set_index("city")

    precipitation_peaks = monthly.dropna(subset=["precipitacion_media"])
    precipitation_peaks = precipitation_peaks.loc[
        precipitation_peaks.groupby("city")["precipitacion_media"].idxmax()
    ].set_index("city")

    insights = []
    for city in CITY_COLORS:
        if city not in temperature_peaks.index or city not in precipitation_peaks.index:
            continue
        hottest_month = MONTHS[int(temperature_peaks.loc[city, "month"]) - 1]
        wettest_month = MONTHS[int(precipitation_peaks.loc[city, "month"]) - 1]
        insights.append(
            f"**{city}** alcanza su temperatura media mensual más alta en "
            f"{hottest_month} y su precipitación media más alta en {wettest_month}"
        )
    return ". ".join(insights) + "." if insights else "No hay datos mensuales suficientes."


def main():
    st.set_page_config(page_title="Clima: Madrid y Londres", layout="wide")
    st.title("Comparativa climática: Madrid y Londres")

    dataframes = load_dashboard_data()
    df_total = add_month_date(dataframes["df_total"])
    resumen_ciudad = dataframes["resumen_ciudad"].set_index("city")
    resumen_anual = dataframes["resumen_anual"]

    st.header("Resumen instantáneo")
    latest = (
        df_total.sort_values("date")
        .groupby("city", as_index=False)
        .tail(1)
        .set_index("city")
    )
    city_columns = st.columns(2)
    for city, column in zip(CITY_COLORS, city_columns):
        with column:
            st.markdown(
                f"<h3 style='color:{CITY_COLORS[city]}'>{city}</h3>",
                unsafe_allow_html=True,
            )
            if city not in latest.index or city not in resumen_ciudad.index:
                st.info("No hay datos disponibles para esta ciudad.")
                continue

            last_record = latest.loc[city]
            city_summary = resumen_ciudad.loc[city]
            year = int(last_record["year"])
            annual_summary = resumen_anual[
                resumen_anual["city"].eq(city)
                & resumen_anual["year"].eq(year)
            ]
            historical_temperature, historical_rain = st.columns(2)
            historical_temperature.metric(
                "Temperatura media histórica",
                f"{city_summary['temperatura_media_historica']:.1f} °C",
            )
            historical_rain.metric(
                "Precipitación acumulada",
                f"{city_summary['precipitacion_total']:.1f} mm",
            )
            latest_temperature, annual_extremes = st.columns(2)
            latest_temperature.metric(
                "Temperatura media del último mes",
                f"{last_record['metrics.temperature.avg']:.1f} °C",
            )
            if annual_summary.empty:
                annual_extremes.metric(f"Extremos de {year}", "Sin datos")
            else:
                annual = annual_summary.iloc[0]
                annual_extremes.metric(
                    f"Máx. / mín. de {year}",
                    f"{annual['temperatura_maxima_absoluta']:.1f} / "
                    f"{annual['temperatura_minima_absoluta']:.1f} °C",
                )

    st.header("Evolución de temperaturas")
    temperature_figure = create_temperature_figure(df_total)
    if temperature_figure is None:
        st.info("No hay datos suficientes para mostrar la evolución térmica.")
    else:
        st.plotly_chart(temperature_figure, use_container_width=True)
        st.markdown(format_evolution_interpretation(df_total))

    st.header("Perfil climático mensual")
    st.caption(
        "El radar separa temperatura y precipitación en paneles distintos "
        "para mantener sus unidades y escalas."
    )
    radar_figure, monthly_profile = create_climate_radar(df_total)
    if radar_figure is None:
        st.info("No hay datos mensuales suficientes para mostrar el radar.")
    else:
        st.plotly_chart(radar_figure, use_container_width=True)
        st.markdown(format_radar_interpretation(monthly_profile))


if __name__ == "__main__":
    main()
