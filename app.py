import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Acidentes nas Estradas Indianas",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


SEVERITY_COLORS = {"fatal": "#E63946", "major": "#F4A261", "minor": "#2A9D8F"}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("indian_roads_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month_name()
    df["festival"] = df["festival"].fillna("Sem festival").astype(str)

    text_cols = [
        "state",
        "city",
        "accident_severity",
        "weather",
        "road_type",
        "cause",
        "traffic_density",
        "visibility",
        "day_of_week",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("Não informado").astype(str)

    return df


def safe_mode(series: pd.Series) -> float:
    modes = series.mode(dropna=True)
    return float(modes.iloc[0]) if not modes.empty else np.nan


def stats_table(data: pd.DataFrame, cols: list[str], names: list[str]) -> pd.DataFrame:
    rows = []
    for col, label in zip(cols, names):
        s = data[col].dropna()
        if s.empty:
            rows.append(
                {
                    "Variável": label,
                    "Média": np.nan,
                    "Mediana": np.nan,
                    "Moda": np.nan,
                    "Amplitude": np.nan,
                    "Desvio Padrão": np.nan,
                    "Variância": np.nan,
                    "CV (%)": np.nan,
                    "DIQ (Q3-Q1)": np.nan,
                }
            )
            continue

        mean = s.mean()
        std = s.std(ddof=1)
        rows.append(
            {
                "Variável": label,
                "Média": round(mean, 4),
                "Mediana": round(s.median(), 4),
                "Moda": round(safe_mode(s), 4),
                "Amplitude": round(s.max() - s.min(), 4),
                "Desvio Padrão": round(std, 4),
                "Variância": round(s.var(ddof=1), 4),
                "CV (%)": round((std / mean) * 100, 2) if mean != 0 else np.nan,
                "DIQ (Q3-Q1)": round(s.quantile(0.75) - s.quantile(0.25), 4),
            }
        )
    return pd.DataFrame(rows)


df = load_data()

st.title("🚨 Acidentes nas Estradas Indianas (2022-2025)")
st.markdown(
    """
Análise exploratória com data storytelling para entender perfil dos acidentes,
severidade, causas, horário, clima e impacto de festivais.
"""
)
st.caption(f"{len(df):,} registros | {df['date'].min().date()} até {df['date'].max().date()}")


with st.sidebar:
    st.header("🔧 Filtros")
    st.caption("Todos os indicadores e gráficos são atualizados pelos filtros abaixo.")

    severity = st.multiselect(
        "Severidade",
        sorted(df["accident_severity"].unique()),
        default=sorted(df["accident_severity"].unique()),
    )
    weather = st.multiselect(
        "Condição climática",
        sorted(df["weather"].unique()),
        default=sorted(df["weather"].unique()),
    )
    road_type = st.multiselect(
        "Tipo de via",
        sorted(df["road_type"].unique()),
        default=sorted(df["road_type"].unique()),
    )

    with st.expander("Filtros avançados"):
        states = st.multiselect("Estado", sorted(df["state"].unique()))
        cities = st.multiselect("Cidade", sorted(df["city"].unique()))
        causes = st.multiselect("Causa", sorted(df["cause"].unique()))
        festivals = st.multiselect("Festival", sorted(df["festival"].unique()))
        traffic = st.multiselect("Densidade de tráfego", sorted(df["traffic_density"].unique()))
        visibility = st.multiselect("Visibilidade", sorted(df["visibility"].unique()))

        period = st.radio("Período", ["Todos", "Fim de semana", "Dia útil"])
        peak = st.radio("Horário de pico", ["Todos", "Sim", "Não"])

    hour_range = st.slider("Hora do dia", 0, 23, (0, 23))
    temp_range = st.slider(
        "Temperatura (°C)",
        int(df["temperature"].min()),
        int(df["temperature"].max()),
        (int(df["temperature"].min()), int(df["temperature"].max())),
    )
    risk_range = st.slider("Risk score", 0.0, 1.0, (0.0, 1.0), step=0.05)


filt = df.copy()
filt = filt[filt["accident_severity"].isin(severity)]
filt = filt[filt["weather"].isin(weather)]
filt = filt[filt["road_type"].isin(road_type)]

if states:
    filt = filt[filt["state"].isin(states)]
if cities:
    filt = filt[filt["city"].isin(cities)]
if causes:
    filt = filt[filt["cause"].isin(causes)]
if festivals:
    filt = filt[filt["festival"].isin(festivals)]
if traffic:
    filt = filt[filt["traffic_density"].isin(traffic)]
if visibility:
    filt = filt[filt["visibility"].isin(visibility)]
if period == "Fim de semana":
    filt = filt[filt["is_weekend"] == 1]
if period == "Dia útil":
    filt = filt[filt["is_weekend"] == 0]
if peak == "Sim":
    filt = filt[filt["is_peak_hour"] == 1]
if peak == "Não":
    filt = filt[filt["is_peak_hour"] == 0]

filt = filt[filt["hour"].between(*hour_range)]
filt = filt[filt["temperature"].between(*temp_range)]
filt = filt[filt["risk_score"].between(*risk_range)]

if filt.empty:
    st.warning("Nenhum registro encontrado com os filtros atuais. Ajuste os filtros para continuar.")
    st.stop()


st.subheader("📌 KPIs principais")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total de acidentes", f"{len(filt):,}")
k2.metric("Acidentes fatais", f"{(filt['accident_severity'] == 'fatal').sum():,}")
k3.metric("Total de vítimas", f"{filt['casualties'].sum():,}")
k4.metric("Vítimas/acidente", f"{filt['casualties'].mean():.2f}")
k5.metric("Risk score médio", f"{filt['risk_score'].mean():.3f}")
k6.metric("Taxa de fatalidade", f"{(filt['accident_severity'] == 'fatal').mean() * 100:.1f}%")

st.divider()


st.header("📊 Seção 1 - Perfil dos acidentes")
col1, col2 = st.columns(2)

with col1:
    sev_count = filt["accident_severity"].value_counts().reset_index()
    sev_count.columns = ["accident_severity", "count"]
    fig_sev = px.bar(
        sev_count,
        x="accident_severity",
        y="count",
        color="accident_severity",
        color_discrete_map=SEVERITY_COLORS,
        labels={"accident_severity": "Severidade", "count": "Acidentes"},
        title="Distribuição por severidade",
    )
    st.plotly_chart(fig_sev, use_container_width=True)

with col2:
    cause_count = filt["cause"].value_counts().nlargest(10).reset_index()
    cause_count.columns = ["cause", "count"]
    fig_cause = px.bar(
        cause_count,
        y="cause",
        x="count",
        orientation="h",
        labels={"cause": "Causa", "count": "Acidentes"},
        title="Top 10 causas",
        color="count",
        color_continuous_scale="Oranges",
    )
    fig_cause.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_cause, use_container_width=True)

hourly = filt.groupby("hour").size().reset_index(name="count")
fig_hour = px.area(
    hourly,
    x="hour",
    y="count",
    labels={"hour": "Hora", "count": "Acidentes"},
    title="Acidentes por hora do dia",
    color_discrete_sequence=["#F4A261"],
)
st.plotly_chart(fig_hour, use_container_width=True)

st.divider()


st.header("📐 Seção 2 - Medidas de posição e dispersão")
num_cols = ["risk_score", "casualties", "vehicles_involved", "temperature"]
labels = ["Risk Score", "Vítimas", "Veículos envolvidos", "Temperatura (°C)"]
summary_df = stats_table(filt, num_cols, labels)
st.dataframe(summary_df, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig_hist = px.histogram(
        filt,
        x="risk_score",
        nbins=20,
        title="Histograma de Risk Score",
        color_discrete_sequence=["#4CC9F0"],
    )
    fig_hist.add_vline(
        x=filt["risk_score"].mean(),
        line_dash="dash",
        line_color="#F4A261",
    )
    fig_hist.add_vline(
        x=filt["risk_score"].median(),
        line_dash="dot",
        line_color="#E63946",
    )
    # Evita sobreposição/bug visual de anotações automáticas no topo do gráfico.
    fig_hist.add_annotation(
        x=0.01,
        y=1.12,
        xref="paper",
        yref="paper",
        text=f"Média: {filt['risk_score'].mean():.3f}",
        showarrow=False,
        font=dict(color="#F4A261"),
    )
    fig_hist.add_annotation(
        x=0.30,
        y=1.12,
        xref="paper",
        yref="paper",
        text=f"Mediana: {filt['risk_score'].median():.3f}",
        showarrow=False,
        font=dict(color="#E63946"),
    )
    fig_hist.update_layout(margin=dict(t=90, l=20, r=20, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

with col4:
    fig_box = px.box(
        filt,
        x="accident_severity",
        y="risk_score",
        color="accident_severity",
        color_discrete_map=SEVERITY_COLORS,
        points=False,
        title="Boxplot de Risk Score por severidade",
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()


st.header("🔗 Seção 3 - Associação entre variáveis")
corr_cols = ["risk_score", "casualties", "vehicles_involved", "temperature", "hour"]
corr = filt[corr_cols].corr(numeric_only=True)

fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdYlGn",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="Mapa de correlação",
)
st.plotly_chart(fig_corr, use_container_width=True)

sample_n = min(3000, len(filt))
sample_df = filt.sample(sample_n, random_state=42)

col5, col6 = st.columns(2)
with col5:
    fig_sc1 = px.scatter(
        sample_df,
        x="vehicles_involved",
        y="casualties",
        color="accident_severity",
        color_discrete_map=SEVERITY_COLORS,
        opacity=0.5,
        title="Veículos envolvidos vs vítimas",
    )
    st.plotly_chart(fig_sc1, use_container_width=True)

with col6:
    fig_sc2 = px.scatter(
        sample_df,
        x="risk_score",
        y="casualties",
        color="accident_severity",
        color_discrete_map=SEVERITY_COLORS,
        opacity=0.5,
        title="Risk score vs vítimas",
    )
    st.plotly_chart(fig_sc2, use_container_width=True)

st.divider()


st.header("🌦️ Seção 4 - Clima e festivais")
col7, col8 = st.columns(2)

with col7:
    weather_tab = pd.crosstab(
        filt["weather"],
        filt["accident_severity"],
        normalize="index",
    ).mul(100)
    fig_weather = px.bar(
        weather_tab,
        barmode="stack",
        color_discrete_map=SEVERITY_COLORS,
        labels={"value": "%", "weather": "Clima"},
        title="Severidade por condição climática (%)",
    )
    st.plotly_chart(fig_weather, use_container_width=True)

with col8:
    fest_tab = pd.crosstab(
        filt["festival"],
        filt["accident_severity"],
        normalize="index",
    ).mul(100)
    fig_fest = px.bar(
        fest_tab,
        barmode="stack",
        color_discrete_map=SEVERITY_COLORS,
        labels={"value": "%", "festival": "Festival"},
        title="Severidade por festival (%)",
    )
    st.plotly_chart(fig_fest, use_container_width=True)

st.divider()


st.header("🗺️ Seção 5 - Visão espacial e temporal")
col9, col10 = st.columns([3, 2])

with col9:
    map_df = filt.sample(min(3000, len(filt)), random_state=42)
    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="accident_severity",
        color_discrete_map=SEVERITY_COLORS,
        hover_data=["state", "city", "cause", "casualties", "risk_score"],
        zoom=4,
        mapbox_style="carto-positron",
        title="Mapa de acidentes",
        height=420,
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

with col10:
    monthly = filt.groupby(filt["date"].dt.to_period("M")).size().reset_index(name="count")
    monthly["date"] = monthly["date"].dt.to_timestamp()
    fig_month = px.line(
        monthly,
        x="date",
        y="count",
        markers=True,
        labels={"date": "Mês", "count": "Acidentes"},
        title="Evolução mensal de acidentes",
    )
    st.plotly_chart(fig_month, use_container_width=True)

st.divider()


st.header("✅ Insights automáticos (baseados nos filtros)")
top_cause = filt["cause"].value_counts().index[0]
fatal_mean = filt.loc[filt["accident_severity"] == "fatal", "risk_score"].mean()
minor_mean = filt.loc[filt["accident_severity"] == "minor", "risk_score"].mean()
veh_cas_corr = filt[["vehicles_involved", "casualties"]].corr().iloc[0, 1]

st.markdown(
    f"""
1. Causa mais frequente no recorte atual: **{top_cause}**.
2. Risk score médio em acidentes fatais: **{fatal_mean:.3f}**.
3. Risk score médio em acidentes leves: **{minor_mean:.3f}**.
4. Correlação entre veículos envolvidos e vítimas: **{veh_cas_corr:.2f}**.
"""
)

with st.expander("📋 Ver dados filtrados"):
    st.dataframe(filt, use_container_width=True, height=380)
 
 