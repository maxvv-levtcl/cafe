# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Analisis de Demanda Cafeteria — version Streamlit
Convertido desde notebook Colab original.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ─── Configuracion de pagina ────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis de Demanda Cafeteria",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Estilos globales ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TOD_ORDER     = ["Morning", "Afternoon", "Night"]
SEASON_MAP    = {
    "Dec": "Winter", "Jan": "Winter", "Feb": "Winter",
    "Mar": "Spring", "Apr": "Spring", "May": "Spring",
    "Jun": "Summer", "Jul": "Summer", "Aug": "Summer",
    "Sep": "Autumn", "Oct": "Autumn", "Nov": "Autumn"
}
PRODUCT_COLORS = px.colors.qualitative.Set2


# ════════════════════════════════════════════════════════════════════════════
# CABECERA DEL INFORME
# ════════════════════════════════════════════════════════════════════════════
st.title("☕ Análisis de Demanda Temporal y Oportunidades de Rentabilidad")

st.markdown("""
**Dataset:** Coffee Store Sales &nbsp;|&nbsp; **Periodo:** Mar 2024 – Mar 2025

**Objetivo:** Identificar patrones de demanda por producto, día de la semana y momento del día.
Derivar recomendaciones accionables de mix y promoción.

---

### Nota metodológica

| Aspecto | Detalle |
|---------|---------|
| Variable dependiente | log(qty) en todos los modelos |
| Regresor principal | Dummies temporales exógenas |
| Control | Año incluido en modelo |
| Implementación | Función reutilizable con statsmodels |
""")


# ════════════════════════════════════════════════════════════════════════════
# 2. CARGA Y PREPARACION DEL DATASET
# ════════════════════════════════════════════════════════════════════════════
st.header("2. Carga y preparación del dataset")

@st.cache_data(ttl=3600, show_spinner="⏳ Cargando datos desde Google Drive…")
def cargar_datos():
    """Carga y prepara el dataset."""
    file_id = '1KsCCqVbNLIHXn2ORRxDCloEKUJTdT94K'
    url_drive = f'https://drive.google.com/uc?id={file_id}'
    _df = pd.read_excel(url_drive, engine='openpyxl')

    _df["date"]        = pd.to_datetime(_df["date"])
    _df["datetime"]    = pd.to_datetime(_df["datetime"])
    _df["year"]        = _df["date"].dt.year
    _df["season"]      = _df["Month_name"].map(SEASON_MAP)
    _df["year_month"]  = _df["date"].dt.to_period("M")
    _df["Weekday"]     = pd.Categorical(_df["Weekday"],     categories=WEEKDAY_ORDER, ordered=True)
    _df["Time_of_Day"] = pd.Categorical(_df["Time_of_Day"], categories=TOD_ORDER,     ordered=True)
    return _df

df = cargar_datos()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Registros", f"{len(df):,}")
with col2:
    st.metric("Productos", df['coffee_name'].nunique())
with col3:
    st.metric("Periodo", f"{df['date'].min().date()} a {df['date'].max().date()}")

nulos = df.isnull().sum()
nulos_dict = nulos[nulos > 0].to_dict()
if nulos_dict:
    st.warning(f"**Valores nulos detectados:** {nulos_dict}")

st.subheader("📊 Estadísticas descriptivas")
st.dataframe(df[["money", "hour_of_day"]].describe().round(2), use_container_width=True)

st.subheader("🔍 Primeras filas")
st.dataframe(df.head(10), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 3. ANALISIS EXPLORATORIO DE DATOS
# ════════════════════════════════════════════════════════════════════════════
st.header("3. Análisis Exploratorio de Datos")

product_stats = (
    df.groupby("coffee_name")
      .agg(qty=("money","count"), revenue=("money","sum"), avg_price=("money","mean"))
      .sort_values("revenue", ascending=False)
      .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].barh(product_stats["coffee_name"], product_stats["qty"],
             color=sns.color_palette("Blues_d", len(product_stats)))
axes[0].set_title("Transacciones por producto"); axes[0].set_xlabel("N ventas")

axes[1].barh(product_stats["coffee_name"], product_stats["revenue"],
             color=sns.color_palette("Greens_d", len(product_stats)))
axes[1].set_title("Ingresos totales"); axes[1].set_xlabel("Ingresos ($)")

axes[2].barh(product_stats["coffee_name"], product_stats["avg_price"],
             color=sns.color_palette("Oranges_d", len(product_stats)))
axes[2].set_title("Precio promedio"); axes[2].set_xlabel("Precio ($)")

plt.suptitle("Overview del mix de productos", fontsize=14, y=1.02)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.dataframe(product_stats.round(2), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 4. HEATMAPS: DEMANDA POR DIA x MOMENTO
# ════════════════════════════════════════════════════════════════════════════
st.header("4. Heatmaps: Demanda por Día × Momento")
st.markdown("""
Visualización clave que permite identificar de un vistazo
en qué combinación día-horario cada producto vende más o menos.
""")

products = df["coffee_name"].value_counts().index.tolist()
ncols = 2
nrows = -(-len(products) // ncols)

# Heatmap de INGRESOS
st.subheader("💰 Ingresos totales por Día × Momento ($)")
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4.5))
axes = axes.flatten()

for i, prod in enumerate(products):
    sub = df[df["coffee_name"] == prod]
    pivot = (
        sub.groupby(["Weekday", "Time_of_Day"])["money"]
           .sum()
           .unstack("Time_of_Day")
           .reindex(WEEKDAY_ORDER)
           .reindex(columns=TOD_ORDER)
    )
    sns.heatmap(pivot, ax=axes[i], annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, cbar=True, annot_kws={"size": 9})
    axes[i].set_title(f"{prod}  (N={len(sub):,})", fontsize=11, fontweight="bold")
    axes[i].set_xlabel(""); axes[i].set_ylabel("")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("INGRESOS totales por Dia x Momento ($)", fontsize=13, y=1.01)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Heatmap de CANTIDAD
st.subheader("📈 N° de transacciones por Día × Momento")
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4.5))
axes = axes.flatten()

for i, prod in enumerate(products):
    sub = df[df["coffee_name"] == prod]
    pivot = (
        sub.groupby(["Weekday", "Time_of_Day"])["money"]
           .count()
           .unstack("Time_of_Day")
           .reindex(WEEKDAY_ORDER)
           .reindex(columns=TOD_ORDER)
    )
    sns.heatmap(pivot, ax=axes[i], annot=True, fmt=".0f", cmap="Blues",
                linewidths=0.5, cbar=True, annot_kws={"size": 9})
    axes[i].set_title(f"{prod}  (N={len(sub):,})", fontsize=11, fontweight="bold")
    axes[i].set_xlabel(""); axes[i].set_ylabel("")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("N de transacciones por Dia x Momento", fontsize=13, y=1.01)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 5. EVOLUCION MENSUAL E INGRESOS POR HORA
# ════════════════════════════════════════════════════════════════════════════
st.header("5. Evolución mensual e ingresos por hora")

monthly = (
    df.groupby(["year_month", "coffee_name"])["money"]
      .sum().reset_index()
      .assign(year_month=lambda x: x["year_month"].astype(str))
)

fig_line = px.line(monthly, x="year_month", y="money", color="coffee_name",
                   title="Evolución mensual de ingresos por producto",
                   labels={"money": "Ingresos ($)", "year_month": "Mes", "coffee_name": "Producto"},
                   color_discrete_sequence=PRODUCT_COLORS)
fig_line.update_layout(xaxis_tickangle=-45, legend_title="Producto")
st.plotly_chart(fig_line, use_container_width=True)

hourly = df.groupby(["hour_of_day", "coffee_name"])["money"].sum().reset_index()
fig_bar = px.bar(hourly, x="hour_of_day", y="money", color="coffee_name",
                 title="Ingresos por hora del día",
                 labels={"money": "Ingresos ($)", "hour_of_day": "Hora", "coffee_name": "Producto"},
                 color_discrete_sequence=PRODUCT_COLORS, barmode="stack")
fig_bar.update_layout(xaxis=dict(dtick=1), legend_title="Producto")
st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 6. MODELO OLS DE EFECTOS TEMPORALES
# ════════════════════════════════════════════════════════════════════════════
st.header("6. Modelo de efectos temporales sobre la demanda (OLS)")
st.markdown("""
**Especificación:**
```
log(qty_d,t,s,a) = alfa + B1*Dia + B2*Momento + B3*Estacion + B4*Ano + error
```

- **Variable dependiente:** log(cantidad de ventas) por celda Día × Momento × Estación × Año
- **Regresores:** dummies temporales, todos exógenos
- **Interpretación:** diferencia porcentual en demanda vs. referencia (Lunes · Mañana · Invierno)
""")

def analizar_producto(nombre, _df, ref_dia="Mon", ref_tod="Morning", ref_season="Winter"):
    """Modelo OLS de efectos temporales."""
    sub = _df[_df["coffee_name"] == nombre].copy()
    grp_cols = ["Weekday", "Time_of_Day", "season", "year"]
    model_data = (
        sub.groupby(grp_cols)
           .agg(qty=("money", "count"), revenue=("money", "sum"))
           .reset_index()
    )
    model_data["log_qty"] = np.log(model_data["qty"] + 1)
    formula = (
        f'log_qty ~ '
        f'C(Weekday, Treatment(reference="{ref_dia}")) + '
        f'C(Time_of_Day, Treatment(reference="{ref_tod}")) + '
        f'C(season, Treatment(reference="{ref_season}")) + '
        f'C(year)'
    )
    return smf.ols(formula=formula, data=model_data).fit(), model_data


def extraer_efectos(modelo, nombre):
    """Extrae coeficientes con IC 95% y p-values."""
    out = pd.DataFrame({
        "coef":  modelo.params,
        "lower": modelo.conf_int()[0],
        "upper": modelo.conf_int()[1],
        "p":     modelo.pvalues
    }).drop("Intercept", errors="ignore")
    out["sig"] = out["p"].apply(lambda p: "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.1 else "")))
    out["producto"] = nombre
    return out.reset_index().rename(columns={"index": "variable"})


resultados    = {}
todos_efectos = []

for prod in products:
    n = (df["coffee_name"] == prod).sum()
    if n < 50:
        st.warning(f"⚠️ {prod}: solo {n} obs., omitido")
        continue

    modelo, data = analizar_producto(prod, df)
    resultados[prod] = (modelo, data)
    todos_efectos.append(extraer_efectos(modelo, prod))

    with st.expander(f"📊 {prod} — N celdas: {len(data)} | R²={modelo.rsquared:.3f}"):
        coef_df = pd.read_html(
            modelo.summary().tables[1].as_html(), header=0, index_col=0
        )[0]
        st.dataframe(coef_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZACION DE EFECTOS TEMPORALES
# ════════════════════════════════════════════════════════════════════════════
st.header("7. Visualización de efectos temporales")

all_eff  = pd.concat(todos_efectos, ignore_index=True)
temporal = all_eff[all_eff["variable"].str.contains("Weekday|Time_of_Day", regex=True)].copy()

temporal["label"] = (
    temporal["variable"]
    .str.replace(r"C\(Weekday.*?\)\[T\.", "", regex=True)
    .str.replace(r"C\(Time_of_Day.*?\)\[T\.", "Momento: ", regex=True)
    .str.replace("]", "", regex=False)
)

fig_scatter = px.scatter(
    temporal, x="coef", y="label", color="producto",
    facet_col="producto", facet_col_wrap=3,
    title="Efectos temporales sobre log(demanda) — coeficientes OLS con IC 95%",
    labels={"coef": "Coef. vs Lunes/Mañana", "label": ""},
    color_discrete_sequence=PRODUCT_COLORS, height=650
)
fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_scatter.update_traces(marker=dict(size=8))
fig_scatter.update_layout(showlegend=False)
st.plotly_chart(fig_scatter, use_container_width=True)
st.caption("💡 Coef positivo = mayor demanda que el día/momento de referencia (Lunes mañana)")


# ════════════════════════════════════════════════════════════════════════════
# 8. TABLA DE OPORTUNIDADES DE RENTABILIDAD
# ════════════════════════════════════════════════════════════════════════════
st.header("8. Tabla de oportunidades de rentabilidad")
st.markdown("""
Cruzamos demanda actual vs. demanda esperada uniforme por slot (Día × Momento)
con el precio promedio del producto. Resultado: ranking de slots con mayor potencial.
""")

slot_stats = (
    df.groupby(["coffee_name", "Weekday", "Time_of_Day"])
      .agg(qty=("money","count"), revenue=("money","sum"), avg_price=("money","mean"))
      .reset_index()
)

total_qty                  = slot_stats.groupby("coffee_name")["qty"].transform("sum")
n_slots                    = slot_stats.groupby("coffee_name")["qty"].transform("count")
slot_stats["expected"]     = total_qty / n_slots
slot_stats["gap_qty"]      = slot_stats["expected"] - slot_stats["qty"]
slot_stats["gap_revenue"]  = slot_stats["gap_qty"] * slot_stats["avg_price"]
slot_stats["share_pct"]    = slot_stats["qty"] / total_qty * 100

oport = (
    slot_stats[slot_stats["gap_qty"] > 0]
    .sort_values("gap_revenue", ascending=False)
    .head(20)
    [["coffee_name","Weekday","Time_of_Day","qty","expected","gap_qty","avg_price","gap_revenue","share_pct"]]
)
oport.columns = ["Producto","Dia","Momento","Qty actual","Qty esperada",
                 "Brecha (u)","Precio avg ($)","Upside ($)","% demanda"]

st.subheader("🎯 TOP 20 OPORTUNIDADES — mayor upside potencial de ingresos")
st.caption("Brecha = Demanda uniforme esperada menos Demanda actual del slot")

st.dataframe(
    oport.round(1).reset_index(drop=True)
         .style.background_gradient(subset=["Upside ($)"], cmap="Greens"),
    use_container_width=True,
)

# Heatmap de brechas
st.subheader("🗺️ Heatmap de brechas — top 4 productos")
st.caption("Verde = slot sub-explotado (oportunidad) | Rojo = slot ya en su máximo")

top4 = df["coffee_name"].value_counts().head(4).index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, prod in enumerate(top4):
    sub = slot_stats[slot_stats["coffee_name"] == prod].copy()
    pivot = (
        sub.set_index(["Weekday", "Time_of_Day"])["gap_revenue"]
           .unstack("Time_of_Day")
           .reindex(WEEKDAY_ORDER)
           .reindex(columns=TOD_ORDER)
    )
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]))
    sns.heatmap(pivot, ax=axes[i], annot=True, fmt=".0f", cmap="RdYlGn",
                linewidths=0.5, cbar=True, annot_kws={"size": 9},
                center=0, vmin=-vmax, vmax=vmax)
    axes[i].set_title(f"Brecha de ingresos — {prod} ($)", fontsize=11, fontweight="bold")
    axes[i].set_xlabel(""); axes[i].set_ylabel("")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 9. VARIACION DE PRECIOS
# ════════════════════════════════════════════════════════════════════════════
st.header("9. Variación de precios detectada (análisis descriptivo)")
st.markdown("""
Se detectó repricing en agosto 2024. Este ejercicio es **solo descriptivo** —
no equivale a elasticidad precio porque el cambio de precio coincide con factores
estacionales (confounding) y no fue un experimento aleatorio controlado.
""")

monthly_det = (
    df.groupby(["year_month", "coffee_name"])
      .agg(qty=("money","count"), avg_price=("money","mean"))
      .reset_index()
      .assign(year_month=lambda x: x["year_month"].astype(str))
)

fig_combo = make_subplots(rows=2, cols=2,
    subplot_titles=[f"{p}" for p in top4],
    specs=[[{"secondary_y": True}, {"secondary_y": True}],
           [{"secondary_y": True}, {"secondary_y": True}]])

positions = [(1,1), (1,2), (2,1), (2,2)]
for idx, prod in enumerate(top4):
    r, c = positions[idx]
    sub = monthly_det[monthly_det["coffee_name"] == prod]
    fig_combo.add_trace(
        go.Bar(x=sub["year_month"], y=sub["qty"],
               name="Qty", marker_color="lightblue", showlegend=(idx==0)),
        row=r, col=c, secondary_y=False)
    fig_combo.add_trace(
        go.Scatter(x=sub["year_month"], y=sub["avg_price"],
                   name="Precio ($)", mode="lines+markers",
                   line=dict(color="red", width=2), showlegend=(idx==0)),
        row=r, col=c, secondary_y=True)

fig_combo.update_layout(title_text="Cantidad mensual vs. Precio promedio (descripción)", height=620)
st.plotly_chart(fig_combo, use_container_width=True)

st.info(
    "**Observación:** el repricing de ago-2024 no produjo un rebote claro de volumen. "
    "Esto sugiere demanda relativamente inelástica, pero la evidencia es débil "
    "(un evento, confounding estacional)."
)


# ════════════════════════════════════════════════════════════════════════════
# 10. RESUMEN EJECUTIVO Y RECOMENDACIONES
# ════════════════════════════════════════════════════════════════════════════
st.header("10. Resumen ejecutivo y recomendaciones estratégicas")

st.subheader("📋 Hallazgos principales")
st.markdown("""
| Dimensión | Hallazgo |
|-----------|----------|
| **Producto estrella** | Americano with Milk y Latte concentran >44 % de ingresos totales |
| **Alto margen, bajo volumen** | Cappuccino y Cocoa: precio promedio ~$36 pero mucho menor volumen |
| **Mejor slot horario** | Morning es el horario líder en la mayoría de productos |
| **Día más fuerte** | Martes y jueves mañana en Americano with Milk; lunes noche en Latte |
| **Mayor oportunidad** | Domingos y viernes noche muestran brechas en casi todos los productos |
| **Efecto precio** | Evidencia descriptiva débil de sensibilidad; demanda más explicada por día/hora |
""")

st.subheader("💡 Estrategias accionables")

st.markdown("""
**1. Impulsar Cappuccino y Cocoa en tardes de lunes a miércoles**
Slots con brechas positivas y precio alto (~$36). Un combo de *"tarde premium"* podría sumar $300–500/semana.

**2. Activar Hot Chocolate en Otoño-Invierno**
OLS confirma efectos estacionales. Periodo oct-feb es el más propicio: comunicación activa de temporada.

**3. Promoción lunes mañana para Americano with Milk**
Martes/jueves son el peak; el lunes muestra brecha. *"Arranca la semana"* podría capturar 10–15 unidades adicionales.

**4. Menú nocturno rotativo**
Latte domina noche lun-jue pero deja espacio para Cappuccino. Rotación por día puede elevar el ticket promedio nocturno.

**5. Test de repricing controlado**
La evidencia descriptiva sugiere baja elasticidad en productos premium. Una subida moderada en Cappuccino/Cocoa/Hot Chocolate
podría mejorar margen sin sacrificar volumen — pero requiere A/B test controlado antes de generalizar.
""")

st.divider()
st.success("✅ Análisis completado. Los datos se cargan automáticamente desde Google Drive.")
