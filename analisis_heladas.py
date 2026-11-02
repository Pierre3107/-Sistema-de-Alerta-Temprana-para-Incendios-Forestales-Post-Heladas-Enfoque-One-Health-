import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------------------

RUTA_CSV = "DINAC_264789_minima.csv"  # cambia por tu archivo

COL_CODIGO = "Código"
COL_ESTACION = "Estación"
COL_ANIO = "Año"
COL_MES = "Mes"

# Columnas de días (Día01, Día02, ..., Día31)
COLUMNAS_DIAS = [f"Día{str(d).zfill(2)}" for d in range(1, 32)]

ANIO_MIN = 2001
ANIO_MAX = 2025

# ------------------------------------------------------------
# 2. CARGA DEL ARCHIVO
# ------------------------------------------------------------

df = pd.read_csv(
    RUTA_CSV,
    sep=';',
    skiprows=5,
    encoding='utf-8-sig',
    decimal=',',
)

# Convertir Año a número
df['Año'] = pd.to_numeric(df['Año'], errors='coerce')

# Identificar todas las columnas DíaXX
columnas_dias = [col for col in df.columns if col.startswith("Día")]

# Convertir cada columna DíaXX a número
for col in columnas_dias:
    df[col] = (
        df[col]
        .astype(str)          # asegurar que todo es texto
        .str.replace(',', '.', regex=False)  # cambiar coma por punto
        .str.strip()          # eliminar espacios
        .replace('', None)    # convertir vacío a None
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Asegurar nombres consistentes (por si vienen con espacios)
df.columns = [c.strip() for c in df.columns]

# Filtrar años de interés
df = df[(df[COL_ANIO] >= ANIO_MIN) & (df[COL_ANIO] <= ANIO_MAX)]
print(df[['Día01','Día02','Día03']].dtypes)

# ------------------------------------------------------------
# 3. REESTRUCTURAR A FORMATO DIARIO (LONG)
# ------------------------------------------------------------

# Pasar de columnas Día01..Día31 a filas
df_long = df.melt(
    id_vars=[COL_CODIGO, COL_ESTACION, COL_ANIO, COL_MES],
    value_vars=COLUMNAS_DIAS,
    var_name="col_dia",
    value_name="tmin"
)

# Extraer número de día desde "Día01" -> 1, etc.
df_long["dia"] = df_long["col_dia"].str.replace("Día", "", regex=False).astype(int)

# Crear fecha (si quieres tenerla explícita)
# Asumimos que Mes es 1–12 y Año es completo
df_long["fecha"] = pd.to_datetime(
    dict(year=df_long[COL_ANIO], month=df_long[COL_MES], day=df_long["dia"]),
    errors="coerce"
)

# Eliminar filas sin fecha válida (por ejemplo, Día31 en meses de 30 días)
df_long = df_long.dropna(subset=["fecha"])

# Asegurar numérico en tmin
df_long["tmin"] = pd.to_numeric(df_long["tmin"], errors="coerce")

# ------------------------------------------------------------
# 4. CLASIFICACIÓN DE EVENTOS (OMM + AGRONÓMICO)
# ------------------------------------------------------------

df_long["helada_meteorologica"] = df_long["tmin"] <= 0
df_long["helada_agronomica"] = df_long["tmin"] <= 2
df_long["estres_frio"] = (df_long["tmin"] > 0) & (df_long["tmin"] <= 5)

def clasificar_evento(row):
    if pd.isna(row["tmin"]):
        return None
    if row["helada_meteorologica"]:
        return "helada_meteorologica"
    elif row["helada_agronomica"]:
        return "helada_agronomica"
    elif row["estres_frio"]:
        return "estres_frio"
    else:
        return None

df_long["tipo_evento"] = df_long.apply(clasificar_evento, axis=1)

eventos = df_long[df_long["tipo_evento"].notna()].copy()

# ------------------------------------------------------------
# 5. TABLA DETALLADA DE EVENTOS
#    (aquí luego puedes añadir departamento/ciudad si los tienes)
# ------------------------------------------------------------

tabla_eventos = eventos[[
    COL_ANIO, COL_MES, "dia",
    COL_CODIGO, COL_ESTACION,
    "tmin", "tipo_evento"
]].rename(columns={
    COL_ANIO: "anio",
    COL_MES: "mes",
    "dia": "dia",
    COL_CODIGO: "codigo_estacion",
    COL_ESTACION: "estacion"
}).sort_values(["anio", "mes", "dia", "estacion"])

tabla_eventos.to_csv("tabla_eventos_heladas_estres_frio.csv", index=False)

# ------------------------------------------------------------
# 6. RESUMEN NACIONAL
# ------------------------------------------------------------

# a) Total de días de evento por año y tipo
res_anual = (
    tabla_eventos
    .groupby(["anio", "tipo_evento"])
    .size()
    .reset_index(name="n_dias_evento")
)
res_anual.to_csv("resumen_nacional_anual_eventos.csv", index=False)

# b) Meses con eventos por año y tipo
meses_por_anio = (
    tabla_eventos
    .groupby(["anio", "tipo_evento"])["mes"]
    .apply(lambda x: sorted(x.unique()))
    .reset_index(name="meses_con_evento")
)
meses_por_anio.to_csv("resumen_meses_por_anio_eventos.csv", index=False)

# c) Total mensual acumulado por año, mes y tipo
res_mensual = (
    tabla_eventos
    .groupby(["anio", "mes", "tipo_evento"])
    .size()
    .reset_index(name="n_dias_evento")
)
res_mensual.to_csv("resumen_mensual_eventos.csv", index=False)

# ------------------------------------------------------------
# 7. RANKING DE ESTACIONES
#    (luego puedes mapear estación -> departamento/ciudad)
# ------------------------------------------------------------

ranking_estaciones = (
    tabla_eventos
    .groupby(["estacion", "tipo_evento"])
    .size()
    .reset_index(name="n_dias_evento")
    .sort_values(["tipo_evento", "n_dias_evento"], ascending=[True, False])
)
ranking_estaciones.to_csv("ranking_estaciones_eventos.csv", index=False)

# ------------------------------------------------------------
# 8. RESUMEN POR ESTACIÓN (ANÁLOGO A DEPARTAMENTO/CIUDAD)
# ------------------------------------------------------------

res_estacion_mes_anio = (
    tabla_eventos
    .groupby(["estacion", "tipo_evento", "anio", "mes"])
    .size()
    .reset_index(name="n_dias_evento")
)
res_estacion_mes_anio.to_csv("resumen_estacion_mes_anio_eventos.csv", index=False)

def lista_unica_ordenada(x):
    return ",".join(str(v) for v in sorted(x.unique()))

res_estacion_compacto = (
    res_estacion_mes_anio
    .groupby(["estacion", "tipo_evento"])
    .agg({
        "mes": lista_unica_ordenada,
        "anio": lista_unica_ordenada,
        "n_dias_evento": "sum"
    })
    .reset_index()
    .rename(columns={
        "mes": "meses_con_evento",
        "anio": "anios_con_evento",
        "n_dias_evento": "total_dias_evento"
    })
)
res_estacion_compacto.to_csv("resumen_estacion_compacto_eventos.csv", index=False)

print("Listo: eventos diarios, resúmenes y rankings generados.")