import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
import zipfile
import os
from io import BytesIO
import folium
from streamlit_folium import st_folium
from folium import Choropleth, Circle, Marker, GeoJson
from folium.plugins import MarkerCluster

st.set_page_config(page_title="Gerador de Rotas por Cluster", layout="centered")
st.title("Gerador de Rotas Comerciais")

# Entrada do usuário
max_pontos_por_rota = st.number_input(
    label="Quantidade máxima de matrículas por rota:",
    min_value=10,
    max_value=1000,
    value=400,
    step=10
)

uploaded_file = st.file_uploader("Envie um shapefile (.zip contendo .shp, .shx, .dbf, etc)", type="zip")

if uploaded_file:
    # Salvar e extrair o zip
    extract_path = "/tmp/shapefile"
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Procurar o .shp
    shp_files = [f for f in os.listdir(extract_path) if f.endswith(".shp")]
    if not shp_files:
        st.error("Nenhum arquivo .shp encontrado no ZIP.")
    else:
        shp_path = os.path.join(extract_path, shp_files[0])

        # Ler os dados e definir CRS se ausente
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)

        gdf = gdf.to_crs("EPSG:31983")
        gdf = gdf.reset_index(drop=True)
        gdf["ponto_id"] = gdf.index

        # Clustering
        coords = np.array(list(gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
        n_clusters = math.ceil(len(gdf) / max_pontos_por_rota)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        gdf["rota_id"] = kmeans.fit_predict(coords)

        # Gerar rotas e ordem de visita
        linhas = []
        ordens = []

        for rota_id, grupo in gdf.groupby("rota_id"):
            coords = np.array(list(grupo.geometry.apply(lambda p: (p.x, p.y))))
            ponto_ids = grupo["ponto_id"].to_list()

            if len(coords) < 2:
                ordens.append((ponto_ids[0], 1))
                continue

            coords = coords.tolist()
            ids_restantes = ponto_ids.copy()
            visitados = [coords.pop(0)]
            visitados_ids = [ids_restantes.pop(0)]

            while coords:
                dists = cdist([visitados[-1]], coords)
                i = np.argmin(dists)
                visitados.append(coords.pop(i))
                visitados_ids.append(ids_restantes.pop(i))

            line = LineString(visitados)
            linhas.append({"rota_id": rota_id, "geometry": line})

            for ordem, ponto_id in enumerate(visitados_ids, start=1):
                ordens.append((ponto_id, ordem))

        ordem_df = pd.DataFrame(ordens, columns=["ponto_id", "ordem_visita"])
        gdf = gdf.merge(ordem_df, on="ponto_id")

        gdf_linhas = gpd.GeoDataFrame(linhas, crs=gdf.crs)

        # Mostrar número de rotas
        st.success(f"Rotas geradas: {gdf['rota_id'].nunique()}")

        # Visualização no mapa com Folium
        st.subheader("Visualização das Rotas no Mapa")
        gdf_latlon = gdf.to_crs("EPSG:4326")
        gdf_linhas_latlon = gdf_linhas.to_crs("EPSG:4326")

        # Criar o mapa
        center = gdf_latlon.geometry.unary_union.centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=13)

        # Cores para rotas
        import random
        random.seed(42)
        rotas_unicas = gdf_latlon["rota_id"].unique()
        cores_rotas = {}
        for rota in rotas_unicas:
            cor = f"#{random.randint(0, 0xFFFFFF):06x}"
            cores_rotas[rota] = cor

        # Adicionar linhas com função style segura
        for _, row in gdf_linhas_latlon.iterrows():
            cor_rota = cores_rotas[row['rota_id']]
            def style_func(x, cor=cor_rota):
                return {"color": cor, "weight": 3}
            folium.GeoJson(
                row.geometry,
                name=f"Rota {row['rota_id']}",
                style_function=style_func
            ).add_to(m)

        st_folium(m, width=700, height=500)

        # Botão de download dos resultados em GeoPackage
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_buffer:
            pontos_path = os.path.join(extract_path, "pontos_com_rotas.gpkg")
            linhas_path = os.path.join(extract_path, "linhas_rotas.gpkg")
            gdf.to_file(pontos_path, layer="pontos", driver="GPKG")
            gdf_linhas.to_file(linhas_path, layer="linhas", driver="GPKG")

            zip_buffer.write(pontos_path, arcname="pontos_com_rotas.gpkg")
            zip_buffer.write(linhas_path, arcname="linhas_rotas.gpkg")

        st.download_button(
            label="📂 Baixar Resultado (GeoPackage ZIP)",
            data=buffer.getvalue(),
            file_name="rotas_resultado.zip",
            mime="application/zip"
        )
