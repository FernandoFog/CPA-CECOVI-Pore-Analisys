from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import networkx as nx

from .models import OverlapResult, GraphResult


def construir_grafo_ids(overlaps: OverlapResult) -> GraphResult:
    """
    Grafo donde cada nodo es un ID global de poro (int).
    Hay arista entre dos poros si se solapan en cortes consecutivos.
    """
    G = nx.Graph()
    image_pairs = overlaps.overlaps

    # deducir orden de imágenes a partir de las claves
    imagenes = sorted({img for par in image_pairs.keys() for img in par})
    image_order = imagenes

    for (img1, img2), mapeo in image_pairs.items():
        for pore1, poros2 in mapeo.items():
            G.add_node(int(pore1))
            for pore2 in poros2:
                G.add_node(int(pore2))
                G.add_edge(int(pore1), int(pore2))

    return GraphResult(graph=G, image_order=image_order)


def construir_grafo_por_capa(overlaps: OverlapResult) -> GraphResult:
    """
    Versión donde cada nodo es (imagen, id_poro) y se conecta a la siguiente capa.
    """
    imagenes = sorted({img for par in overlaps.overlaps.keys() for img in par})
    indice_imagen = {img: i for i, img in enumerate(imagenes)}

    G = nx.DiGraph()

    for (img1, img2), mapeo in overlaps.overlaps.items():
        capa1 = indice_imagen[img1]
        capa2 = indice_imagen[img2]

        for pore1, lista_poros2 in mapeo.items():
            nodo1 = (img1, int(pore1))
            if nodo1 not in G:
                G.add_node(nodo1, imagen=img1, pore_id=int(pore1), capa=capa1)

            for pore2 in lista_poros2:
                nodo2 = (img2, int(pore2))
                if nodo2 not in G:
                    G.add_node(nodo2, imagen=img2, pore_id=int(pore2), capa=capa2)

                G.add_edge(nodo1, nodo2)

    return GraphResult(graph=G, image_order=imagenes)


def calcular_posiciones_por_capa(graph_result: GraphResult) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """
    Calcula posiciones (x, y) para un grafo construido con construir_grafo_por_capa.
    Devuelve un dict nodo -> (x, y).
    """
    G = graph_result.graph
    imagenes = graph_result.image_order
    indice_imagen = {img: i for i, img in enumerate(imagenes)}

    nodos_por_capa: Dict[int, List] = defaultdict(list)
    for nodo, data in G.nodes(data=True):
        capa = data.get("capa", indice_imagen[data["imagen"]])
        nodos_por_capa[capa].append(nodo)

    pos: Dict[Tuple[str, int], Tuple[float, float]] = {}

    for capa, nodos in nodos_por_capa.items():
        n = len(nodos)
        if n == 1:
            ys = [0.5]
        else:
            ys = np.linspace(0.1, 0.9, n)
        for y, nodo in zip(ys, nodos):
            pos[nodo] = (float(capa), float(y))

    return pos
