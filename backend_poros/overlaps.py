from __future__ import annotations

from typing import Dict, Tuple, Set, List

import numpy as np

from .models import OverlapResult


def calcular_superposiciones(
    labels_by_image: Dict[str, np.ndarray]
) -> OverlapResult:
    """
    Recibe un diccionario nombre_imagen -> matriz de etiquetas (int32, 0=fondo)
    y calcula qué poros se solapan entre imágenes consecutivas.
    """
    nombres_imagenes: List[str] = sorted(labels_by_image.keys())
    superposiciones: Dict[Tuple[str, str], Dict[int, Set[int]]] = {}

    for i in range(len(nombres_imagenes) - 1):
        nombre_actual = nombres_imagenes[i]
        nombre_siguiente = nombres_imagenes[i + 1]

        matriz_actual = np.array(labels_by_image[nombre_actual])
        matriz_siguiente = np.array(labels_by_image[nombre_siguiente])

        if matriz_actual.shape != matriz_siguiente.shape:
            print(
                f"[calcular_superposiciones] ⚠ Formas distintas entre "
                f"{nombre_actual} y {nombre_siguiente}: "
                f"{matriz_actual.shape} vs {matriz_siguiente.shape}. Se omite este par."
            )
            continue

        # Máscaras de donde hay poros (id > 0)
        mask_actual = matriz_actual > 0
        mask_siguiente = matriz_siguiente > 0
        mask_solapadas = mask_actual & mask_siguiente

        filas, columnas = np.where(mask_solapadas)

        par_imagenes = (nombre_actual, nombre_siguiente)
        mapeo_poros: Dict[int, Set[int]] = {}
        superposiciones[par_imagenes] = mapeo_poros

        pares_ya_vistos = set()

        for f, c in zip(filas, columnas):
            etiqueta_actual = int(matriz_actual[f, c])
            etiqueta_siguiente = int(matriz_siguiente[f, c])

            if etiqueta_actual == 0 or etiqueta_siguiente == 0:
                continue

            par = (etiqueta_actual, etiqueta_siguiente)
            if par in pares_ya_vistos:
                continue
            pares_ya_vistos.add(par)

            if etiqueta_actual not in mapeo_poros:
                mapeo_poros[etiqueta_actual] = set()
            mapeo_poros[etiqueta_actual].add(etiqueta_siguiente)

    return OverlapResult(overlaps=superposiciones)
