from __future__ import annotations

from typing import List, Dict, Tuple
import csv
from pathlib import Path

import numpy as np


from skimage import measure
from .models import (
    MaskConfig,
    GeometryConfig,
    AnalysisResult,
    ImageAnalysisRecord,
)


def _build_circular_mask(shape: Tuple[int, int], mask_cfg: MaskConfig) -> np.ndarray:
    """
    Misma lógica geométrica que en segmentation._build_circular_mask,
    pero local a este módulo.
    """
    h, w = shape
    if not mask_cfg.use_circular_mask:
        return np.ones((h, w), dtype=bool)

    dy, dx = mask_cfg.circle_center_offset
    cy = h // 2 + dy
    cx = w // 2 + dx
    radio = min(h, w) // 2 - mask_cfg.circle_margin

    if radio <= 0:
        return np.zeros((h, w), dtype=bool)

    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx) ** 2 + (Y - cy) ** 2
    mask = dist2 <= radio ** 2
    return mask



def _build_pore_mask(matriz_poros: np.ndarray) -> np.ndarray:
    """Máscara booleana de poros, equivalente a (matriz != "") en analisis.py.

    - Si la matriz es de strings/objetos: considera poro a cualquier valor distinto de "".
    - Si la matriz es numérica: considera fondo al valor MÁS FRECUENTE (mode) y marca poro
      a todo lo distinto a ese valor. Esto evita errores típicos como usar `> 0` cuando
      el fondo es -1 y los IDs empiezan en 0 (o cualquier convención similar).
    """
    if matriz_poros.dtype.kind in {"U", "S", "O"}:
        # analisis.py: fondo = ""
        return matriz_poros != ""
    # numérico: inferir fondo como el valor modal (más frecuente)
    vals, counts = np.unique(matriz_poros, return_counts=True)
    if vals.size == 0:
        return np.zeros_like(matriz_poros, dtype=bool)
    fondo = vals[int(np.argmax(counts))]
    return matriz_poros != fondo


def _normalize_external_ids(external_ids, matriz_poros: np.ndarray):
    """Normaliza IDs exteriores al tipo de la matriz para que np.isin funcione bien."""
    ids = list(external_ids) if external_ids is not None else []
    if not ids:
        return ids
    # Si la matriz es numérica, intentar castear los IDs a int (sin romper si hay strings raros)
    if matriz_poros.dtype.kind in {"i", "u", "f"}:
        norm = []
        for x in ids:
            try:
                # aceptar strings tipo "12"
                norm.append(int(x))
            except Exception:

                continue
        return norm
    return [str(x) for x in ids]

def _calcular_longitud_contornos(mascara_poros: np.ndarray) -> float:
    """
    Estima la longitud total del contorno (perímetro) en unidades de píxel.

    Se extraen contornos sub-píxel con marching squares (skimage.measure.find_contours)
    y se suma la distancia euclídea entre puntos consecutivos (cerrando cada contorno).
    Esto reduce el sesgo de la grilla y da una aproximación más realista.
    """
    mask = mascara_poros.astype(bool)
    if not np.any(mask):
        return 0.0

    # Padding para evitar contornos abiertos si el poro toca el borde de la imagen.
    padded = np.pad(mask.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)

    # Contornos a nivel 0.5 (frontera entre 0 y 1). Coordenadas (fila, col) en float.
    contornos = measure.find_contours(padded, level=0.5)

    perimetro = 0.0
    for c in contornos:
        if c.shape[0] < 2:
            continue
        diffs = np.diff(c, axis=0)
        perimetro += float(np.sqrt((diffs * diffs).sum(axis=1)).sum())
        # cerrar el contorno
        perimetro += float(np.linalg.norm(c[0] - c[-1]))

    return float(perimetro)


def compute_per_image_analysis(
    analysis: AnalysisResult,
    mask_cfg: MaskConfig,
    geom_cfg: GeometryConfig,
) -> List[ImageAnalysisRecord]:
    """
    Replica los cálculos de analisis.py, pero usando:
      - labels_by_image y external_pore_ids del AnalysisResult,
      - MaskConfig para la máscara circular,
      - GeometryConfig para convertir a cm (desde mm).
    """
    labels_by_image = analysis.segmentation.labels_by_image
    external_ids = analysis.external_pore_ids

    # Conversión de mm -> cm según geom_cfg
    pixel_size_mm = geom_cfg.pixel_size_mm
    PIXEL_SIZE_CM = pixel_size_mm / 10.0
    AREA_PIXEL_CM2 = PIXEL_SIZE_CM ** 2
    
    dz_mm = geom_cfg.slice_distance_mm 
    DZ_CM = dz_mm / 10.0

    records: List[ImageAnalysisRecord] = []

    for image_name in sorted(labels_by_image.keys()):
        matriz_poros = np.array(labels_by_image[image_name])
        h, w = matriz_poros.shape

        # Máscara circular y área
        mascara_circular = _build_circular_mask((h, w), mask_cfg)
        area_mascara = int(np.count_nonzero(mascara_circular))

        # --- TODOS LOS POROS ---
        mascara_poros = _build_pore_mask(matriz_poros)
        mascara_poros_en_circulo = mascara_poros & mascara_circular

        pixeles_poro = int(np.count_nonzero(mascara_poros_en_circulo))
        porosidad = pixeles_poro / area_mascara if area_mascara > 0 else 0.0

        area_poros_cm2 = pixeles_poro * AREA_PIXEL_CM2

        longitud_contorno_px = _calcular_longitud_contornos(mascara_poros_en_circulo)
        longitud_contorno_cm = longitud_contorno_px * PIXEL_SIZE_CM
        area_pared_poros_cm2 = longitud_contorno_px * PIXEL_SIZE_CM * DZ_CM

        # --- SOLO POROS CONECTADOS AL EXTERIOR ---
        mascara_ids_exteriores = np.isin(matriz_poros, _normalize_external_ids(external_ids, matriz_poros))
        mascara_poros_ext_en_circulo = (mascara_ids_exteriores & mascara_poros) & mascara_circular

        pixeles_poro_ext = int(np.count_nonzero(mascara_poros_ext_en_circulo))
        porosidad_exterior = pixeles_poro_ext / area_mascara if area_mascara > 0 else 0.0

        longitud_contorno_px_ext = _calcular_longitud_contornos(mascara_poros_ext_en_circulo)
        area_pared_poros_exteriores_cm2 = longitud_contorno_px_ext * PIXEL_SIZE_CM * DZ_CM

        records.append(
            ImageAnalysisRecord(
                image_name=image_name,
                pixeles_poro=pixeles_poro,
                area_mascara_pixeles=area_mascara,
                porosidad=porosidad,
                area_poros_cm2=area_poros_cm2,
                longitud_contorno_px=longitud_contorno_px,
                longitud_contorno_cm=longitud_contorno_cm,
                area_pared_poros_cm2=area_pared_poros_cm2,
                porosidad_exterior=porosidad_exterior,
                area_pared_poros_exteriores_cm2=area_pared_poros_exteriores_cm2,
            )
        )

    return records


def export_per_image_analysis_to_csv(
    records: List[ImageAnalysisRecord],
    output_path: str | Path,
) -> str:
    """
    Exporta los registros a un CSV.
    """
    output_path = str(output_path)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "imagen",
            "pixeles_poro",
            "area_mascara_pixeles",
            "porosidad",
            "area_poros_cm2",
            "longitud_contorno_px",
            "longitud_contorno_cm",
            "area_pared_poros_cm2",
            "porosidad_exterior",
            "area_pared_poros_exteriores_cm2",
        ])

        for r in records:
            writer.writerow([
                r.image_name,
                r.pixeles_poro,
                r.area_mascara_pixeles,
                f"{r.porosidad:.6f}",
                f"{r.area_poros_cm2:.6f}",
                f"{r.longitud_contorno_px:.2f}",
                f"{r.longitud_contorno_cm:.4f}",
                f"{r.area_pared_poros_cm2:.6f}",
                f"{r.porosidad_exterior:.6f}",
                f"{r.area_pared_poros_exteriores_cm2:.6f}",
            ])

    return output_path
