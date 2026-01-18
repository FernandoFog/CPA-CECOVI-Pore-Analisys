# CPA — CECOVI Pore Analysis  
**Open-source tool for pore structure analysis from CT (computed tomography) slices**

> 🇪🇸 **Resumen:** Este repositorio contiene el el codigo en python para procesar cortes 2D de tomografía computarizada (TC), segmentar poros, reconstruir conectividad 3D por solapamiento entre capas, clasificar porosidad **conectada al exterior** vs **interna**, estimar métricas y exportar resultados (CSV / STL).  
> 🇬🇧 **Summary:** Python proyect to segment pores from CT slices, infer 3D connectivity via overlap, classify external vs internal porosity, compute metrics, and export CSV/STL.

---

## ¿Qué hace esta herramienta?
A partir de una secuencia de imágenes (cortes 2D) de una probeta (típicamente cilíndrica), el flujo de trabajo implementa:

1. **Preprocesamiento y segmentación 2D:** recorte (ROI), máscara circular de probeta, filtrado opcional y umbralización para identificar poros como regiones discretas.
2. **Solapamiento inter-capa (k → k+1):** detección de coincidencias píxel a píxel entre poros de cortes consecutivos.
3. **Conectividad 3D basada en grafos:** construcción de un grafo donde nodos = poros 2D (IDs) y aristas = solapamientos.
4. **Clasificación exterior / interior:** identificación de poros conectados al exterior (borde lateral y caras extremas) y poros internos (cerrados).
5. **Componentes internas 3D y volumen:** agrupación de poros internos en componentes 3D y estimación de volumen (mm³).
6. **Exportaciones:**
   - **STL** de poros internos (modelo 3D).
   - **CSV** con métricas 2D por imagen (porosidad, área, contornos, etc.).

---

## Arquitectura del proyecto
El diseño es **modular**.  
Esto facilita extender funcionalidades, integrar nuevos módulos o conectar la herramienta a una GUI sin modificar el núcleo del análisis.

Módulos principales:
- `segmentation.py` — segmentación 2D por corte (labels/IDs/áreas).
- `overlaps.py` — solapamiento entre cortes consecutivos.
- `graph_3d.py` — construcción del grafo de conectividad.
- `pores3d.py` — clasificación exterior/interior, volumen 3D, STL.
- `analysis_2d.py` — métricas 2D por corte + exportación CSV.
- `pipeline.py` — orquestación (función de alto nivel).

<p align="center">
  <img width="720" height="695" alt="image" src="https://github.com/user-attachments/assets/7aec69a4-c786-457b-9ae5-de3157bf039d" />
</p>

---

## Requisitos
- **Python 3.9+** (recomendado 3.10+)

Dependencias principales:
- `numpy`
- `opencv-python`
- `scikit-image`
- `networkx`


