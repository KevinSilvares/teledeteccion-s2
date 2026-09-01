# Sentinel-2 Road Detection: Super-Resolución y Segmentación Semántica

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Transformers-SegFormer-FFD21E?logo=huggingface&logoColor=black)
![GIS](https://img.shields.io/badge/GIS-GDAL%20%2F%20Rasterio-347434)

> 🇪🇸 Esta es la versión en español del README. [Read in English](/README.md)

#### ¡Prueba la demo!
[![Abrir Demo en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l7XRxP14V5jfWV47x9xbktujC_N4kJeH?usp=sharing)

## Problema e Impacto
La cartografía de vías rurales y cortafuegos en zonas densamente forestadas es un desafío vital para la prevención de incendios. Este proyecto automatiza la extracción de estas geometrías procesando datos de la API de Copernicus, mejorando su resolución nativa (10 m/px a 2.5 m/px) y aplicando visión artificial.

> **Nota:** Aunque el programa PNOA ofrece resoluciones altísimas (0.25-0.5 m/px), los vuelos se realizan cada 3 o más años. Sentinel-2 ofrece una captura casi en tiempo real, vital para la actualización logística.

## Arquitectura y Stack Tecnológico
- **Lenguajes y Frameworks:** Python, PyTorch, Transformers (Hugging-Face)
- **Modelos Base:** [Sen2SR](https://github.com/ESAOpenSR/SEN2SR) (Super-Resolución) y [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) [mit-b3](https://huggingface.co/nvidia/mit-b3) (Segmentación)
- **Datos y GIS:** Imágenes Sentinel-2, Ortofotos [PNOA](https://pnoa.ign.es/), QGIS, Rasterio, Xarray, GDAL

## Retos Técnicos Resueltos
- **Adaptación de Arquitectura de SegFormer a 4 canales:** Se modificó la capa de entrada original del modelo Segformer mit-b3 (diseñada originalmente para RGB) con PyTorch para procesar una cuarta banda (Infrarrojo cercano - NIR), crucial para el análisis de vegetación.
- **Descarga masiva y adaptación de los rásters (resolución de 10 m/px a 2.5 m/px, conversión de 10 bandas a 4 bandas):** Se ha logrado descargar imágenes de forma masiva para entrenar un modelo Sen2SR con la imágenes capturas por el PNOA (originalmente a 0.25-0.5 m/px) para lograr una resolución a 2.5 m/px, usando sólo las bandas R, G, B y NIR del Sentinel-2 (originalmente a 10 m/px).
- **Descarga masiva y manejo de errores de imágenes satelitales Sentinel-2:** Se ha logrado descargar de forma masiva imágenes del Sentinel-2 mediante *POIs (Points Of Interest -* Puntos de Interés) para su posterior Super-Resolución manejando errores y gestionando auto autenticaciones con la API de Copernicus, así como el procesamiento automático de sus respectivos sistemas de coordenadas (CRS)
- **Super-Resolución masiva:** Implementación de scripts para inferir de forma masiva los datos anteriormente descargados.
- **Manejo de Errores Geoespaciales:** Manejo de recortes dinámicos en memoria para solucionar desajustes de matrices (”*Off-By-One pixel errors*”) generados por el redondeo de QGIS (GDAL) en las máscaras.
- **Tiling y Normalización:** Scripting automatizado para dividir grandes TIFs de 2048x2048 en parches de 512x512, aplicando recortes de percentiles (2º-98º) consistentes tanto en entrenamiento como en inferencia.

## Resultados Preliminares (v0.1)
El modelo identifica formas alargadas y sinuosas típicas de los caminos, demostrando que la base geométrica ha sido aprendida. También ha desarrollado alta sensibilidad al suelo desnudo (útil para detectar cortafuegos).

![Imagen Super-Resuelta a 2.5 m/px cerca de Villablino, León. Coords: 42°57'20.9"N 6°24'04.7"W](./docs/imgs/img_og_2.png)

(Imagen Super-Resuelta a 2.5 m/px cerca de Villablino, León. Coords: 42°57'20.9"N 6°24'04.7"W)

![(Máscara del modelo Preliminar v0.1)](./docs/imgs/img_mask_2.png)

(Máscara del modelo Preliminar v0.1)

![Imagen Super-Resuelta a 2.5 m/px cerca de Villablino, León. Coords: 42°55'14.6"N 6°16'55.8"W](./docs/imgs/img_og_1.png)

Imagen Super-Resuelta a 2.5 m/px cerca de Villablino, León. Coords: 42°55'14.6"N 6°16'55.8"W

![(Máscara del modelo Preliminar v0.1)](./docs/imgs/img_mask_1.png)

(Máscara del modelo Preliminar v0.1)

## Roadmap
- [ ] **Hard Negative Mining:** Inclusión de parches con grandes claros de tierra sin caminos para evitar falsos positivos.
- [ ] **Optimización Topológica:** Mejora de la función de pérdida (implementación de *clDice* o *TopoLoss*) para penalizar fuertemente la fragmentación de la línea del camino.
- [x] Despliegue de demo interactiva.
- [ ] Guía de reproducibilidad.
