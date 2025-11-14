# 📰 Task 1: Sistema de Recuperación de Noticias RPP

## 🎯 Objetivo

Construir un sistema end-to-end de recuperación de noticias desde el RSS feed de RPP Perú, utilizando embeddings semánticos, ChromaDB para almacenamiento vectorial y LangChain para orquestación modular.

## 📋 Descripción

Este proyecto implementa un pipeline completo de NLP que:

1. **Extrae** 50 noticias del feed RSS de RPP Perú
2. **Tokeniza** el contenido usando tiktoken
3. **Genera embeddings** con SentenceTransformers (all-MiniLM-L6-v2)
4. **Almacena** en ChromaDB con persistencia
5. **Recupera** noticias relevantes por similitud semántica
6. **Orquesta** todo con LangChain

## 🏗️ Arquitectura

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  RSS Feed   │────▶│ Tokenización │────▶│  Embeddings │────▶│   ChromaDB   │
│ (RPP Perú)  │     │  (tiktoken)  │     │  (MiniLM)   │     │ (Persistent) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
                                                              ┌──────────────┐
                                                              │  LangChain   │
                                                              │  Retriever   │
                                                              └──────────────┘
                                                                      │
                                                                      ▼
                                                              ┌──────────────┐
                                                              │   Resultados │
                                                              │  (DataFrame) │
                                                              └──────────────┘
```

## 📁 Estructura del Proyecto

```
Task1_news-query_RPP-lab/
│
├── news_retrieval_rpp.ipynb    # Notebook principal con todo el pipeline
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Este archivo
│
├── data/                        # Datos crudos y procesados
│   ├── rss_raw.json            # Noticias originales del RSS
│   ├── rss_raw.csv             # Noticias en formato CSV
│   ├── processed_news.json     # Metadatos procesados
│   └── embeddings.npy          # Embeddings en formato numpy
│
├── output/                      # Resultados y análisis
│   ├── token_analysis.json     # Estadísticas de tokenización
│   ├── query_results_*.csv     # Resultados de búsquedas
│   └── langchain_query_results.csv
│
└── chroma_db/                   # Base de datos vectorial (generada)
    └── [ChromaDB persistent storage]
```

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.10 o superior
- Google Colab (recomendado) o entorno local con Jupyter

### Paso 1: Clonar o Descargar

```bash
# Si usas Git
git clone <repo-url>
cd Task1_news-query_RPP-lab

# O simplemente descarga el archivo news_retrieval_rpp.ipynb
```

### Paso 2: Instalación de Dependencias

**En Google Colab:**
```python
# Las dependencias se instalan automáticamente en la primera celda del notebook
!pip install feedparser tiktoken sentence-transformers chromadb langchain langchain-community pandas numpy -q
```

**En entorno local:**
```bash
pip install -r requirements.txt
```

### Paso 3: Ejecutar el Notebook

1. Abre `news_retrieval_rpp.ipynb` en Google Colab o Jupyter
2. Ejecuta las celdas secuencialmente (⚠️ **IMPORTANTE**: ejecutar en orden)
3. El notebook creará automáticamente las carpetas `data/`, `output/` y `chroma_db/`

## 📊 Componentes del Pipeline

### 0️⃣ Instalación e Imports
- Instalación silenciosa de todas las librerías
- Importación de módulos necesarios
- Creación de estructura de carpetas

### 1️⃣ Carga de Datos (RSS Feed)
- Extracción de 50 noticias desde `https://rpp.pe/rss`
- Campos: `title`, `description`, `link`, `published`
- Guardado en `data/rss_raw.json` y `data/rss_raw.csv`

### 2️⃣ Tokenización (tiktoken)
- Tokenización con encoding `cl100k_base`
- Análisis estadístico de conteo de tokens
- **Conclusión**: Artículos cortos (~68 tokens promedio), no requieren chunking
- Guardado en `output/token_analysis.json`

### 3️⃣ Generación de Embeddings
- Modelo: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensión: 384
- Formato: título + descripción
- Guardado en `data/embeddings.npy`

### 4️⃣ Almacenamiento en ChromaDB
- Cliente persistente en `./chroma_db/`
- Colección: `rpp_news`
- Operación: Upsert (inserta o actualiza)
- 50 documentos con metadatos completos

### 5️⃣ Sistema de Búsqueda
- Búsqueda por similitud semántica
- Top 10 resultados por query
- Ejemplos implementados:
  - "Últimas noticias de economía"
  - "Noticias sobre política y gobierno"
  - "Noticias de música y conciertos"

### 6️⃣ Orquestación con LangChain
- Pipeline modular y extensible
- `HuggingFaceEmbeddings` para embeddings
- `Chroma` VectorStore con persistencia
- Retriever configurado para similarity search

## 🔍 Ejemplos de Uso

### Búsqueda Básica

```python
# Realizar una búsqueda
results = search_news("Últimas noticias de economía", n_results=10)

# Ver resultados
print(results[['title', 'date_published', 'similarity_score']])
```
