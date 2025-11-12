# 🚗 RAGberto — Agente Híbrido de Seguridad Vehicular NHTSA

[![Streamlit App](https://img.shields.io/badge/🚀_Ver_Demo_en_Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://nhtsa-rag-demo-9mxdsrhlghsz68kvgs5t7l.streamlit.app/)

---

## 🧭 Descripción general

**RAGberto** es un agente híbrido de búsqueda y razonamiento (**RAG + Grafo + LLM**) desarrollado para explorar y vincular información de seguridad vehicular de la **NHTSA**.  
Combina **búsqueda semántica (Qdrant)**, **análisis relacional (Neo4j)** y **razonamiento lingüístico (Groq LLM)** para ofrecer respuestas expertas a partir de quejas de usuarios, recalls técnicos e investigaciones.

El agente puede:
- Analizar una descripción libre de una falla vehicular.
- Recuperar quejas similares y sus *recalls* asociados mediante relaciones explícitas en el grafo.
- Si no hay vínculos directos, realizar una búsqueda **semántica** ampliada en corpus técnicos.
- Generar un informe experto en español, incluyendo una **visualización de causa raíz**.

---

## ⚙️ Arquitectura

```mermaid
flowchart LR
    A["🧠 Consulta del usuario"] --> B["💬 Vectorización (E5-Large-Instruct)"]
    B --> C["🔍 Búsqueda en Qdrant"]
    C --> D["🧩 Vínculos Neo4j ([:ASSOCIATED_RECALL])"]
    D --> E["🧠 Análisis con LLM (Groq - Llama 3.3 70B)"]
    E --> F["📊 Resumen técnico + Visualización (Graphviz)"]
````

### Componentes principales

| Módulo                         | Descripción                                                                                                                         |
| :----------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| **Streamlit UI**               | Interfaz web para interacción con el usuario.                                                                                       |
| **Qdrant**                     | Almacén vectorial para `Complaints`, `Recalls` e `Investigations`.                                                                  |
| **Neo4j**                      | Grafo de relaciones `(:Complaint)-[:ASSOCIATED_RECALL]->(:Recall)`.                                                                 |
| **Hugging Face Inference API** | Modelo [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct) para embeddings. |
| **Groq LLM API**               | Modelo [`llama-3.3-70b-versatile`](https://console.groq.com) para razonamiento y respuesta.                                         |

---

## 🧪 Corpus vectorizados

Los tres dominios fueron preprocesados y vectorizados:

| Corpus              |                                   Entradas |       Dimensiones | Modelo              | Artefacto principal                |
| :------------------ | -----------------------------------------: | ----------------: | :------------------ | :--------------------------------- |
| **Recalls**         |                              12,871 chunks |              1024 | `E5-Large-Instruct` | `recalls_embeddings.npy`           |
| **Investigations**  |                               6,354 chunks |              1024 | `E5-Large-Instruct` | `invest_embeddings.npy`            |
| **Complaints (CV)** | 142,399 vectores (→ 19,474 representantes) | 1024 → 256 (IPCA) | `E5-Large-Instruct` | `representantes_GOLDEN_kmeans.csv` |

---

## 🧠 Flujo operativo del agente

### 🔹 Flujo 1 — Vinculado (Grafo)

Busca quejas similares en Qdrant y recupera sus *recalls* directamente conectados en Neo4j.

### 🔹 Flujo 2 — Semántico (Fallback)

Si no hay vínculos explícitos, busca *recalls* o *investigations* semánticamente cercanos.

### 🔹 Generación de respuesta

El modelo LLM (Groq / Llama 3.3 70B) sintetiza una respuesta técnica clara y empática, firmada por **RAGberto**, incluyendo posibles causas y recomendaciones.

---

## 🧰 Ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu_usuario>/<tu_repo>.git
cd <tu_repo>

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar credenciales (en .streamlit/secrets.toml)
[secrets]
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "tu_password"
QDRANT_URL = "http://localhost:6333"
QDRANT_KEY = "tu_api_key"
GROQ_API_KEY = "tu_api_key"
HF_TOKEN = "tu_token_hf"

# 4. Ejecutar la app
streamlit run app.py
```

---

## 📈 Resultados

| Evaluación                      | nDCG@10 | MRR@10 | Recall@10 |
| ------------------------------- | ------- | ------ | --------- |
| **R2R (Recalls-to-Recalls)**    | 0.6451  | 0.6310 | 0.8540    |
| **C2R (Complaints-to-Recalls)** | 0.3024  | 0.2589 | 0.4750    |

El sistema mantiene una cobertura semántica excelente y un rendimiento en tiempo real (~3.8 s promedio por consulta).

---

## 🧩 Visualización

<p align="center">
  <img src="docs/fig_agent_ui.png" width="80%">
</p>

<p align="center">
  <img src="docs/fig_agent_result.png" width="80%">
</p>

---

## 🌐 Demo pública

➡️ **Probar ahora:** [https://nhtsa-rag-demo-9mxdsrhlghsz68kvgs5t7l.streamlit.app](https://nhtsa-rag-demo-9mxdsrhlghsz68kvgs5t7l.streamlit.app/)

---

## 🧾 Licencia

Este proyecto se distribuye bajo la licencia **MIT**.
© 2025 Lucero Contreras Hernández
