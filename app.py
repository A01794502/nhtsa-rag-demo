# ===============================================
#          app.py (Híbrido FINAL v4 - API de HF)
# ===============================================
# CAMBIOS:
# 1. Se eliminó 'torch' y 'sentence-transformers'.
# 2. Se eliminó 'get_encoder()'.
# 3. 'e5_query()' ahora usa la API de Hugging Face.
# 4. Se añadió 'HF_TOKEN' a los secrets.
# ===============================================

import streamlit as st
import os
import numpy as np
import json
import requests  # Para llamar a la API
import time      # Para los reintentos
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Dict, Any, List

# --- 1. CONFIGURACIÓN DE CREDENCIALES ---
NEO4J_URI = st.secrets.get("NEO4J_URI", os.getenv("NEO4J_URI"))
NEO4J_USER = st.secrets.get("NEO4J_USER", os.getenv("NEO4J_USER"))
NEO4J_PASS = st.secrets.get("NEO4J_PASS", os.getenv("NEO4J_PASS"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_KEY = st.secrets.get("QDRANT_KEY", os.getenv("QDRANT_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# ¡NUEVA CLAVE!
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

# Endpoint de la API gratuita de Hugging Face
API_URL_E5 = "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-large-instruct"
HEADERS_E5 = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- 2. CLIENTES Y MODELOS (@st.cache_resource) ---
@st.cache_resource
def get_neo_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()
    print("Conectado a Neo4j.")
    return driver

@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=180)
    print("Conectado a Qdrant.")
    return client

# ¡YA NO NECESITAMOS get_encoder()!

@st.cache_resource
def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY no está configurada en st.secrets")
    return ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=2048,
    )

# --- 3. FUNCIONES DE BÚSQUEDA (Comunes) ---

# ¡FUNCIÓN ACTUALIZADA!
@st.cache_data  # Cacheamos la *salida* de esta función
def e5_query(text: str) -> np.ndarray:
    """
    Llama a la API de Hugging Face para obtener el vector.
    Ya no carga el modelo en la RAM de Streamlit.
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN no fue encontrado en st.secrets.")

    payload = {
        "inputs": f"query: {text}",
        "options": {"wait_for_model": True} # Importante si el modelo está "frío"
    }

    # Reintentar 3 veces si el modelo está cargando (503)
    for i in range(3):
        response = requests.post(API_URL_E5, headers=HEADERS_E5, json=payload)
        
        if response.status_code == 200:
            vector = response.json()
            if isinstance(vector, list) and len(vector) > 0:
                # La API devuelve [[vector]], así que tomamos el primer elemento
                return np.asarray(vector[0], dtype=np.float32)
            else:
                raise ValueError(f"Hugging Face API payload inesperado: {vector}")

        elif response.status_code == 503: # Modelo está cargando
            print(f"Modelo E5 está cargando (intento {i+1}/3), reintentando en 5s...")
            time.sleep(5)
        else:
            # Otro error (ej. 401 Unauthorized si el token es malo)
            raise RuntimeError(
                f"Error en API de Hugging Face: {response.status_code}\n"
                f"Respuesta: {response.text}"
            )
    
    raise RuntimeError("Falló la obtención del vector de Hugging Face después de 3 reintentos.")


def cypher(query: str, params: Optional[Dict[str, Any]] = None):
    with get_neo_driver().session() as s:
        return s.run(query, **(params or {})).data()

def qdrant_search(qv, k=15, mmy_filter=None, collection=None):
    client = get_qdrant_client()
    qdrant_filter = None
    
    if mmy_filter:
        conditions = []
        if mmy_filter.get("make"):
            conditions.append(models.FieldCondition(key="make", match=models.MatchValue(value=mmy_filter["make"])))
        if mmy_filter.get("model"):
            conditions.append(models.FieldCondition(key="model", match=models.MatchValue(value=mmy_filter["model"])))
        if mmy_filter.get("year"):
            conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=mmy_filter["year"])))
        
        if conditions:
            qdrant_filter = models.Filter(must=conditions)
    
    qv_list = qv.tolist()
    
    try:
        resp = client.query_points(
            collection_name=collection,
            query=qv_list,
            limit=k,
            with_payload=True,
            with_vectors=False,
            query_filter=qdrant_filter
        )
    except TypeError:
        resp = client.query_points(
            collection_name=collection,
            query=qv_list,
            limit=k,
            with_payload=True,
            with_vectors=False,
            filter=qdrant_filter
        )
    
    points = getattr(resp, "points", resp)
    out = []
    for h in points:
        payload = dict(h.payload) if getattr(h, "payload", None) else {}
        out.append({"id": str(h.id), "score": float(h.score), "payload": payload})
    return out

def qdrant_retrieve_by_ids(ids: list, collection_name: str) -> dict:
    client = get_qdrant_client()
    qdrant_filter = models.Filter(
        should=[
            models.FieldCondition(key="id", match=models.MatchValue(value=recall_id))
            for recall_id in ids
        ]
    )
    try:
        resp, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=len(ids),
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        print(f"Error en qdrant_retrieve_by_ids: {e}")
        return {}
    
    text_map = {}
    for point in resp:
        payload = dict(point.payload)
        recall_id = payload.get("id")
        full_text = payload.get("text")
        if recall_id and full_text:
            text_map[recall_id] = full_text
    return text_map

# --- 4. PIPELINE DE DATOS - FLUJO 1 (GOLDEN) ---
def run_golden_flow(question: str, mmy_filter: Optional[Dict[str, Any]] = None, k: int = 25):
    try:
        qv = e5_query(question)
    except Exception as e:
        return {"error": f"Falla al codificar texto: {e}"}

    try:
        hits = qdrant_search(qv, k=k, collection="nhtsa_complaints")
    except Exception as e:
        return {"error": f"Falla en búsqueda de Qdrant(Complaints): {e}"}

    ids = [str(h.get("payload", {}).get("id")) for h in hits if h.get("payload", {}).get("id")]
    if not ids:
        return {"query": question, "hits": hits, "results": []}

    cypher_params = {
        "ids": ids,
        "make": mmy_filter.get("make"),
        "model": mmy_filter.get("model"),
        "year": mmy_filter.get("year")
    }

    query = """
    UNWIND $ids AS cid
    MATCH (c:Complaint {comp_id: cid})
    MATCH (c)-[rel:ASSOCIATED_RECALL]->(r:Recall)
    WHERE ($make IS NULL OR r.make = $make)
      AND ($model IS NULL OR r.model = $model)
      AND ($year IS NULL OR r.year = $year)
    RETURN 
      c.comp_id AS complaint_comp_id,
      r.id AS recall_id,
      r.subject AS recall_summary,
      r.consequence AS recall_consequence,
      r.component AS recall_component_text
    ORDER BY c.comp_id
    """
    neo4j_details = cypher(query, cypher_params)
    
    if not neo4j_details:
         return {"query": question, "hits": hits, "results": []}

    recall_ids_encontrados = list(set(r['recall_id'] for r in neo4j_details if r['recall_id']))
    mapa_texto_completo = qdrant_retrieve_by_ids(recall_ids_encontrados, "nhtsa_recalls")
    
    results_by_complaint = {}
    for hit in hits:
        p = hit.get("payload", {})
        comp_id = str(p.get("id"))
        if comp_id:
            results_by_complaint[comp_id] = {
                "complaint_hit": hit,
                "actionable_recalls": [],
                "historical_links": []
            }
    
    TEXT_NOT_FOUND_MSG = "--- Texto no encontrado en Qdrant(Recalls) ---"
    
    for record in neo4j_details:
        comp_id = record.get("complaint_comp_id")
        recall_id = record.get("recall_id")
        full_text = mapa_texto_completo.get(recall_id, TEXT_NOT_FOUND_MSG)
        record['full_text_from_qdrant'] = full_text
        
        if comp_id in results_by_complaint:
            if full_text == TEXT_NOT_FOUND_MSG:
                results_by_complaint[comp_id]["historical_links"].append(record)
            else:
                results_by_complaint[comp_id]["actionable_recalls"].append(record)

    final_results = [
        res for res in results_by_complaint.values() 
        if res["actionable_recalls"] or res["historical_links"]
    ]
    return {"query": question, "results": final_results}

# --- 5. PIPELINE DE DATOS - FLUJO 2 (SEMÁNTICO) ---
def run_semantic_flow(question: str, mmy_filter: Optional[Dict[str, Any]] = None, k: int = 5):
    try:
        qv = e5_query(question)
    except Exception as e:
        return {"error": f"Falla al codificar texto: {e}"}

    try:
        recall_hits = qdrant_search(qv, k=k, mmy_filter=mmy_filter, collection="nhtsa_recalls")
        investigation_hits = qdrant_search(qv, k=k, mmy_filter=mmy_filter, collection="nhtsa_investigations")
    except Exception as e:
        return {"error": f"Falla en búsqueda de Qdrant: {e}"}

    return {
        "query": question,
        "recall_hits": recall_hits,
        "investigation_hits": investigation_hits
    }

# --- 6. LÓGICA DEL AGENTE LLM ---
# (Prompts y funciones de formato sin cambios)
GOLDEN_FLOW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
Eres un asistente experto en seguridad vehicular de la NHTSA.
Tu trabajo es analizar la queja de un usuario y compararla con el CONTEXTO TÉCNICO VINCULADO (basado en un grafo).

El contexto se divide en dos partes:
1.  **RECALLS ACCIONABLES:** Estos son recalls con texto completo que están *directamente vinculados* a quejas similares a la del usuario Y COINCIDEN CON SU VEHÍCULO.
2.  **VÍNCULOS HISTÓRICOS:** Estos son IDs de recalls más antiguos que también están *vinculados* a esta queja.

QUEJA DEL USUARIO:
{question}

CONTEXTO TÉCNICO RECUPERADO (VÍA GRAFO):
{context}

INSTRUCCIONES PARA TU RESPUESTA:
1.  **Analiza** los "RECALLS ACCIONABLES". Resume los hallazgos más relevantes.
2.  **Menciona** los "VÍNCULOS HISTÓRICOS" si existen.
3.  **Concluye** con un resumen profesional. Responde en español. Si el contexto está vacío, dilo.
""")
])

SEMANTIC_FLOW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
Eres un asistente experto en seguridad vehicular de la NHTSA.
No se encontró ningún recall *directamente vinculado* a la queja del usuario en el grafo de seguridad.
Tu trabajo ahora es realizar una **búsqueda semántica** general para encontrar documentos (recalls o investigaciones) que *hablen de temas similares* Y COINCIDAN CON SU VEHÍCULO.

QUEJA DEL USUARIO:
{question}

CONTEXTO SEMÁNTICO RECUPERADO (búsqueda por similitud de texto):
{context}

INSTRUCCIONES PARA TU RESPUESTA:
1.  **Informa** al usuario que no se encontraron recalls *directamente vinculados* a su queja.
2.  **PERO**, informa que has encontrado los siguientes recalls o investigaciones que son *semánticamente similares* a su descripción.
3.  **Resume** los 2-3 hallazgos más relevantes del contexto semántico.
4.  **Concluye** con un resumen profesional. Responde en español.
""")
])

def format_golden_context(rag_results: dict) -> str:
    if not rag_results.get("results"):
        return "No se encontraron quejas similares ni recalls asociados en la base de datos."
    context_parts = []
    for i, res in enumerate(rag_results["results"], 1):
        comp_id = res['complaint_hit']['payload'].get('id')
        score = res['complaint_hit']['score']
        context_parts.append(f"\n--- ANÁLISIS DEL HIT DE QUEJA #{i} (ID: {comp_id}, Similitud: {score:.2f}) ---")
        if res['actionable_recalls']:
            context_parts.append("\n**RECALLS ACCIONABLES (Con texto completo):**")
            for recall in res['actionable_recalls']:
                context_parts.append(f"  - ID Recall: {recall['recall_id']}")
                context_parts.append(f"    Defecto (Resumen): {recall['recall_summary']}")
                context_parts.append(f"    Riesgo (Consecuencia): {recall['recall_consequence']}")
                context_parts.append(f"    Texto Completo: {recall['full_text_from_qdrant'][:500]}...")
        if res['historical_links']:
            context_parts.append("\n**VÍNCULOS HISTÓRICOS (Sin texto, solo ID):**")
            ids = [r['recall_id'] for r in res['historical_links']]
            context_parts.append(f"  - IDs vinculados: {', '.join(ids)}")
    return "\n".join(context_parts)

def format_semantic_context(rag_results: dict) -> str:
    context_parts = []
    if rag_results.get("recall_hits"):
        context_parts.append("\n**Recalls Semánticamente Similares:**")
        for hit in rag_results["recall_hits"]:
            payload = hit['payload']
            context_parts.append(f"  - ID Recall: {payload.get('id')}")
            context_parts.append(f"    Texto: {payload.get('text', 'N/A')[:500]}...")
    if rag_results.get("investigation_hits"):
        context_parts.append("\n**Investigaciones Semánticamente Similares:**")
        for hit in rag_results["investigation_hits"]:
            payload = hit['payload']
            context_parts.append(f"  - ID Investigación: {payload.get('id')}")
            context_parts.append(f"    Texto: {payload.get('text', 'N/A')[:500]}...")
    if not context_parts:
        return "No se encontraron recalls o investigaciones semánticamente similares para este vehículo."
    return "\n".join(context_parts)

def get_llm_summary(user_query: str, prompt_template: ChatPromptTemplate, context_str: str):
    llm = get_llm()
    chain = prompt_template | llm | StrOutputParser()
    return chain.stream({
        "context": context_str,
        "question": user_query
    })

# --- 7. APLICACIÓN STREAMLIT (UI HÍBRIDA MEJORADA) ---

st.set_page_config(page_title="Agente NHTSA RAG", layout="wide")
st.title("🚗 Agente Híbrido de Búsqueda NHTSA")
st.markdown("""
Este demo implementa un pipeline **Híbrido (Grafo + Semántico)** con **Filtros de Entidad**.
1.  **Intento 1 (Grafo):** Busca quejas similares (Qdrant) y atraviesa el grafo (Neo4j) para encontrar recalls *directamente vinculados* que coincidan con el vehículo.
2.  **Intento 2 (Semántico):** Si el paso 1 falla, ejecuta una búsqueda semántica *filtrada* en Recalls/Investigaciones.
3.  **Agente:** Un LLM (Groq/Llama 3) analiza y resume los hallazgos.
""")

st.subheader("Información del Vehículo (Opcional, pero recomendado)")
col1, col2, col3 = st.columns(3)
with col1:
    make = st.text_input("Make (Marca)", placeholder="Ej: Toyota")
with col2:
    model = st.text_input("Model (Modelo)", placeholder="Ej: Corolla")
with col3:
    year = st.number_input("Year (Año)", min_value=1950, max_value=2025, value=None, placeholder="Ej: 2014")

st.subheader("Descripción de la Falla")
user_query = st.text_area(
    "Describe la falla de tu vehículo:", 
    placeholder="Ej: Check engine light flashing and vehicle is shaking...",
    height=150
)

if st.button("Analizar Falla", type="primary"):
    if not user_query:
        st.error("Por favor, introduce una descripción de la falla.")
    else:
        try:
            mmy_filter = {}
            if make: mmy_filter["make"] = make.upper().strip()
            if model: mmy_filter["model"] = model.upper().strip()
            if year: mmy_filter["year"] = int(year)
            
            rag_results = None
            llm_prompt = None
            context_str = None
            
            with st.spinner("Paso 1/3: Buscando vínculos en el Grafo (Flujo 1)..."):
                golden_results = run_golden_flow(user_query, mmy_filter=mmy_filter)
            
            if golden_results.get("error"):
                st.error(f"Error en Flujo 1: {golden_results['error']}")
                st.stop()

            actionable_recalls_found = any(
                res.get("actionable_recalls") for res in golden_results.get("results", [])
            )

            if actionable_recalls_found:
                st.success("Paso 1 Exitoso: Se encontraron recalls vinculados en el grafo.")
                rag_results = golden_results
                llm_prompt = GOLDEN_FLOW_PROMPT
                context_str = format_golden_context(rag_results)
            
            else:
                st.warning("Paso 1: No se encontraron recalls *con texto* en el grafo.")
                st.info("Ejecutando Paso 1.5: Búsqueda semántica directa (Flujo 2)...")
                
                with st.spinner("Paso 1.5: Buscando recalls/investigaciones similares..."):
                    semantic_results = run_semantic_flow(user_query, mmy_filter=mmy_filter)
                
                if semantic_results.get("error"):
                    st.error(f"Error en Flujo 2: {semantic_results['error']}")
                    st.stop()
                
                rag_results = semantic_results
                llm_prompt = SEMANTIC_FLOW_PROMPT
                context_str = format_semantic_context(rag_results)

            st.success("Paso 2/3: Datos recuperados. Pasando al LLM de Groq para análisis...")
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = st.write_stream(
                    get_llm_summary(user_query, llm_prompt, context_str)
                )
            
            st.success("Paso 3/3: Análisis completado.")

            with st.expander("Ver datos crudos recuperados (JSON)"):
                st.json(rag_results)

        except Exception as e:
            st.error(f"Ha ocurrido un error fatal durante el análisis: {e}")
            import traceback
            st.error(traceback.format_exc())
