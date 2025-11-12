# ===============================================
#          app.py (Híbrido FINAL v5.1 - Lazy Import)
# ===============================================
# CAMBIOS:
# 1. ELIMINADO 'import graphviz' de la parte superior.
# 2. AÑADIDO 'import graphviz' DENTRO de la función
#    'build_component_graph()' para que solo se llame
#    después de hacer clic en el botón.
# ===============================================

import streamlit as st
import os
os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin' # Dejamos esto, es correcto
import numpy as np
import json
import requests
import time
# import graphviz # <--- ¡LO HEMOS QUITADO DE AQUÍ!
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Dict, Any, List

# --- 1. CONFIGURACIÓN DE CREDENCIALES ---
# (Todo el código de get_neo_driver, get_qdrant_client, get_llm, e5_query, etc.
#  es idéntico al anterior... lo incluyo todo para que sea un solo copiado y pegado)

NEO4J_URI = st.secrets.get("NEO4J_URI", os.getenv("NEO4J_URI"))
NEO4J_USER = st.secrets.get("NEO4J_USER", os.getenv("NEO4J_USER"))
NEO4J_PASS = st.secrets.get("NEO4J_PASS", os.getenv("NEO4J_PASS"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_KEY = st.secrets.get("QDRANT_KEY", os.getenv("QDRANT_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

API_URL_E5 = "https://router.huggingface.co/hf-inference/models/intfloat/multilingual-e5-large-instruct"
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

def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY no está configurada en st.secrets")
    print("Creando NUEVA instancia de ChatGroq (stateless)...")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=2048,
    )

# --- 3. FUNCIONES DE BÚSQUEDA (Comunes) ---
@st.cache_data
def e5_query(text: str) -> np.ndarray:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN no fue encontrado en st.secrets.")
    payload = {"inputs": f"query: {text}", "options": {"wait_for_model": True}}
    for i in range(3):
        response = requests.post(API_URL_E5, headers=HEADERS_E5, json=payload)
        if response.status_code == 200:
            vector_data = response.json()
            v = np.asarray(vector_data, dtype=np.float32)
            if v.ndim == 2 and v.shape[0] == 1: v = v.flatten() 
            if v.ndim == 1 and v.shape[0] > 1: return v 
            raise ValueError(f"Hugging Face API payload inesperado. Shape: {v.shape}. JSON: {vector_data}")
        elif response.status_code == 503: 
            print(f"Modelo E5 está cargando (intento {i+1}/3), reintentando en 5s...")
            time.sleep(5)
        else:
            raise RuntimeError(f"Error en API de HF: {response.status_code}\n{response.text}")
    raise RuntimeError("Falló la obtención del vector de HF después de 3 reintentos.")

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
            collection_name=collection, query=qv_list, limit=k,
            with_payload=True, with_vectors=False, query_filter=qdrant_filter
        )
    except TypeError:
        resp = client.query_points(
            collection_name=collection, query=qv_list, limit=k,
            with_payload=True, with_vectors=False, filter=qdrant_filter
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
        should=[ models.FieldCondition(key="id", match=models.MatchValue(value=recall_id)) for recall_id in ids ]
    )
    try:
        resp, _ = client.scroll(
            collection_name=collection_name, scroll_filter=qdrant_filter,
            limit=len(ids), with_payload=True, with_vectors=False
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
      r.component AS recall_component_text,
      r.make AS recall_make,
      r.model AS recall_model,
      r.year AS recall_year
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

# --- 6. LÓGICA DEL AGENTE LLM Y GRAFO ---
GOLDEN_FLOW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
Eres un asistente experto en seguridad vehicular de la NHTSA.
Tu trabajo es analizar la queja de un usuario y compararla con el CONTEXTO TÉCNICO VINCULADO.
QUEJA DEL USUARIO:
{question}
CONTEXTO TÉCNICO RECUPERADO (VÍA GRAFO):
{context}
INSTRUCCIONES PARA TU RESPUESTA:
1.  **Analiza** la "QUEJA DEL USUARIO" para identificar el vehículo (Make, Model, Year).
2.  **Revisa** los "RECALLS ACCIONABLES". Compara cada recall con el vehículo del usuario.
3.  **Resume ÚNICAMENTE** los recalls que sean *altamente relevantes* para el vehículo y el problema del usuario.
4.  **Si NINGÚN recall accionable es relevante** (ej. son de marcas o modelos diferentes), debes informarlo. Di: "No se encontraron recalls vinculados que coincidan con su vehículo. Los recalls encontrados (IDs: [...]) se refieren a otros modelos."
5.  **Menciona** los "VÍNCULOS HISTÓRICOS" si existen, como evidencia de patrones.
6.  **Concluye** con un resumen profesional. Responde en español.
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
                context_parts.append(f"    Vehículo del Recall: {recall.get('recall_make')} {recall.get('recall_model')} {recall.get('recall_year')}") 
                context_parts.append(f"    Componente: {recall.get('recall_component_text')}")
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
            context_parts.append(f"    Componente: {payload.get('component')}")
            context_parts.append(f"    Texto: {payload.get('text', 'N/A')[:500]}...")
    if rag_results.get("investigation_hits"):
        context_parts.append("\n**Investigaciones Semánticamente Similares:**")
        for hit in rag_results["investigation_hits"]:
            payload = hit['payload']
            context_parts.append(f"  - ID Investigación: {payload.get('id')}")
            context_parts.append(f"    Componente: {payload.get('component')}")
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

# ===============================================================
# ¡NUEVA FUNCIÓN! PARA EL GRAFO (Estilo Dark Mode CORREGIDO)
# ===============================================================
@st.cache_data # Cacheamos el resultado de esta consulta
def build_component_graph(component_strings_list: List[str]) -> Optional[str]:
    """
    Construye un gráfico Graphviz DOT string.
    Importa graphviz aquí para que el resto de la app cargue.
    """
    try:
        import graphviz
    except ImportError:
        st.error("Librería 'graphviz' no encontrada. Instálala con 'pip install graphviz'")
        return None

    if not component_strings_list:
        return None

    dot = graphviz.Digraph(comment='Análisis de Causa Raíz')
    dot.attr(rankdir='RL')
    dot.attr(bgcolor='transparent')
    
    # --- ¡ARREGLO! ---
    # La línea 'dot.attr('fontcolor', '#FAFAFA')' fue eliminada
    # porque 'fontcolor' no es un atributo de grafo válido.
    
    dot.node_attr.update(
        shape='box', 
        style='filled', 
        fontname='sans-serif',
        fontcolor='#FFFFFF' # Texto del nodo blanco (esto es correcto)
    )
    dot.edge_attr.update(
        fontname='sans-serif', 
        fontsize='10',
        color='#999999',     # Color de flecha gris (esto es correcto)
        fontcolor='#999999'  # Color de texto de flecha gris (esto es correcto)
    )
    # --- FIN DEL ARREGLO ---

    leaf_nodes = set()
    all_nodes = set()
    edges = set()

    for comp_str in component_strings_list:
        if not comp_str:
            continue
        
        # Limpiar y dividir la jerarquía
        parts = [p.strip() for p in comp_str.split(':') if p and p.upper() != 'NONE']
        if not parts:
            continue
        
        # El nodo hoja es el más específico
        leaf_nodes.add(parts[-1])
        all_nodes.update(parts)
        
        # Crear los bordes (L3->L2, L2->L1)
        for i in range(len(parts) - 1, 0, -1):
            child = parts[i]
            parent = parts[i-1]
            edges.add((child, parent))

    if not all_nodes:
        return None

    # Añadir nodos al gráfico
    for node_name in all_nodes:
        if node_name in leaf_nodes:
            # Nodo hoja (la causa raíz) se resalta
            dot.node(node_name, fillcolor='#FF6B6B') # Rojo/Naranja
        else:
            # Nodo padre (sistema)
            dot.node(node_name, fillcolor='#4D96FF') # Azul
    
    # Añadir bordes al gráfico
    for child, parent in edges:
        dot.edge(child, parent, label="es sub-parte de")

    return str(dot) # Devuelve el string DOT

# --- 7. APLICACIÓN STREAMLIT (UI HÍBRIDA MEJORADA) ---
st.set_page_config(page_title="Agente NHTSA RAG", layout="wide")
st.title("🚗 Agente Híbrido de Búsqueda NHTSA")
st.markdown("""
Este demo implementa un **Agente de RAG Híbrido** que combina búsqueda vectorial, grafos de conocimiento y un LLM.

**Pipeline de Análisis:**
1.  **Filtro de Entidad:** Utiliza Make, Model y Year para filtrar la búsqueda.
2.  **Flujo 1 (Grafo "Golden"):** Busca quejas similares (Qdrant) y atraviesa el grafo de Neo4j (`[:ASSOCIATED_RECALL]`) para encontrar recalls *directamente vinculados* al vehículo.
3.  **Flujo 2 (Semántico "Fallback"):** Si el Flujo 1 no encuentra vínculos de texto, realiza una búsqueda vectorial *filtrada* en todo el corpus de Recalls e Investigaciones (Qdrant).
4.  **Agente (LLM):** Un LLM de Groq (Llama 3) analiza el contexto recuperado y genera un resumen experto.
5.  **Visualización (Causa Raíz):** Un grafo (Graphviz) renderiza la jerarquía de los componentes afectados (`[:SUB_OF]`) para un diagnóstico visual inmediato.
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
            component_strings = [] # ¡NUEVO!
            graph_title = ""       # ¡NUEVO!
            
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
                
                graph_title = "Análisis de Causa Raíz (Vía Grafo)"
                component_strings = [
                    r['recall_component_text'] 
                    for res in golden_results.get("results", []) 
                    for r in res.get("actionable_recalls", [])
                    if r.get('recall_component_text')
                ]
            
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
                
                graph_title = "Posibles Componentes Afectados (Vía Semántica)"
                recall_comps = [
                    h['payload'].get('component') 
                    for h in semantic_results.get("recall_hits", []) 
                    if h.get('payload', {}).get('component')
                ]
                invest_comps = [
                    h['payload'].get('component') 
                    for h in semantic_results.get("investigation_hits", []) 
                    if h.get('payload', {}).get('component')
                ]
                component_strings = list(set(recall_comps + invest_comps))

            st.success("Paso 2/3: Datos recuperados. Pasando al LLM de Groq para análisis...")
            
            col_summary, col_graph = st.columns([2, 1])

            with col_summary:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response = st.write_stream(
                        get_llm_summary(user_query, llm_prompt, context_str)
                    )
            
            st.success("Paso 3/3: Análisis completado.")

            # ¡NUEVO BLOQUE DE CÓDIGO PARA EL GRAFO!
            if component_strings:
                with col_graph:
                    st.subheader(graph_title)
                    with st.spinner("Generando visualización de componentes..."):
                        # Usamos 'try' por si 'graphviz' falla
                        try:
                            graph_dot_string = build_component_graph(component_strings)
                            if graph_dot_string:
                                st.graphviz_chart(graph_dot_string)
                            else:
                                st.warning("No se pudieron extraer componentes para el gráfico.")
                        except Exception as e_graph:
                            st.error(f"Error al renderizar el gráfico: {e_graph}")
                            st.caption("Asegúrate de que 'graphviz' esté instalado en el sistema (ej. 'brew install graphviz')")

            with st.expander("Ver datos crudos recuperados (JSON)"):
                st.json(rag_results)

        except Exception as e:
            st.error(f"Ha ocurrido un error fatal durante el análisis: {e}")
            import traceback
            st.error(traceback.format_exc())
