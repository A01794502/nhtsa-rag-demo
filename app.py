# ===============================================
#          app.py (Tu Agente NHTSA en vivo)
# ===============================================

import streamlit as st
import os
import numpy as np
import json
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import torch
import joblib  # Para el .pkl (aunque ya no lo usemos, es buena práctica)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Dict, Any, List

# --- 1. CONFIGURACIÓN DE CREDENCIALES ---
# Streamlit Cloud leerá esto desde los "Secrets" de tu dashboard
NEO4J_URI = st.secrets.get("NEO4J_URI", os.getenv("NEO4J_URI"))
NEO4J_USER = st.secrets.get("NEO4J_USER", os.getenv("NEO4J_USER"))
NEO4J_PASS = st.secrets.get("NEO4J_PASS", os.getenv("NEO4J_PASS"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_KEY = st.secrets.get("QDRANT_KEY", os.getenv("QDRANT_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
E5_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"


# --- 2. CLIENTES Y MODELOS (¡CON CACHÉ!) ---
# @st.cache_resource asegura que esto solo se ejecute UNA VEZ
# (Evita recargar el modelo E5 de 1.1GB en cada clic)

@st.cache_resource
def get_neo_driver():
    """Conecta a Neo4j (una sola vez)."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()
    print("Conectado a Neo4j.")
    return driver

@st.cache_resource
def get_qdrant_client():
    """Conecta a Qdrant (una sola vez)."""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=180)
    print("Conectado a Qdrant.")
    return client

@st.cache_resource
def get_encoder():
    """Carga el modelo E5 (una sola vez)."""
    print(f"Cargando modelo E5: {E5_MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(E5_MODEL_NAME, device=device)
    if torch.cuda.is_available():
        try:
            encoder._first_module().auto_model.to(dtype=torch.float16)
        except Exception:
            pass
    encoder.max_seq_length = 256
    print("Modelo E5 cargado.")
    return encoder

@st.cache_resource
def get_llm():
    """Carga el LLM de Groq (una sola vez)."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY no está configurada en st.secrets")
    # Usamos Llama 3.1 70b, es rápido y potente
    return ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.1,  # Baja temperatura para análisis factual
        max_tokens=2048,
    )

# --- 3. FUNCIONES DE BÚSQUEDA (NUESTRO PIPELINE v5) ---
# (Estas son las funciones que depuramos y validamos)

@torch.no_grad()
def e5_query(text: str) -> np.ndarray:
    enc = get_encoder()
    vec = enc.encode(
        [f"query: {text}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=4,
        show_progress_bar=False
    )[0]
    return np.asarray(vec, dtype=np.float32)

def cypher(query: str, params: Optional[Dict[str, Any]] = None):
    with get_neo_driver().session() as s:
        return s.run(query, **(params or {})).data()

def qdrant_search(qv, k=15, mmy=None, collection=None, type_in=None):
    client = get_qdrant_client()
    if collection is None:
        mapping = {
            "recall": "nhtsa_recalls",
            "investigation": "nhtsa_investigations",
            "complaint": "nhtsa_complaints",
        }
        collection = mapping.get((type_in or "recall").lower(), "nhtsa_recalls")
    
    qdrant_filter = None
    if mmy:
        conds = [models.FieldCondition(key=kf, match=models.MatchValue(value=vf)) for kf, vf in mmy.items()]
        qdrant_filter = models.Filter(must=conds)
    
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
    """Recupera payloads de Qdrant filtrando por el campo 'id' del payload."""
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

# --- 4. FUNCIÓN 'ask()' PRINCIPAL (Nuestra v5) ---
def ask(question: str, mmy: Optional[Dict[str, Any]] = None, k: int = 15, type_in: str = "complaint"):
    """
    Ejecuta el pipeline RAG Híbrido:
    1. Qdrant(Complaints) -> comp_id
    2. Neo4j(comp_id) -> [:ASSOCIATED_RECALL] -> recall_id
    3. Qdrant(Recalls) -> full_text
    """
    try:
        qv = e5_query(question)
    except Exception as e:
        return {"error": f"Falla al codificar texto: {e}"}

    try:
        hits = qdrant_search(qv, k=k, mmy=mmy, type_in=type_in)
    except Exception as e:
        return {"error": f"Falla en búsqueda de Qdrant: {e}"}

    # Flujo 2 (Directo) - No lo usamos para este demo, pero está listo
    if type_in == "recall" or type_in == "investigation":
        return {"query": question, "hits_con_texto_completo": hits}
    
    # Flujo 1 (Golden) - El núcleo de nuestro demo
    elif type_in == "complaint":
        ids = [str(h.get("payload", {}).get("id")) for h in hits if h.get("payload", {}).get("id")]
        if not ids:
            return {"query": question, "hits": hits, "results": []}

        # PASO 2: Neo4j (Obtener IDs de Recall)
        query = """
        UNWIND $ids AS cid
        MATCH (c:Complaint {comp_id: cid})
        MATCH (c)-[rel:ASSOCIATED_RECALL]->(r:Recall)
        RETURN 
          c.comp_id AS complaint_comp_id,
          r.id AS recall_id,
          r.subject AS recall_summary,
          r.consequence AS recall_consequence,
          r.component AS recall_component_text
        ORDER BY c.comp_id
        """
        neo4j_details = cypher(query, {"ids": ids})
        
        if not neo4j_details:
             return {"query": question, "hits": hits, "results": []}

        # PASO 3: OBTENER TEXTO COMPLETO DE QDRANT(RECALLS)
        recall_ids_encontrados = list(set(r['recall_id'] for r in neo4j_details if r['recall_id']))
        mapa_texto_completo = qdrant_retrieve_by_ids(recall_ids_encontrados, "nhtsa_recalls")
        
        # PASO 4: Unir todo y CLASIFICAR
        results_by_complaint = {}
        for hit in hits:
            p = hit.get("payload", {})
            comp_id = str(p.get("id"))
            if comp_id:
                results_by_complaint[comp_id] = {
                    "complaint_hit": hit,
                    "actionable_recalls": [],  # Para el LLM
                    "historical_links": []   # Para mención
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

        return {
            "query": question, 
            "results": final_results
        }
    
    return {"error": "Tipo de búsqueda no válido."}

# --- 5. LÓGICA DEL AGENTE LLM ---

# El prompt le dice al LLM CÓMO interpretar nuestro JSON
SYSTEM_PROMPT = """
Eres un asistente experto en seguridad vehicular de la NHTSA.
Tu trabajo es analizar la queja de un usuario y compararla con el CONTEXTO TÉCNICO proporcionado.

El contexto se divide en dos partes:
1.  **RECALLS ACCIONABLES:** Estos son recalls con texto completo que coinciden con quejas similares a la del usuario.
2.  **VÍNCULOS HISTÓRICOS:** Estos son IDs de recalls más antiguos que también están vinculados a esta queja, indicando un posible patrón a largo plazo.

QUEJA DEL USUARIO:
{question}

CONTEXTO TÉCNICO RECUPERADO:
{context}

INSTRUCCIONES PARA TU RESPUESTA:
1.  **Analiza** la queja del usuario.
2.  **Revisa** los "RECALLS ACCIONABLES". Resume los hallazgos más relevantes. Explica *por qué* el recall (el defecto y la consecuencia) coincide con la queja del usuario.
3.  **Menciona** los "VÍNCULOS HISTÓRICOS" si existen. Explica que estos son IDs de recalls más antiguos que también están vinculados, sugiriendo un patrón histórico.
4.  **Concluye** con un resumen profesional.
5.  **Responde** en español y sé directo. No inventes información. Si el contexto está vacío, dilo.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT)
])

def format_context_for_llm(rag_results: dict) -> str:
    """Formatea nuestro JSON 'v5' en un texto limpio para el prompt del LLM."""
    if not rag_results.get("results"):
        return "No se encontraron quejas similares ni recalls asociados en la base de datos."

    context_parts = []
    
    for i, res in enumerate(rag_results["results"], 1):
        comp_id = res['complaint_hit']['payload'].get('id')
        score = res['complaint_hit']['score']
        context_parts.append(f"\n--- ANÁLISIS DEL HIT DE QUEJA #{i} (ID: {comp_id}, Similitud: {score:.2f}) ---")

        # 1. Recalls Accionables (con texto)
        if res['actionable_recalls']:
            context_parts.append("\n**RECALLS ACCIONABLES (Con texto completo):**")
            for recall in res['actionable_recalls']:
                context_parts.append(f"  - ID Recall: {recall['recall_id']}")
                context_parts.append(f"    Defecto (Resumen): {recall['recall_summary']}")
                context_parts.append(f"    Riesgo (Consecuencia): {recall['recall_consequence']}")
                context_parts.append(f"    Texto Completo: {recall['full_text_from_qdrant'][:600]}...") # Más texto
        
        # 2. Vínculos Históricos (sin texto)
        if res['historical_links']:
            context_parts.append("\n**VÍNCULOS HISTÓRICOS (Sin texto, solo ID):**")
            ids = [r['recall_id'] for r in res['historical_links']]
            context_parts.append(f"  - IDs vinculados: {', '.join(ids)}")
    
    return "\n".join(context_parts)

def get_summary_from_llm(user_query: str, rag_results: dict):
    """
    Esta es la cadena LangChain: (Formato de Contexto) -> Prompt -> LLM -> Output
    """
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    
    # Formatear el contexto de nuestro pipeline v5
    context_str = format_context_for_llm(rag_results)
    
    # Invocar con streaming
    return chain.stream({
        "context": context_str,
        "question": user_query
    })

# --- 6. APLICACIÓN STREAMLIT (UI) ---

st.set_page_config(page_title="Agente NHTSA RAG", layout="wide")
st.title("🚗 Agente Híbrido de Búsqueda NHTSA")
st.markdown("""
Este demo implementa el pipeline **"Golden" (Flujo 1)**.
1.  Busca quejas similares en **Qdrant**.
2.  Usa el ID para atravesar el grafo en **Neo4j** (`[:ASSOCIATED_RECALL]`).
3.  Recupera el texto completo del recall desde **Qdrant**.
4.  Un LLM (**Groq/Llama 3.1**) analiza y resume los hallazgos.
""")

user_query = st.text_input(
    "Describe la falla de tu vehículo:", 
    placeholder="Ej: El pedal de freno se fue al fondo y perdí potencia de frenado"
)

if st.button("Analizar Falla", type="primary"):
    if not user_query:
        st.error("Por favor, introduce una descripción de la falla.")
    else:
        try:
            # Paso 1: Llamar a nuestro pipeline de datos v5
            with st.spinner("Paso 1/3: Buscando quejas similares en Qdrant..."):
                rag_results = ask(user_query, type_in="complaint") # ¡Nuestra función v5!
            
            if rag_results.get("error"):
                st.error(f"Error en el pipeline: {rag_results['error']}")
            elif not rag_results.get("results"):
                st.warning("Se encontraron quejas similares, pero ninguna tiene un recall asociado en Neo4j.")
            else:
                st.success("Paso 2/3: Datos recuperados. Pasando al LLM de Groq para análisis...")

                # Paso 2: Llamar al LLM con esos resultados
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    # Usar write_stream para el efecto de "escritura"
                    response = st.write_stream(get_summary_from_llm(user_query, rag_results))
                
                st.success("Paso 3/3: Análisis completado.")

                # Opcional: Mostrar los datos crudos recuperados
                with st.expander("Ver datos crudos recuperados (JSON)"):
                    st.json(rag_results)

        except Exception as e:
            st.error(f"Ha ocurrido un error fatal durante el análisis: {e}")
            import traceback
            st.error(traceback.format_exc())