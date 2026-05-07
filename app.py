
import streamlit as st
import pymongo
from google import genai
from google.genai import types

# =======================
# CONFIGURACIÓN DE PÁGINA
# =======================

st.set_page_config(
    page_title="Agentes en IA Generativa — Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =======================
# ESTILOS PERSONALIZADOS
# =======================

st.markdown("""
<style>
    /* Paleta de colores institucional */
    :root {
        --azul-ese: #0068B4;
        --azul-claro: #E8F3FB;
        --gris-texto: #2C2C2C;
        --borde: #D0E6F5;
    }

    /* Encabezado principal */
    .titulo-principal {
        background: linear-gradient(135deg, #0068B4 0%, #004A8C 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    .titulo-principal h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .titulo-principal p {
        font-size: 0.9rem;
        opacity: 0.85;
        margin: 0;
    }

    /* Tarjeta de sugerencia */
    .sugerencia-card {
        background: #E8F3FB;
        border: 1px solid #B3D4EE;
        border-left: 4px solid #0068B4;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        margin-bottom: 0.4rem;
        cursor: pointer;
        font-size: 0.88rem;
        color: #004A8C;
        transition: background 0.2s;
    }
    .sugerencia-card:hover {
        background: #CCE4F5;
    }

    /* Badge de sección */
    .seccion-badge {
        display: inline-block;
        background: #0068B4;
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
        letter-spacing: 0.03em;
    }

    /* Info lateral */
    .info-box {
        background: #F0F7FF;
        border: 1px solid #B3D4EE;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        font-size: 0.84rem;
        color: #2C2C2C;
        margin-bottom: 1rem;
    }
    .info-box strong {
        color: #0068B4;
    }
</style>
""", unsafe_allow_html=True)


# =======================
# CONFIGURACIÓN (secrets)
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI    = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de entorno GOOGLE_API_KEY o MONGODB_URI")
    st.stop()


# =======================
# CLIENTES (cacheados)
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["pdf_embeddings_db"]
    return db["pdf_vectors"]

client_genai = get_genai_client()
collection   = get_mongo_collection()


# =======================
# FUNCIONES PRINCIPALES
# =======================

def crear_embedding(texto: str):
    """
    Genera embedding de la query con task_type='RETRIEVAL_QUERY'
    (modelo gemini-embedding-001, 768 dims, normalizado L2).
    """
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def buscar_similares(embedding, k: int = 5):
    """
    Búsqueda semántica en MongoDB Atlas Vector Search
    (índice 'vector_index' sobre el campo 'embedding').
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def generar_respuesta(pregunta: str, contextos: list[dict]) -> str:
    """RAG: responde usando EXCLUSIVAMENTE los fragmentos recuperados del PDF."""
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""Eres un asistente experto en inteligencia artificial generativa y desarrollo de agentes.
Usa EXCLUSIVAMENTE el siguiente contexto extraído del documento "El Desarrollo de Agentes en la Inteligencia Artificial Generativa" (ESE Business School, marzo 2025) para responder la pregunta del usuario.
Si la respuesta no está en el contexto, indícalo claramente y sugiere qué sección del documento podría abordar el tema.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma concisa, clara y estructurada en español. Si corresponde, menciona conceptos clave del documento."""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# =======================
# SIDEBAR — INFORMACIÓN
# =======================

with st.sidebar:
    st.markdown('<span class="seccion-badge">📄 DOCUMENTO BASE</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>El Desarrollo de Agentes en la IA Generativa</strong><br>
        ESE Business School · Universidad de los Andes<br>
        <em>Cápsulas sobre IA Generativa — Marzo 2025</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="seccion-badge">📚 TEMAS DEL DOCUMENTO</span>', unsafe_allow_html=True)
    temas = [
        "🔷 Perspectiva histórica de los agentes",
        "🔷 IA generativa y rol de los agentes",
        "🔷 Diferencias con transformers",
        "🔷 Arquitecturas modulares",
        "🔷 Aprendizaje por refuerzo jerárquico",
        "🔷 Meta-aprendizaje y adaptabilidad",
    ]
    for t in temas:
        st.markdown(f"<div style='font-size:0.83rem; padding:0.2rem 0; color:#2C2C2C'>{t}</div>",
                    unsafe_allow_html=True)

    st.divider()

    # Botón para limpiar historial
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.historial = []
        st.session_state.pregunta_sugerida = ""
        st.rerun()

    st.caption("Powered by Gemini + MongoDB Atlas Vector Search")


# =======================
# ENCABEZADO PRINCIPAL
# =======================

st.markdown("""
<div class="titulo-principal">
    <h1>🤖 Chatbot sobre Agentes en IA Generativa</h1>
    <p>Consulta el documento de ESE Business School · Universidad de los Andes (Marzo 2025)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")


# =======================
# PREGUNTAS SUGERIDAS
# =======================

SUGERENCIAS = [
    "¿Qué son los agentes autónomos en IA y cómo funcionan?",
    "¿Cuál es la diferencia entre un agente de IA y un transformer?",
    "¿Cómo funciona el aprendizaje por refuerzo jerárquico (HRL)?",
    "¿Qué módulos componen la arquitectura de un agente moderno?",
    "¿Qué es el meta-aprendizaje y por qué es importante en agentes?",
    "¿Cómo evolucionaron los sistemas multiagente en las décadas de 1980-1990?",
]

if "historial" not in st.session_state:
    st.session_state.historial = []

if "pregunta_sugerida" not in st.session_state:
    st.session_state.pregunta_sugerida = ""

# Mostrar sugerencias solo si la conversación está vacía
if not st.session_state.historial:
    st.markdown('<span class="seccion-badge">💡 PREGUNTAS SUGERIDAS</span>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, sug in enumerate(SUGERENCIAS):
        col = cols[i % 2]
        with col:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.pregunta_sugerida = sug
                st.rerun()
    st.markdown("")


# =======================
# HISTORIAL DE CHAT
# =======================

for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])


# =======================
# INPUT DEL USUARIO
# =======================

# Si se seleccionó una sugerencia, usarla como pregunta
pregunta_inicial = st.session_state.get("pregunta_sugerida", "")
pregunta = st.chat_input("Escribe tu pregunta sobre el documento...")

# Usar la sugerencia si no hay nueva pregunta escrita
if not pregunta and pregunta_inicial:
    pregunta = pregunta_inicial
    st.session_state.pregunta_sugerida = ""


# =======================
# PROCESAMIENTO
# =======================

if pregunta:
    st.chat_message("user").write(pregunta)
    st.session_state.historial.append({"rol": "usuario", "texto": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el documento..."):
            try:
                emb      = crear_embedding(pregunta)
                similares = buscar_similares(emb, k=5)

                if not similares:
                    respuesta = (
                        "⚠️ No encontré fragmentos relevantes en el documento para tu pregunta. "
                        "Intenta reformularla usando términos del documento, como *agentes*, "
                        "*transformers*, *aprendizaje por refuerzo* o *arquitecturas modulares*."
                    )
                else:
                    respuesta = generar_respuesta(pregunta, similares)

            except Exception as e:
                respuesta = f"⚠️ Ocurrió un error al procesar tu consulta: {e}"

        st.write(respuesta)

        # Fragmentos recuperados (expandible)
        if "similares" in locals() and similares:
            with st.expander("🔍 Ver fragmentos recuperados del PDF"):
                for i, c in enumerate(similares, 1):
                    score_pct = round(c["score"] * 100, 1)
                    st.markdown(
                        f"**Fragmento {i}** &nbsp;·&nbsp; "
                        f"<span style='color:#0068B4;font-weight:600'>Relevancia: {score_pct}%</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(c["texto"][:500] + ("…" if len(c["texto"]) > 500 else ""))
                    st.divider()

    st.session_state.historial.append({"rol": "bot", "texto": respuesta})
