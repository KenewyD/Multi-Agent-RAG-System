"""
🕸️ Multi-Agent RAG System — LangGraph-style Architecture
Auteur: KENEWY DIALLO | AI Engineer | LLM, RAG & AWS
Dashboard de démonstration d'un système multi-agent pour la réduction des hallucinations.
"""
import json
import streamlit as st

# ========== DONNÉES SIMULÉES : RAG Simple vs Multi-Agent ==========
SIMPLE_RAG_RESULTS = [
    {"question": "Qu'est-ce que le RAG ?", "answer": "Le RAG est une méthode d'IA.", "hallucination_score": 0.35, "latency_ms": 420},
    {"question": "Comment réduire les hallucinations ?", "answer": "On utilise des techniques avancées.", "hallucination_score": 0.52, "latency_ms": 380},
    {"question": "Quels frameworks pour le RAG ?", "answer": "LangChain et d'autres outils.", "hallucination_score": 0.28, "latency_ms": 510},
    {"question": "Différence fine-tuning vs RAG ?", "answer": "Le fine-tuning modifie le modèle, le RAG enrichit le prompt.", "hallucination_score": 0.41, "latency_ms": 450},
    {"question": "Rôle du chunking ?", "answer": "Le chunking découpe les documents.", "hallucination_score": 0.33, "latency_ms": 390},
]

MULTI_AGENT_RESULTS = [
    {"question": "Qu'est-ce que le RAG ?", "answer": "Le RAG combine la récupération d'informations documentaires avec la génération de texte par LLM.", "hallucination_score": 0.08, "latency_ms": 850},
    {"question": "Comment réduire les hallucinations ?", "answer": "Techniques : prompt strict, citations sources, boucle de vérification, filtrage par score de similarité.", "hallucination_score": 0.12, "latency_ms": 920},
    {"question": "Quels frameworks pour le RAG ?", "answer": "LangChain (orchestration), LlamaIndex (indexation), Haystack (recherche), ChromaDB/Qdrant (vector stores).", "hallucination_score": 0.05, "latency_ms": 780},
    {"question": "Différence fine-tuning vs RAG ?", "answer": "Fine-tuning : entraînement du modèle sur données spécifiques. RAG : enrichissement dynamique du prompt sans modifier les poids du modèle.", "hallucination_score": 0.09, "latency_ms": 880},
    {"question": "Rôle du chunking ?", "answer": "Le chunking découpe les documents en segments optimaux pour le retrieval, équilibrant granularité et contexte.", "hallucination_score": 0.07, "latency_ms": 810},
]

# Stats globales
simple_avg_hallu = sum(r["hallucination_score"] for r in SIMPLE_RAG_RESULTS) / len(SIMPLE_RAG_RESULTS)
multi_avg_hallu = sum(r["hallucination_score"] for r in MULTI_AGENT_RESULTS) / len(MULTI_AGENT_RESULTS)
reduction = ((simple_avg_hallu - multi_avg_hallu) / simple_avg_hallu) * 100

simple_avg_lat = sum(r["latency_ms"] for r in SIMPLE_RAG_RESULTS) / len(SIMPLE_RAG_RESULTS)
multi_avg_lat = sum(r["latency_ms"] for r in MULTI_AGENT_RESULTS) / len(MULTI_AGENT_RESULTS)

st.set_page_config(page_title="Multi-Agent RAG System", page_icon="🕸️", layout="wide")

st.title("🕸️ Multi-Agent RAG System")
st.caption("Architecture LangGraph — Réduction des hallucinations par agents spécialisés | KENEWY DIALLO")

st.markdown("---")

# ========== ARCHITECTURE VISUELLE ==========
st.markdown("### 🏗️ Architecture du Système Multi-Agent")

col_arch = st.columns(5)
with col_arch[0]:
    st.info("📥\n**Agent Router**\nAnalyse la question et route vers le bon agent")
with col_arch[1]:
    st.success("🔍\n**Agent Recherche**\nRécupère les documents pertinents (top-k)")
with col_arch[2]:
    st.warning("📝\n**Agent Synthèse**\nGénère une réponse structurée avec citations")
with col_arch[3]:
    st.error("✅\n**Agent Vérification**\nDétecte les hallucinations et corrige")
with col_arch[4]:
    st.success("💬\n**Réponse Finale**\nRéponse validée avec score de confiance")

st.markdown("---")

# ========== KPIs COMPARATIFS ==========
st.markdown("### 📊 RAG Simple vs Multi-Agent")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hallucinations RAG Simple", f"{simple_avg_hallu:.2f}")
c2.metric("Hallucinations Multi-Agent", f"{multi_avg_hallu:.2f}", delta=f"-{reduction:.0f}%")
c3.metric("Réduction", f"{reduction:.0f}%", delta_color="inverse")
c4.metric("Latence RAG Simple", f"{simple_avg_lat:.0f}ms")
c5.metric("Latence Multi-Agent", f"{multi_avg_lat:.0f}ms")

st.markdown("---")

# ========== GRAPHIQUES COMPARATIFS ==========
st.markdown("### 📈 Comparaison par Question")

questions = [r["question"][:40] + "..." for r in SIMPLE_RAG_RESULTS]

# Hallucination
hallu_simple = [r["hallucination_score"] for r in SIMPLE_RAG_RESULTS]
hallu_multi = [r["hallucination_score"] for r in MULTI_AGENT_RESULTS]

st.markdown("**Score d'hallucination par question (plus bas = mieux)**")
hallu_data = {f"Q{i+1}": [hallu_simple[i], hallu_multi[i]] for i in range(len(questions))}
# Streamlit bar_chart attend un dict de listes ou un DataFrame
import pandas as pd
hallu_df = pd.DataFrame({
    "RAG Simple": hallu_simple,
    "Multi-Agent": hallu_multi
}, index=[f"Q{i+1}" for i in range(len(questions))])
st.bar_chart(hallu_df, height=300)

# Latence
lat_df = pd.DataFrame({
    "RAG Simple": [r["latency_ms"] for r in SIMPLE_RAG_RESULTS],
    "Multi-Agent": [r["latency_ms"] for r in MULTI_AGENT_RESULTS]
}, index=[f"Q{i+1}" for i in range(len(questions))])
st.markdown("**Latence par question (ms)**")
st.bar_chart(lat_df, height=300)

st.markdown("---")

# ========== DÉTAIL QUESTION PAR QUESTION ==========
st.markdown("### 🔍 Détail des Réponses")

selected_q = st.selectbox("Sélectionner une question", range(len(SIMPLE_RAG_RESULTS)), 
                           format_func=lambda i: SIMPLE_RAG_RESULTS[i]["question"])

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🤖 RAG Simple")
    st.markdown(f"**Réponse :** {SIMPLE_RAG_RESULTS[selected_q]['answer']}")
    st.progress(1 - SIMPLE_RAG_RESULTS[selected_q]["hallucination_score"], 
                text=f"Fiabilité : {(1-SIMPLE_RAG_RESULTS[selected_q]['hallucination_score'])*100:.0f}%")
    st.caption(f"Latence : {SIMPLE_RAG_RESULTS[selected_q]['latency_ms']}ms")

with col_right:
    st.markdown("#### 🕸️ Multi-Agent (LangGraph)")
    st.markdown(f"**Réponse :** {MULTI_AGENT_RESULTS[selected_q]['answer']}")
    st.progress(1 - MULTI_AGENT_RESULTS[selected_q]["hallucination_score"], 
                text=f"Fiabilité : {(1-MULTI_AGENT_RESULTS[selected_q]['hallucination_score'])*100:.0f}%")
    st.caption(f"Latence : {MULTI_AGENT_RESULTS[selected_q]['latency_ms']}ms")

st.markdown("---")

# ========== TABLEAU RÉCAPITULATIF ==========
st.markdown("### 📋 Tableau Comparatif")

table_rows = []
for i in range(len(SIMPLE_RAG_RESULTS)):
    table_rows.append({
        "Question": f"Q{i+1}",
        "Hallu. Simple": SIMPLE_RAG_RESULTS[i]["hallucination_score"],
        "Hallu. Multi": MULTI_AGENT_RESULTS[i]["hallucination_score"],
        "Gain": round(SIMPLE_RAG_RESULTS[i]["hallucination_score"] - MULTI_AGENT_RESULTS[i]["hallucination_score"], 2),
        "Lat. Simple": SIMPLE_RAG_RESULTS[i]["latency_ms"],
        "Lat. Multi": MULTI_AGENT_RESULTS[i]["latency_ms"],
    })

st.dataframe(table_rows, use_container_width=True, hide_index=True)

st.markdown("---")

# ========== EXPORT ==========
st.markdown("### 💾 Export")
export_data = {
    "architecture": "Multi-Agent RAG (LangGraph-style)",
    "agents": ["Router", "Recherche", "Synthèse", "Vérification"],
    "comparison": {
        "rag_simple": {
            "avg_hallucination": round(simple_avg_hallu, 3),
            "avg_latency_ms": round(simple_avg_lat, 1),
            "results": SIMPLE_RAG_RESULTS
        },
        "multi_agent": {
            "avg_hallucination": round(multi_avg_hallu, 3),
            "avg_latency_ms": round(multi_avg_lat, 1),
            "hallucination_reduction_percent": round(reduction, 1),
            "results": MULTI_AGENT_RESULTS
        }
    }
}

st.download_button(
    "📥 Télécharger le rapport (JSON)",
    data=json.dumps(export_data, indent=2, ensure_ascii=False),
    file_name="multi_agent_rag_report.json",
    mime="application/json"
)

st.markdown("---")
st.caption("Multi-Agent RAG System — KENEWY DIALLO | AI Engineer | LLM, RAG & AWS | Entretien 24 Août 2026")
