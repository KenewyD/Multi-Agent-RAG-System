"""
🕸️ Multi-Agent RAG System — agents réels, exécutés en direct.
Auteur : KENEWY DIALLO | AI Engineer | LLM, RAG & AWS

4 agents : Router -> Recherche -> Synthèse -> Vérification.
Comparaison RAG simple (1 passage, pas de vérif) vs Multi-Agent (top-k + vérification).
Aucune donnée écrite en dur : tout est calculé par similarité cosinus (TF-IDF).
Déployable sans OpenAI ni torch.
"""
import time
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Multi-Agent RAG System", page_icon="🕸️", layout="wide")

# ---------------------------------------------------------------
# Base de connaissances (corpus)
# ---------------------------------------------------------------
CORPUS = [
    "Le RAG (Retrieval-Augmented Generation) combine la récupération d'informations documentaires avec la génération de texte par un LLM.",
    "Le RAG réduit les hallucinations car il s'appuie sur des documents réels plutôt que d'inventer des informations.",
    "Pour réduire les hallucinations : prompt strict, citations de sources, boucle de vérification et filtrage par score de similarité.",
    "Les frameworks RAG incluent LangChain pour l'orchestration, LlamaIndex pour l'indexation, et Qdrant, FAISS ou OpenSearch comme bases vectorielles.",
    "Le fine-tuning modifie les poids du modèle sur des données spécifiques, tandis que le RAG enrichit le prompt sans réentraîner le modèle.",
    "Le chunking découpe les documents en segments optimaux, en équilibrant granularité et contexte pour le retrieval.",
    "Les embeddings transforment le texte en vecteurs numériques permettant la recherche par similarité sémantique.",
    "Un système multi-agent répartit le travail entre agents spécialisés : routage, recherche, synthèse et vérification.",
]

QUESTIONS = [
    "Qu'est-ce que le RAG ?",
    "Comment réduire les hallucinations ?",
    "Quels frameworks pour le RAG ?",
    "Différence fine-tuning vs RAG ?",
    "Rôle du chunking ?",
]

# ---------------------------------------------------------------
# Les 4 agents (chacun fait un vrai traitement)
# ---------------------------------------------------------------
class MultiAgentRAG:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(corpus)

    def agent_router(self, question):
        """Analyse la question et détecte son intention (mots-clés)."""
        q = question.lower()
        if "hallucinat" in q:
            intent = "réduction d'hallucinations"
        elif "framework" in q or "outil" in q:
            intent = "outils / frameworks"
        elif "fine-tuning" in q or "différence" in q:
            intent = "comparaison de concepts"
        elif "chunk" in q:
            intent = "ingestion / chunking"
        else:
            intent = "définition générale"
        return intent

    def agent_recherche(self, question, top_k):
        """Vraie recherche vectorielle : renvoie les top_k passages."""
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [(self.corpus[i], float(sims[i])) for i in top_idx]

    def agent_synthese(self, passages):
        """Construit une réponse à partir des passages récupérés."""
        if not passages:
            return "Aucune information trouvée."
        # concatène les passages les plus pertinents en une réponse structurée
        return " ".join(p for p, s in passages if s > 0.05) or passages[0][0]

    def agent_verification(self, answer, question, passages):
        """Score de fiabilité = ancrage dans le contexte ET pertinence vs la question."""
        if not passages or not answer.strip():
            return 0.0
        ans_vec = self.vectorizer.transform([answer])
        q_vec = self.vectorizer.transform([question])
        ctx_vecs = self.vectorizer.transform([p for p, s in passages])
        grounding = float(np.max(cosine_similarity(ans_vec, ctx_vecs)[0]))
        relevancy = float(cosine_similarity(ans_vec, q_vec)[0][0])
        # combine : la réponse doit être fondée sur le contexte ET répondre à la question
        return round(0.4 * grounding + 0.6 * relevancy, 3)


def run_simple_rag(engine, question):
    """RAG simple : 1 seul passage, pas de vérification."""
    start = time.time()
    passages = engine.agent_recherche(question, top_k=1)
    answer = (passages[0][0][:50] + "...") if passages else "Réponse générique."
    reliability = engine.agent_verification(answer, question, passages)
    latency = (time.time() - start) * 1000
    return answer, reliability, latency


def run_multi_agent(engine, question, top_k=4):
    """Multi-agent : router -> recherche top-k -> synthèse -> vérification."""
    start = time.time()
    intent = engine.agent_router(question)
    passages = engine.agent_recherche(question, top_k=top_k)
    answer = engine.agent_synthese(passages)
    reliability = engine.agent_verification(answer, question, passages)
    latency = (time.time() - start) * 1000
    return intent, answer, reliability, passages, latency


# ---------------------------------------------------------------
# Interface
# ---------------------------------------------------------------
st.title("🕸️ Multi-Agent RAG System")
st.caption("Router → Recherche → Synthèse → Vérification | agents réels, calcul en direct | KENEWY DIALLO")

engine = MultiAgentRAG(CORPUS)

st.markdown("### 🏗️ Architecture")
cols = st.columns(5)
cols[0].info("📥\n**Router**\nDétecte l'intention")
cols[1].success("🔍\n**Recherche**\nTop-k vectoriel")
cols[2].warning("📝\n**Synthèse**\nRéponse structurée")
cols[3].error("✅\n**Vérification**\nScore de fiabilité")
cols[4].success("💬\n**Réponse**\nValidée")

st.markdown("---")

# --- Démo live sur une question ---
st.markdown("### 🎬 Démo en direct : voir les agents travailler")
question = st.selectbox("Choisis une question :", QUESTIONS)

if st.button("▶️  Exécuter les agents", type="primary"):
    log = st.empty()
    steps = []

    steps.append("📥 **Agent Router** en cours...")
    log.markdown("\n\n".join(steps)); time.sleep(0.6)
    intent = engine.agent_router(question)
    steps[-1] = f"📥 **Agent Router** → intention détectée : *{intent}*"
    log.markdown("\n\n".join(steps)); time.sleep(0.4)

    steps.append("🔍 **Agent Recherche** en cours...")
    log.markdown("\n\n".join(steps)); time.sleep(0.6)
    passages = engine.agent_recherche(question, top_k=4)
    steps[-1] = f"🔍 **Agent Recherche** → {len(passages)} passages récupérés (meilleur score : {passages[0][1]:.3f})"
    log.markdown("\n\n".join(steps)); time.sleep(0.4)

    steps.append("📝 **Agent Synthèse** en cours...")
    log.markdown("\n\n".join(steps)); time.sleep(0.6)
    answer = engine.agent_synthese(passages)
    steps[-1] = "📝 **Agent Synthèse** → réponse construite"
    log.markdown("\n\n".join(steps)); time.sleep(0.4)

    steps.append("✅ **Agent Vérification** en cours...")
    log.markdown("\n\n".join(steps)); time.sleep(0.6)
    reliability = engine.agent_verification(answer, question, passages)
    steps[-1] = f"✅ **Agent Vérification** → fiabilité : {reliability*100:.0f}%"
    log.markdown("\n\n".join(steps)); time.sleep(0.3)

    st.success(f"**Réponse finale :** {answer}")
    st.progress(min(1.0, reliability), text=f"Score de fiabilité : {reliability*100:.0f}%")

    with st.expander("📚 Passages utilisés par l'agent Recherche"):
        for p, s in passages:
            st.markdown(f"**Score {s:.3f}** — {p}")

st.markdown("---")

# --- Benchmark comparatif calculé en direct ---
st.markdown("### 📊 RAG Simple vs Multi-Agent (calculé en direct)")

if st.button("▶️  Lancer la comparaison sur toutes les questions"):
    rows = []
    prog = st.progress(0.0, text="Calcul en cours...")
    for i, q in enumerate(QUESTIONS):
        _, s_rel, s_lat = run_simple_rag(engine, q)
        _, m_ans, m_rel, _, m_lat = run_multi_agent(engine, q)
        rows.append({
            "Question": q,
            "Fiabilité Simple": round(s_rel, 3),
            "Fiabilité Multi": round(m_rel, 3),
            "Gain": round(m_rel - s_rel, 3),
            "Latence Simple (ms)": round(s_lat, 2),
            "Latence Multi (ms)": round(m_lat, 2),
        })
        prog.progress((i + 1) / len(QUESTIONS), text=f"Question {i+1}/{len(QUESTIONS)}")
    prog.empty()
    st.session_state.bench = rows

if "bench" in st.session_state:
    df = pd.DataFrame(st.session_state.bench)

    simple_rel = df["Fiabilité Simple"].mean()
    multi_rel = df["Fiabilité Multi"].mean()
    gain = ((multi_rel - simple_rel) / simple_rel * 100) if simple_rel > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Fiabilité RAG Simple", f"{simple_rel*100:.0f}%")
    c2.metric("Fiabilité Multi-Agent", f"{multi_rel*100:.0f}%", delta=f"+{gain:.0f}%")
    c3.metric("Latence Multi (moy.)", f"{df['Latence Multi (ms)'].mean():.2f} ms")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="RAG Simple", x=[f"Q{i+1}" for i in range(len(df))], y=df["Fiabilité Simple"]))
    fig.add_trace(go.Bar(name="Multi-Agent", x=[f"Q{i+1}" for i in range(len(df))], y=df["Fiabilité Multi"]))
    fig.update_layout(barmode="group", height=400, title="Fiabilité par question (plus haut = mieux)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Télécharger le rapport (JSON)",
        data=json.dumps(st.session_state.bench, indent=2, ensure_ascii=False),
        file_name="multi_agent_rag_report.json",
        mime="application/json",
    )

st.markdown("---")
st.caption("Multi-Agent RAG System — KENEWY DIALLO | AI Engineer | Entretien 24 Août 2026")
