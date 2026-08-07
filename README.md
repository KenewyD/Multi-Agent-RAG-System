# 🕸️ Multi-Agent RAG System

> Système multi-agent inspiré de LangGraph pour la réduction des hallucinations dans les pipelines RAG.  
> Dashboard comparatif : RAG Simple vs Architecture Multi-Agent.

---

## 🎯 Objectif

Démontrer l'impact d'une architecture **multi-agent** (Router → Recherche → Synthèse → Vérification) sur la qualité des réponses générées par un système RAG.

**Résultat clé :** Réduction des hallucinations de **~80%** par rapport à un RAG classique.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Question  │────▶│    Router   │────▶│  Recherche  │────▶│   Synthèse  │────▶│ Vérification│
│   Utilisateur│     │   (Routing) │     │  (Retrieval)│     │ (Generation)│     │(Fact-Check) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                                        │
                                                                                        ▼
                                                                                 ┌─────────────┐
                                                                                 │   Réponse   │
                                                                                 │   Finale    │
                                                                                 └─────────────┘
```

### Agents

| Agent | Rôle | Technologie |
|-------|------|-------------|
| **Router** | Analyse la question et choisit la stratégie | LangGraph StateGraph |
| **Recherche** | Récupère les documents pertinents (top-k) | FAISS + Embeddings |
| **Synthèse** | Génère une réponse structurée avec citations | Claude-3 / GPT-4 |
| **Vérification** | Détecte les hallucinations et corrige | Similarité cosinus + Règles |

---

## 📊 Résultats

| Métrique | RAG Simple | Multi-Agent | Gain |
|----------|-----------|-------------|------|
| Hallucination moyenne | 0.38 | 0.08 | **-79%** |
| Fiabilité moyenne | 62% | 92% | **+48%** |
| Latence moyenne | 430ms | 848ms | +97% |

> 💡 **Trade-off** : La latence augmente (~2x) mais la qualité est massivement améliorée. Adapté aux cas où la précision prime sur la vitesse.

---

## 🚀 Déploiement

### Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud
1. Push sur GitHub
2. Connecter le repo sur [share.streamlit.io](https://share.streamlit.io)
3. Déployer

---

---

**Auteur :** KENEWY DIALLO — AI Engineer | LLM, RAG & AWS
