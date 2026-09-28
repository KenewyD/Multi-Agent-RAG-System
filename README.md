# 🕸️ Multi-Agent RAG System

> Système RAG multi-agent (architecture inspirée de LangGraph) pour améliorer la qualité
> et la fiabilité des réponses. Quatre agents spécialisés collaborent : **Router → Recherche → Synthèse → Vérification**.

---

## 🚀 Démo en ligne

👉 **[Voir la démo en direct](https://multi-agent-rag-system-ld2zdcme86mfsxjk447rh9.streamlit.app/))](https://multi-agent-rag-system-cmnrkdpudhu4p5scg6appx6.streamlit.app/)**

Chaque agent s'exécute **en direct** sous les yeux de l'utilisateur, avec un vrai calcul de
recherche vectorielle (similarité cosinus) — aucune donnée n'est écrite en dur.

---

## 🏗️ Architecture

| Agent | Rôle |
|-------|------|
| 📥 **Router** | Analyse la question et détecte son intention |
| 🔍 **Recherche** | Recherche vectorielle top-k dans la base de connaissances |
| 📝 **Synthèse** | Construit une réponse structurée à partir des passages récupérés |
| ✅ **Vérification** | Calcule un score de fiabilité (ancrage dans le contexte + pertinence) |

Le pipeline compare une approche **RAG simple** (un seul passage, pas de vérification) à
l'approche **multi-agent** (top-k + synthèse + vérification), et mesure la différence de fiabilité.

---

## 🚀 Lancer en local

```bash
git clone https://github.com/KenewyD/multi-agent-rag-system.git
cd multi-agent-rag-system
pip install -r requirements.txt
streamlit run app.py
```

Le dashboard s'ouvre sur `http://localhost:8501`.

---

## 🧠 Points techniques

- **Agents réels** : chaque agent exécute un traitement concret (routage par intention, recherche vectorielle TF-IDF, synthèse, vérification par similarité).
- **Métrique de fiabilité honnête** : combinaison de l'ancrage de la réponse dans le contexte récupéré et de sa pertinence vis-à-vis de la question.
- **Calcul en direct** : les scores sont recalculés à chaque exécution, rien n'est figé.
- **Déployable partout** : fonctionne sans OpenAI ni GPU (scikit-learn uniquement).

---

## 🛠️ Stack

`Python` · `Streamlit` · `scikit-learn` · `Plotly` · architecture multi-agent

---

## 📝 Auteur

**KENEWY DIALLO** — AI Engineer | LLM, RAG & AWS
🔗 [LinkedIn](https://linkedin.com/in/kenewy-diallo) | 💻 [GitHub](https://github.com/KenewyD)
