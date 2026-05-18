import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration de la page 
st.set_page_config(
    page_title="Prédiction Décrochage Universitaire",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1a56db;
    }
    div[data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.title("Navigation")
    st.markdown("---")

    # on laisse le responsable choisir quelle section voir
    section = st.radio(
        "Aller à :",
        ["Vue générale", "Étudiants à risque", "Performance modèle", "Simulation intervention", "Variables importantes"],
        index=0
    )

    st.markdown("---")
    st.caption("Ecole Nationale des TIC - ENASTIC")
    st.caption("2025-2026")
    st.caption("Random Forest + MLP")

# Chargement des fichiers avec gestion d'erreur 
@st.cache_data
def charger_donnees():
    try:
        features   = pd.read_csv("feature_importance.csv")
        simulation = pd.read_csv("simulation_intervention.csv")

        dropout = pd.read_csv("student_dropout_500.csv")

        at_risk = pd.read_csv("students_at_risk.csv")

        cols_risque = ["id","prob_decrochage","prob_diplome","prob_inscrit",
                       "niveau_risque","facteur_principal","pred_abandon"]
        cols_dispo = [c for c in cols_risque if c in at_risk.columns]

        if "id" in dropout.columns and "id" in at_risk.columns:
            dropout = dropout.merge(at_risk[cols_dispo], on="id", how="left")

        if "niveau_risque" not in dropout.columns or dropout["niveau_risque"].isna().all():
            if "prob_decrochage" in dropout.columns:
                def niveau(p):
                    if p >= 0.65: return "ÉLEVÉ 🔴"
                    if p >= 0.40: return "MOYEN 🟡"
                    return "FAIBLE 🟢"
                dropout["niveau_risque"] = dropout["prob_decrochage"].apply(niveau)
            else:
                dropout["niveau_risque"] = "Inconnu"

        return dropout, features, simulation, at_risk, None

    except FileNotFoundError as e:
        return None, None, None, None, str(e)

dropout, features, simulation, at_risk, erreur = charger_donnees()

if erreur:
    st.error(f"Fichier manquant : {erreur}")
    st.info("Vérifie que tous les fichiers CSV sont dans le même dossier que ce script.")
    st.stop()

st.title("🎓 Dashboard — Prédiction du Décrochage Scolaire")
st.caption("Tableau de bord du responsable de promotion · ENASTIC · 2025-2026")
st.divider()


# SECTION 1 — VUE GÉNÉRALE
if section == "Vue générale":
    st.subheader("Vue générale de la promotion")
    n_total  = len(dropout)

    n_eleve  = len(dropout[dropout["niveau_risque"].str.contains("ÉLEVÉ",  na=False)])
    n_moyen  = len(dropout[dropout["niveau_risque"].str.contains("MOYEN",  na=False)])
    n_faible = len(dropout[dropout["niveau_risque"].str.contains("FAIBLE", na=False)])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Total étudiants",
        value=n_total,
        delta="3 filières",
        delta_color="off"
    )
    col2.metric(
        label="Zone rouge 🔴",
        value=n_eleve,
        delta=f"{round(n_eleve/n_total*100, 1)}% de la promotion",
        delta_color="inverse" 
    )
    col3.metric(
        label="Zone orange 🟡",
        value=n_moyen,
        delta=f"{round(n_moyen/n_total*100, 1)}% de la promotion",
        delta_color="inverse"
    )
    col4.metric(
        label="Zone verte 🟢",
        value=n_faible,
        delta=f"{round(n_faible/n_total*100, 1)}% de la promotion",
        delta_color="normal"  
    )

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Répartition par statut d'abandon réel**")
        if "abandon" in dropout.columns:
            dist = dropout["abandon"].value_counts()
            couleurs = {"Diplômé": "#16a34a", "Décrochage": "#dc2626", "Inscrit": "#1a56db"}
            colors_list = [couleurs.get(k, "#94a3b8") for k in dist.index]

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            bars = ax1.bar(dist.index, dist.values, color=colors_list, edgecolor="white", linewidth=1.5)

            for bar, val in zip(bars, dist.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                         str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax1.set_ylabel("Nombre d'étudiants")
            ax1.set_ylim(0, dist.max() * 1.15)
            ax1.spines[['top', 'right']].set_visible(False)
            ax1.set_facecolor('#f8fafc')
            fig1.patch.set_facecolor('#f8fafc')
            st.pyplot(fig1)
            plt.close(fig1)

    with col_g2:
        st.markdown("**Répartition par niveau de risque**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        labels  = ['Zone rouge\n(risque élevé)', 'Zone orange\n(risque moyen)', 'Zone verte\n(faible risque)']
        valeurs = [n_eleve, n_moyen, n_faible]
        colors2 = ['#dc2626', '#d97706', '#16a34a']

        wedges, texts, autotexts = ax2.pie(
            valeurs, labels=labels, colors=colors2,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_fontweight('bold')

        ax2.set_facecolor('#f8fafc')
        fig2.patch.set_facecolor('#f8fafc')
        st.pyplot(fig2)
        plt.close(fig2)


# SECTION 2 — ÉTUDIANTS À RISQUE
elif section == "Étudiants à risque":
    st.subheader("Liste des étudiants — tous niveaux de risque")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        filtre_risque = st.selectbox(
            "Niveau de risque",
            options=["Tous", "ÉLEVÉ 🔴", "MOYEN 🟡", "FAIBLE 🟢"]
        )
    with col_f2:
        if "filiere" in dropout.columns:
            filieres = ["Toutes"] + sorted(dropout["filiere"].dropna().unique().tolist())
            filtre_filiere = st.selectbox("Filière", options=filieres)
        else:
            filtre_filiere = "Toutes"
    with col_f3:
        recherche = st.text_input("Rechercher un nom / prénom", placeholder="ex: Hamid")

    df_filtre = dropout.copy()

    if filtre_risque != "Tous":
        mot = filtre_risque.split()[0]
        df_filtre = df_filtre[df_filtre["niveau_risque"].str.contains(mot, na=False)]

    if filtre_filiere != "Toutes" and "filiere" in df_filtre.columns:
        df_filtre = df_filtre[df_filtre["filiere"] == filtre_filiere]

    if recherche:
        masque = (
            df_filtre.get("nom",    pd.Series(dtype=str)).astype(str).str.lower().str.contains(recherche.lower(), na=False) |
            df_filtre.get("prenom", pd.Series(dtype=str)).astype(str).str.lower().str.contains(recherche.lower(), na=False)
        )
        df_filtre = df_filtre[masque]

    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Affichés", len(df_filtre), f"sur {len(dropout)} total")
    col_r2.metric("Zone rouge 🔴", len(df_filtre[df_filtre["niveau_risque"].str.contains("ÉLEVÉ", na=False)]))
    col_r3.metric("Zone orange 🟡", len(df_filtre[df_filtre["niveau_risque"].str.contains("MOYEN", na=False)]))

    
    cols_afficher = [c for c in ["id","nom","prenom","age","sexe","filiere",
                                  "moyenne_sem1","moyenne_sem2","absences",
                                  "niveau_risque","facteur_principal","abandon"]
                     if c in df_filtre.columns]

    col_config = {}
    if "prob_decrochage" in df_filtre.columns:
        cols_afficher.append("prob_decrochage")
        col_config["prob_decrochage"] = st.column_config.ProgressColumn(
            "Probabilité décrochage", min_value=0, max_value=1, format="%.1f%%"
        )

    st.dataframe(
        df_filtre[cols_afficher],
        use_container_width=True,
        height=450,
        hide_index=True,
        column_config=col_config
    )


# SECTION 3 — PERFORMANCE DU MODÈLE
elif section == "Performance modèle":
    st.subheader("Performance des modèles de prédiction")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("Random Forest — calibré (Isotonic)")
        metriques_rf = {
            "Accuracy"              : ("75.2%", "bon"),
            "AUC (OVR pondéré)"    : ("0.862",  "bon"),
            "F1-score (CV 5 plis)" : ("0.732 ± 0.034", "bon"),
            "Log-Loss"             : ("0.668",  "moyen"),
            "Brier Score brut"     : ("0.0876", "bon"),
            "Brier Score calibré"  : ("0.0842 ✓", "bon"),
        }
        for nom, (val, niveau) in metriques_rf.items():
            icone = "🟢" if niveau == "bon" else "🟡"
            st.markdown(f"**{nom}** — {icone} `{val}`")

    with col_m2:
        st.markdown("Réseau de neurones MLP — calibré (Sigmoid)")
        metriques_mlp = {
            "Accuracy"              : ("72.8%", "moyen"),
            "AUC (OVR pondéré)"    : ("0.859",  "bon"),
            "F1-score (CV 3 plis)" : ("0.577 ± 0.016", "moyen"),
            "Log-Loss"             : ("0.674",  "moyen"),
            "Brier Score brut"     : ("0.0919", "moyen"),
            "Brier Score calibré"  : ("0.0873", "bon"),
        }
        for nom, (val, niveau) in metriques_mlp.items():
            icone = "🟢" if niveau == "bon" else "🟡"
            st.markdown(f"**{nom}** — {icone} `{val}`")

    st.divider()

    # rapport de classification
    st.markdown("#### Rapport de classification — Random Forest")
    rapport = pd.DataFrame({
        "Classe"    : ["Diplômé", "Décrochage", "Inscrit", "Moyenne pondérée"],
        "Précision" : [0.80, 0.86, 0.55, 0.75],
        "Rappel"    : [0.82, 0.82, 0.56, 0.75],
        "F1-score"  : [0.81, 0.84, 0.55, 0.75],
        "Support"   : [55, 38, 32, 125],
    })
    st.dataframe(rapport, use_container_width=True, hide_index=True)

    st.info("Le modèle Random Forest est retenu comme modèle principal avec un AUC de 0.862. La classe Inscrit est la plus difficile à prédire (F1=0.55), ce qui est attendu car c'est la catégorie la plus ambiguë.")


# SECTION 4 — SIMULATION INTERVENTION
elif section == "Simulation intervention":
    st.subheader("Simulation intervention / groupe témoin")

    st.markdown("""
    Les **127 étudiants en zone rouge** ont été divisés en deux groupes :
    - **Groupe intervention (63)** : accompagnement simulé — tuteur, accès LMS renforcé, suivi des devoirs
    - **Groupe témoin (64)** : aucun changement
    """)

    # résultats chiffrés
    col_i1, col_i2, col_i3 = st.columns(3)
    col_i1.metric("Réduction du risque moyen", "−18.4%", delta="groupe intervention", delta_color="normal")
    col_i2.metric("Étudiants sortis zone rouge", "21 / 63", delta="33% du groupe traité", delta_color="normal")
    col_i3.metric("Groupe témoin", "0%", delta="aucune amélioration", delta_color="off")

    st.divider()

    # graphique avant / après
    inter = simulation[simulation["groupe"] == "Intervention"].head(20)

    if len(inter) > 0 and "prob_avant" in inter.columns and "prob_apres" in inter.columns:
        st.markdown("**Avant / après intervention — groupe traité (20 premiers)**")

        noms = inter.apply(
            lambda r: f"{r.get('prenom','')} {r.get('nom','')}".strip(), axis=1
        )

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        x = range(len(inter))
        largeur = 0.35

        ax3.bar([i - largeur/2 for i in x], inter["prob_avant"],
                largeur, label="Avant intervention", color="#dc2626", alpha=0.85)
        ax3.bar([i + largeur/2 for i in x], inter["prob_apres"],
                largeur, label="Après intervention", color="#16a34a", alpha=0.85)

        ax3.axhline(65, color="red",    linestyle="--", linewidth=1.2, label="Seuil zone rouge (65%)")
        ax3.axhline(40, color="orange", linestyle="--", linewidth=1.2, label="Seuil zone orange (40%)")

        ax3.set_xticks(list(x))
        ax3.set_xticklabels(noms, rotation=45, ha="right", fontsize=9)
        ax3.set_ylabel("Probabilité de décrochage (%)")
        ax3.set_ylim(0, 110)
        ax3.legend(loc="upper right")
        ax3.spines[['top', 'right']].set_visible(False)
        ax3.set_facecolor('#f8fafc')
        fig3.patch.set_facecolor('#f8fafc')
        fig3.tight_layout()

        st.pyplot(fig3)
        plt.close(fig3)

    st.divider()
    st.markdown("**Données complètes de la simulation**")
    st.dataframe(simulation, use_container_width=True, height=350, hide_index=True)


# SECTION 5 — VARIABLES IMPORTANTES
elif section == "Variables importantes":
    st.subheader("Importance des variables — Random Forest")

    st.markdown("""
    Les variables sont classées par leur contribution à la prédiction.
    Les **variables rouges** sont des facteurs de risque.
    Les **variables vertes** sont des facteurs protecteurs.
    Le symbole **⚙** indique une variable construite par feature engineering.
    """)

    if features is not None and len(features) > 0:
        top = features.sort_values("importance", ascending=False).head(14)

        facteurs_risque      = ["absences", "retards", "ratio_absence", "score_risque"]
        facteurs_protection  = ["connexions_lms", "devoirs_rendus", "participation",
                                "taux_presence", "ratio_engagement", "moyenne"]

        def couleur(feat):
            if any(k in feat for k in facteurs_risque):     return "#dc2626"
            if any(k in feat for k in facteurs_protection): return "#16a34a"
            return "#1a56db"

        def label(feat):
            engineered = ["trend_notes", "ratio_engagement", "ratio_absence_connexion", "score_risque"]
            suffix = " ⚙" if feat in engineered else ""
            return feat + suffix

        colors = [couleur(f) for f in top["feature"]]
        labels = [label(f) for f in top["feature"]]

        fig4, ax4 = plt.subplots(figsize=(9, 6))
        bars = ax4.barh(labels, top["importance"], color=colors, edgecolor="white", linewidth=1)

        for bar, val in zip(bars, top["importance"]):
            ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va='center', fontsize=9)

        ax4.invert_yaxis()
        ax4.set_xlabel("Importance")
        ax4.spines[['top', 'right']].set_visible(False)
        ax4.set_facecolor('#f8fafc')
        fig4.patch.set_facecolor('#f8fafc')

        legende = [
            mpatches.Patch(color='#dc2626', label='Facteur de risque'),
            mpatches.Patch(color='#16a34a', label='Facteur protecteur'),
            mpatches.Patch(color='#1a56db', label='Variable construite / neutre'),
        ]
        ax4.legend(handles=legende, loc='lower right', fontsize=9)
        fig4.tight_layout()

        st.pyplot(fig4)
        plt.close(fig4)

        st.caption("⚙ = variable construite par feature engineering (trend_notes, ratio_engagement, ratio_absence_connexion, score_risque)")
