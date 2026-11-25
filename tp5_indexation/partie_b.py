
# -*- coding: utf-8 -*-
"""
TP5 : Saturation de TF dans BM25
Partie B : B1, B2, B3, B4 et B5

Le corpus est chargé depuis le fichier corpus.tsv.
"""

import math


# =====================================================
# B1 : CHARGEMENT DU CORPUS
# =====================================================

def charger_corpus(nom_fichier):
    """
    Charge les documents depuis un fichier TSV.
    Chaque ligne contient :
    DocID [tabulation] texte du document
    """

    corpus = {}

    with open(nom_fichier, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne = ligne.strip()

            if ligne:
                doc_id, texte = ligne.split("\t", 1)
                corpus[doc_id] = texte

    return corpus


CORPUS = charger_corpus("corpus.tsv")

# Tokenisation demandée : texte.split()
documents = {
    doc_id: texte.split()
    for doc_id, texte in CORPUS.items()
}

N = len(documents)

# Requête
termes_requete = ["blockchain", "sécurité"]


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def calculer_df(documents):
    """Calcule le nombre de documents contenant chaque terme."""

    df = {}

    for tokens in documents.values():
        termes_uniques = set(tokens)

        for terme in termes_uniques:
            df[terme] = df.get(terme, 0) + 1

    return df


def calculer_avgdl(documents):
    """Calcule la longueur moyenne des documents."""

    if not documents:
        return 0

    total = sum(len(tokens) for tokens in documents.values())
    return total / len(documents)


def calculer_idf_bm25(df, N):
    """Calcule l'IDF de BM25."""

    idf = {}

    for terme, frequence in df.items():
        idf[terme] = math.log(
            1 + (N - frequence + 0.5) / (frequence + 0.5)
        )

    return idf


def tf_sat(tf, k1=1.2, b=0.75, dl=None, avgdl=None):
    """Calcule la fréquence saturée de BM25."""

    if dl is None or avgdl is None:
        K = k1
    else:
        K = k1 * (1 - b + b * dl / avgdl)

    if tf == 0:
        return 0.0

    return tf * (k1 + 1) / (tf + K)


# =====================================================
# CALCUL DES SCORES TF-IDF
# =====================================================

def calculer_tfidf(documents, termes_requete):
    """
    Calcule un score TF-IDF simple pour chaque document.

    TF = 1 + log(tf) si tf > 0
    IDF = log(N / df)
    Le score est la somme des poids des termes de la requête.
    """

    N = len(documents)
    df = calculer_df(documents)
    scores = {}

    for doc_id, tokens in documents.items():

        score = 0.0

        for terme in termes_requete:

            tf = tokens.count(terme)

            if tf > 0 and df.get(terme, 0) > 0:
                tf_pondere = 1 + math.log(tf)
                idf = math.log(N / df[terme])

                score += tf_pondere * idf

        scores[doc_id] = score

    return scores


# =====================================================
# CALCUL DES SCORES BM25
# =====================================================

def calculer_bm25(
    documents,
    termes_requete,
    k1=1.2,
    b=0.75,
    avgdl_personnalise=None
):
    """Calcule le score BM25 de chaque document."""

    N = len(documents)
    df = calculer_df(documents)
    idf = calculer_idf_bm25(df, N)

    if avgdl_personnalise is None:
        avgdl = calculer_avgdl(documents)
    else:
        avgdl = avgdl_personnalise

    scores = {}

    for doc_id, tokens in documents.items():

        dl = len(tokens)
        score = 0.0

        for terme in termes_requete:

            tf = tokens.count(terme)

            if tf > 0:
                score += idf[terme] * tf_sat(
                    tf,
                    k1=k1,
                    b=b,
                    dl=dl,
                    avgdl=avgdl
                )

        scores[doc_id] = score

    return scores


def afficher_classement(scores, titre):
    """Affiche les documents du meilleur score au plus faible."""

    print("\n", titre)
    print("-" * 45)

    classement = sorted(
        scores.items(),
        key=lambda element: element[1],
        reverse=True
    )

    for rang, (doc_id, score) in enumerate(classement, start=1):
        print(f"{rang}. {doc_id} : {score:.4f}")

    return classement


# =====================================================
# B2 : COMPARAISON TF-IDF ET BM25
# =====================================================

print("\n========== B2 : COMPARAISON TF-IDF ET BM25 ==========")

avgdl = calculer_avgdl(documents)
df = calculer_df(documents)
idf = calculer_idf_bm25(df, N)

print("\nNombre de documents :", N)
print("Longueur moyenne avgdl :", round(avgdl, 4))

print("\nLongueur de chaque document :")

for doc_id, tokens in documents.items():
    print(f"{doc_id} : {len(tokens)} mots")

print("\nIDF des termes de la requête :")

for terme in termes_requete:
    print(f"IDF({terme}) = {idf.get(terme, 0):.4f}")

# Scores TF-IDF
scores_tfidf = calculer_tfidf(
    documents,
    termes_requete
)

classement_tfidf = afficher_classement(
    scores_tfidf,
    "Classement TF-IDF"
)

# Scores BM25
scores_bm25 = calculer_bm25(
    documents,
    termes_requete,
    k1=1.2,
    b=0.75
)

classement_bm25 = afficher_classement(
    scores_bm25,
    "Classement BM25 (k1=1.2, b=0.75)"
)


# =====================================================
# B3 : INFLUENCE DE k1
# =====================================================

print("\n========== B3 : INFLUENCE DE k1 ==========")

liste_k1 = [0.01, 0.5, 1.2, 2, 5, 10, 50]

for k1 in liste_k1:

    scores = calculer_bm25(
        documents,
        termes_requete,
        k1=k1,
        b=0.75
    )

    classement = afficher_classement(
        scores,
        f"Classement BM25 pour k1 = {k1}"
    )

    position_D1 = next(
        rang
        for rang, (doc_id, score) in enumerate(classement, start=1)
        if doc_id == "D1"
    )

    position_D2 = next(
        rang
        for rang, (doc_id, score) in enumerate(classement, start=1)
        if doc_id == "D2"
    )

    if position_D1 < position_D2:
        print("D1 passe devant D2 pour k1 =", k1)
    else:
        print("D1 reste derrière D2 pour k1 =", k1)


# =====================================================
# B4 : INFLUENCE DU KEYWORD STUFFING
# =====================================================

print("\n========== B4 : KEYWORD STUFFING ==========")

liste_n = [1, 5, 20, 100, 1000]

k1 = 1.2
b = 0.75

# IDF de blockchain calculé sur le corpus initial.
# La fréquence documentaire reste inchangée :
# D1 et D2 contiennent toujours blockchain.
df_initial = calculer_df(documents)
idf_initial = calculer_idf_bm25(df_initial, N)

idf_blockchain = idf_initial["blockchain"]

borne_theorique = idf_blockchain * (k1 + 1)

print("IDF(blockchain) =", round(idf_blockchain, 4))
print("Borne théorique de D1 =", round(borne_theorique, 4))

print(
    f"\n{'n':<10}{'avgdl':<15}"
    f"{'Score D1':<20}{'Score D2':<20}"
)

for n in liste_n:

    # Copie du corpus initial
    documents_test = {
        doc_id: tokens.copy()
        for doc_id, tokens in documents.items()
    }

    # Remplacer D1 par n répétitions de blockchain
    documents_test["D1"] = ["blockchain"] * n

    # Recalculer avgdl avec le nouveau D1
    nouvel_avgdl = calculer_avgdl(documents_test)

    # Recalculer les scores BM25
    scores = calculer_bm25(
        documents_test,
        termes_requete,
        k1=k1,
        b=b,
        avgdl_personnalise=nouvel_avgdl
    )

    print(
        f"{n:<10}{nouvel_avgdl:<15.4f}"
        f"{scores['D1']:<20.4f}{scores['D2']:<20.4f}"
    )


# =====================================================
# B5 : INFLUENCE DE b
# =====================================================

print("\n========== B5 : INFLUENCE DE b ==========")

liste_b = [0, 0.75, 1]

for b in liste_b:

    print(f"\nRésultats pour b = {b}")

    for k1 in liste_k1:

        scores = calculer_bm25(
            documents,
            termes_requete,
            k1=k1,
            b=b
        )

        classement = sorted(
            scores.items(),
            key=lambda element: element[1],
            reverse=True
        )

        print(
            f"k1 = {k1:<6} "
            f"Classement : "
            + " > ".join(doc_id for doc_id, score in classement)
        )

print("\n========== FIN DU TP5 ==========")
