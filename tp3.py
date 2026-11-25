
import math
import re
from collections import Counter, defaultdict





# ------------------------------------------------------------
# STOPWORDS
# ------------------------------------------------------------

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "for", "from", "had", "has", "have", "he", "her",
    "here", "hers", "him", "his", "i", "if", "in", "into", "is",
    "it", "its", "me", "more", "most", "my", "no", "not", "of",
    "on", "or", "our", "ours", "she", "so", "such", "than", "that",
    "the", "their", "theirs", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "under", "up", "us",
    "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your"
}


# ============================================================
# EXERCICE 1 : CHARGEMENT DU CORPUS
# ============================================================

def charger_documents(nom_fichier="documents.tsv"):
    """
    Charge les documents depuis un fichier TSV.
    Format : DocID [TAB] Texte
    """

    documents = {}

    with open(nom_fichier, "r", encoding="utf-8") as fichier:

        for numero, ligne in enumerate(fichier, start=1):

            ligne = ligne.strip()

            if not ligne:
                continue

            parties = ligne.split("\t", 1)

            if len(parties) != 2:
                print(f"Ligne {numero} ignoree : format incorrect.")
                continue

            doc_id = parties[0].strip()
            contenu = parties[1].strip()

            if doc_id and contenu:

                if doc_id in documents:
                    print(f"Attention : identifiant {doc_id} duplique.")
                    continue

                documents[doc_id] = contenu

    return documents


# ============================================================
# EXERCICE 2 : PRETRAITEMENT
# ============================================================

def tokeniser(texte):
    """
    Convertit le texte en minuscules et extrait les mots.
    """

    return re.findall(r"[a-z]+", texte.lower())


def stemmer_simple(mot):
    """
    Stemmer pédagogique simplifié.
    Permet notamment de rapprocher model et models.
    Ce n'est pas un stemmer linguistique complet.
    """

    if len(mot) > 4 and mot.endswith("ies"):
        return mot[:-3] + "y"

    if len(mot) > 5 and mot.endswith(
        ("ches", "shes", "xes", "zes", "sses", "oes")
    ):
        return mot[:-2]

    if len(mot) > 3 and mot.endswith("s"):
        return mot[:-1]

    return mot


def preprocess(texte, use_stemming=False):
    """
    Pipeline :
    1. Minuscules
    2. Tokenisation
    3. Suppression des stopwords
    4. Stemming optionnel
    """

    tokens = tokeniser(texte)

    tokens = [
        mot for mot in tokens
        if mot not in STOPWORDS
    ]

    if use_stemming:
        tokens = [
            stemmer_simple(mot)
            for mot in tokens
        ]

    return tokens


def afficher_pipeline_exemples(documents, use_stemming=False):
    """
    Affiche les étapes du prétraitement
    pour les trois premiers documents.
    """

    print("\n" + "=" * 65)
    print("EXERCICE 2 : PIPELINE DE PRETRAITEMENT")
    print("=" * 65)

    for doc_id in list(documents.keys())[:3]:

        texte = documents[doc_id]

        tokens = tokeniser(texte)

        sans_stopwords = [
            mot for mot in tokens
            if mot not in STOPWORDS
        ]

        termes_finaux = preprocess(
            texte, use_stemming
        )

        print(f"\nDocument : {doc_id}")
        print("Texte original :", texte)
        print("Tokens :", tokens)
        print("Apres suppression des stopwords :", sans_stopwords)
        print("Termes finalement indexes :", termes_finaux)


# ============================================================
# EXERCICE 3 : INDEX INVERSE ENRICHI
# ============================================================

def construire_index_enrichi(documents, use_stemming=False):
    """
    Construit un index inversé contenant :
    - DF : nombre de documents contenant le terme.
    - Postings : documents contenant le terme.
    - TF : fréquence du terme dans chaque document.
    """

    index_inverse = defaultdict(
        lambda: {
            "df": 0,
            "postings": {}
        }
    )

    for doc_id, texte in documents.items():

        termes = preprocess(
            texte, use_stemming
        )

        frequences = Counter(termes)

        for terme, tf in frequences.items():

            index_inverse[terme]["postings"][doc_id] = {
                "tf": tf
            }

            index_inverse[terme]["df"] += 1

    return dict(index_inverse)


def afficher_index_enrichi(index_inverse):

    print("\n" + "=" * 65)
    print("EXERCICE 3 : INDEX INVERSE ENRICHI")
    print("=" * 65)

    print(f"{'TERM':<20} {'DF':<8} POSTINGS")

    print("-" * 80)

    for terme in sorted(index_inverse):

        df = index_inverse[terme]["df"]

        postings = index_inverse[terme]["postings"]

        liste = ", ".join(
            f"{doc_id}(tf={info['tf']})"
            for doc_id, info in postings.items()
        )

        print(f"{terme:<20} {df:<8} {liste}")


# ============================================================
# EXERCICE 4 : CALCUL DE IDF
# ============================================================

def calculer_idf(terme, index_inverse, N):

    if terme not in index_inverse or N <= 0:
        return 0.0

    df = index_inverse[terme]["df"]

    if df == 0:
        return 0.0

    return math.log10(N / df)


def afficher_table_idf(index_inverse, N):

    print("\n" + "=" * 65)
    print("EXERCICE 4 : TABLEAU DF / IDF")
    print("=" * 65)

    print(f"{'TERME':<20} {'DF':<10} {'IDF':<15}")

    print("-" * 50)

    statistiques = []

    for terme in sorted(index_inverse):

        df = index_inverse[terme]["df"]

        idf = calculer_idf(
            terme, index_inverse, N
        )

        statistiques.append((terme, df, idf))

        print(f"{terme:<20} {df:<10} {idf:.5f}")

    if statistiques:

        terme_max_df = max(
            statistiques, key=lambda x: x[1]
        )

        terme_min_df = min(
            statistiques, key=lambda x: x[1]
        )

        terme_max_idf = max(
            statistiques, key=lambda x: x[2]
        )

        print("\nAnalyse des statistiques :")

        print(
            "DF maximal :",
            terme_max_df[0],
            "DF =", terme_max_df[1]
        )

        print(
            "DF minimal :",
            terme_min_df[0],
            "DF =", terme_min_df[1]
        )

        print(
            "IDF maximal :",
            terme_max_idf[0],
            "IDF =", round(terme_max_idf[2], 5)
        )


# ============================================================
# EXERCICE 5 : CALCUL DES POIDS TF-IDF
# ============================================================

def poids_tfidf(terme, doc_id, index_inverse, N):

    if terme not in index_inverse:
        return 0.0

    postings = index_inverse[terme]["postings"]

    if doc_id not in postings:
        return 0.0

    tf = postings[doc_id]["tf"]

    idf = calculer_idf(
        terme, index_inverse, N
    )

    poids = (1 + math.log10(tf)) * idf

    return poids


def afficher_poids_document(doc_id, index_inverse, N):

    print("\n" + "=" * 65)
    print(f"EXERCICE 5 : POIDS TF-IDF DU DOCUMENT {doc_id}")
    print("=" * 65)

    resultats = []

    for terme, donnees in index_inverse.items():

        postings = donnees["postings"]

        if doc_id in postings:

            tf = postings[doc_id]["tf"]

            df = donnees["df"]

            idf = calculer_idf(
                terme, index_inverse, N
            )

            poids = poids_tfidf(
                terme, doc_id, index_inverse, N
            )

            resultats.append(
                (terme, tf, df, idf, poids)
            )

    resultats.sort(
        key=lambda x: x[4],
        reverse=True
    )

    print(
        f"{'TERM':<20} {'TF':<6} {'DF':<6} "
        f"{'IDF':<12} {'TF-IDF':<12}"
    )

    print("-" * 65)

    for terme, tf, df, idf, poids in resultats:

        print(
            f"{terme:<20} {tf:<6} {df:<6} "
            f"{idf:<12.5f} {poids:<12.5f}"
        )


# ============================================================
# EXERCICE 6 : VECTEURS DOCUMENTS
# ============================================================

def document_vector(doc_id, vocabulary, index_inverse, N):
    """
    Construit le vecteur TF-IDF d'un document
    dans l'ordre du vocabulaire global.
    """

    vector = []

    for terme in vocabulary:

        poids = poids_tfidf(
            terme, doc_id, index_inverse, N
        )

        vector.append(poids)

    return vector


def afficher_vecteur_document(
    doc_id, vocabulary, index_inverse, N
):

    vector = document_vector(
        doc_id, vocabulary, index_inverse, N
    )

    nombre_nuls = sum(
        1 for valeur in vector
        if valeur == 0
    )

    print(f"\nDocument : {doc_id}")

    print("Vecteur TF-IDF :")

    print([
        (terme, round(poids, 5))
        for terme, poids in zip(vocabulary, vector)
        if poids != 0
    ])

    print("Nombre total de composantes :", len(vector))

    print("Nombre de composantes nulles :", nombre_nuls)


# ============================================================
# EXERCICE 7 : VECTEUR DE REQUETE
# ============================================================

def query_vector(
    query, vocabulary, index_inverse, N,
    use_stemming=False
):
    """
    Construit le vecteur TF-IDF d'une requête.
    Les termes absents du vocabulaire reçoivent un poids nul.
    """

    tokens = preprocess(
        query, use_stemming
    )

    frequences = Counter(tokens)

    vector_q = []

    for terme in vocabulary:

        tf = frequences.get(terme, 0)

        if tf > 0:

            idf = calculer_idf(
                terme, index_inverse, N
            )

            poids = (1 + math.log10(tf)) * idf

        else:
            poids = 0.0

        vector_q.append(poids)

    return vector_q


# ============================================================
# EXERCICE 8 : SIMILARITE COSINUS
# ============================================================

def cosine_similarity(v1, v2):
    """
    Calcule manuellement la similarité cosinus.
    Aucun recours à sklearn.
    """

    if len(v1) != len(v2):
        raise ValueError(
            "Les vecteurs doivent avoir la meme dimension."
        )

    # Produit scalaire
    produit_scalaire = sum(
        a * b for a, b in zip(v1, v2)
    )

    # Norme du premier vecteur
    norme1 = math.sqrt(
        sum(a ** 2 for a in v1)
    )

    # Norme du deuxième vecteur
    norme2 = math.sqrt(
        sum(b ** 2 for b in v2)
    )

    # Gestion des vecteurs nuls
    if norme1 == 0 or norme2 == 0:
        return 0.0

    score = produit_scalaire / (norme1 * norme2)

    return score


# ============================================================
# EXERCICE 9 : RECHERCHE ET RANKING TOP-K
# ============================================================

def search(
    query, documents, index_inverse,
    vocabulary, N, k=5, use_stemming=False
):
    """
    Recherche les documents les plus pertinents.
    """

    vecteur_q = query_vector(
        query,
        vocabulary,
        index_inverse,
        N,
        use_stemming
    )

    resultats = []

    for doc_id in documents:

        vecteur_d = document_vector(
            doc_id,
            vocabulary,
            index_inverse,
            N
        )

        score = cosine_similarity(
            vecteur_q, vecteur_d
        )

        # Eliminer les scores nuls
        if score > 0:

            resultats.append(
                (doc_id, score)
            )

    # Trier par score décroissant
    resultats.sort(
        key=lambda x: (-x[1], x[0])
    )

    return resultats[:k]


def afficher_resultats(
    query, resultats, documents=None
):

    print("\n" + "=" * 65)
    print("REQUETE :", query)
    print("=" * 65)

    print("TOP RESULTS")

    if not resultats:
        print("Aucun document pertinent trouve.")
        return

    print(
        f"{'RANG':<8} {'DOCID':<12} {'SCORE':<12} DOCUMENT"
    )

    print("-" * 90)

    for rang, (doc_id, score) in enumerate(
        resultats, start=1
    ):

        texte = (
            documents[doc_id]
            if documents is not None
            else ""
        )

        print(
            f"{rang:<8} {doc_id:<12} "
            f"{score:<12.5f} {texte}"
        )


# ============================================================
# EXERCICE 10 : EXPERIENCE MODEL / MODELS
# ============================================================

def experience_stemming(documents):

    print("\n" + "=" * 65)
    print("EXERCICE 10 : EXPERIENCE MODEL / MODELS")
    print("=" * 65)

    requete = "information retrieval model"

    # Identifier les documents contenant model ou models
    documents_model = []

    for doc_id, texte in documents.items():

        tokens = tokeniser(texte)

        if "model" in tokens or "models" in tokens:
            documents_model.append(doc_id)

    print(
        "Documents contenant model ou models :",
        documents_model
    )

    # EXPERIENCE A : sans stemming
    index_normal = construire_index_enrichi(
        documents, use_stemming=False
    )

    vocabulaire_normal = sorted(
        index_normal.keys()
    )

    resultats_normal = search(
        requete,
        documents,
        index_normal,
        vocabulaire_normal,
        len(documents),
        k=len(documents),
        use_stemming=False
    )

    # EXPERIENCE B : avec stemming
    index_stem = construire_index_enrichi(
        documents, use_stemming=True
    )

    vocabulaire_stem = sorted(
        index_stem.keys()
    )

    resultats_stem = search(
        requete,
        documents,
        index_stem,
        vocabulaire_stem,
        len(documents),
        k=len(documents),
        use_stemming=True
    )

    print("\nComparaison des scores :")

    print(
        f"{'DOCUMENT':<15} {'SANS STEMMING':<20} "
        f"{'AVEC STEMMING':<20}"
    )

    print("-" * 60)

    for doc_id in documents_model:

        score_normal = next(
            (
                score for d, score in resultats_normal
                if d == doc_id
            ),
            0.0
        )

        score_stem = next(
            (
                score for d, score in resultats_stem
                if d == doc_id
            ),
            0.0
        )

        print(
            f"{doc_id:<15} {score_normal:<20.5f} "
            f"{score_stem:<20.5f}"
        )

    print("\nComparaison des rangs :")

    for doc_id in documents_model:

        rang_normal = next(
            (
                i for i, (d, score)
                in enumerate(resultats_normal, start=1)
                if d == doc_id
            ),
            "Non classe"
        )

        rang_stem = next(
            (
                i for i, (d, score)
                in enumerate(resultats_stem, start=1)
                if d == doc_id
            ),
            "Non classe"
        )

        print(
            f"{doc_id} : sans stemming = {rang_normal}, "
            f"avec stemming = {rang_stem}"
        )

    print("\nTOP 5 sans stemming :")

    afficher_resultats(
        requete, resultats_normal[:5], documents
    )

    print("\nTOP 5 avec stemming :")

    afficher_resultats(
        requete, resultats_stem[:5], documents
    )


# ============================================================
# EXERCICE 11 : BOOLEEN AND VS TF-IDF
# ============================================================

def recherche_booleenne_and(
    query, index_inverse, use_stemming=False
):
    """
    Recherche booléenne AND par intersection
    des listes de postings.
    """

    termes = preprocess(
        query, use_stemming
    )

    termes = list(dict.fromkeys(termes))

    if not termes:
        return []

    # Si un terme n'existe pas, l'intersection est vide.
    if any(
        terme not in index_inverse
        for terme in termes
    ):
        return []

    listes_postings = [
        set(
            index_inverse[terme]["postings"].keys()
        )
        for terme in termes
    ]

    resultats = set.intersection(
        *listes_postings
    )

    return sorted(resultats)


def comparer_and_vectoriel(
    query, documents, index_inverse,
    vocabulary, N
):

    print("\n" + "=" * 65)
    print("EXERCICE 11 : BOOLEEN AND VS VECTORIEL TF-IDF")
    print("=" * 65)

    print("Requete :", query)

    # Mode A : recherche booléenne AND
    resultats_and = recherche_booleenne_and(
        query, index_inverse
    )

    # Mode B : recherche vectorielle
    resultats_vectoriels = search(
        query,
        documents,
        index_inverse,
        vocabulary,
        N,
        k=len(documents)
    )

    print("\nMODE A : BOOLEEN AND")

    if resultats_and:

        for doc_id in resultats_and:
            print(doc_id, ":", documents[doc_id])

    else:
        print("Aucun document retrouve.")

    print("\nMODE B : VECTORIEL TF-IDF")

    afficher_resultats(
        query, resultats_vectoriels, documents
    )

    print("\nTABLEAU COMPARATIF")

    print(
        f"{'CRITERE':<30} "
        f"{'BOOLEEN AND':<25} "
        f"{'TF-IDF':<25}"
    )

    print("-" * 80)

    print(
        f"{'Documents retrouves':<30} "
        f"{len(resultats_and):<25} "
        f"{len(resultats_vectoriels):<25}"
    )

    print(
        f"{'Correspondance partielle':<30} "
        f"{'Non':<25} {'Oui':<25}"
    )

    print(
        f"{'Score de pertinence':<30} "
        f"{'Non':<25} {'Cosinus':<25}"
    )

    print(
        f"{'Classement':<30} "
        f"{'Non':<25} {'Decroissant':<25}"
    )


# ============================================================
# EXERCICE 12 : MOTEUR INTERACTIF
# ============================================================

def moteur_interactif(
    documents, index_inverse, vocabulary, N
):

    print("\n" + "=" * 65)
    print("       FIGL SEARCH ENGINE V2")
    print("=" * 65)

    print("Documents indexed :", N)

    print("Vocabulary size :", len(vocabulary))

    print("Tapez 'quit' pour quitter.")

    while True:

        query = input("\nQuery > ").strip()

        if query.lower() in {"quit", "exit", "q"}:

            print("Fermeture du moteur de recherche.")

            break

        if not query:

            print("Veuillez saisir une requete.")

            continue

        choix = input(
            "Nombre de resultats (3, 5 ou 10) [5] : "
        ).strip()

        if choix in {"3", "5", "10"}:

            k = int(choix)

        else:

            k = 5

        resultats = search(
            query,
            documents,
            index_inverse,
            vocabulary,
            N,
            k=k
        )

        afficher_resultats(
            query, resultats, documents
        )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    try:

        documents = charger_documents(
            "documents.tsv"
        )

    except FileNotFoundError:

        print("Erreur : documents.tsv est introuvable.")

        print(
            "Placez documents.tsv dans le meme dossier que main.py."
        )

        return

    if not documents:

        print("Le corpus est vide ou son format est incorrect.")

        return

    N = len(documents)

    print("=" * 65)

    print("TP3 - MOTEUR DE RECHERCHE VECTORIEL TF-IDF")

    print("=" * 65)

    print("Nombre de documents :", N)

    if N < 20:

        print(
            "Attention : le corpus doit contenir au moins 20 documents."
        )

    # Construction de l'index une seule fois
    index_inverse = construire_index_enrichi(
        documents
    )

    vocabulary = sorted(
        index_inverse.keys()
    )

    # Exercice 2
    afficher_pipeline_exemples(documents)

    # Exercice 3
    afficher_index_enrichi(index_inverse)

    # Exercice 4
    afficher_table_idf(index_inverse, N)

    # Exercice 5
    doc_exemple = (
        "D8" if "D8" in documents
        else list(documents.keys())[0]
    )

    afficher_poids_document(
        doc_exemple, index_inverse, N
    )

    # Exercice 6
    print("\n" + "=" * 65)

    print("EXERCICE 6 : VECTEURS DOCUMENTS")

    print("=" * 65)

    for doc_id in list(documents.keys())[:3]:

        afficher_vecteur_document(
            doc_id, vocabulary, index_inverse, N
        )

    # Exercice 7
    requete_exemple = "information retrieval model"

    vecteur_q = query_vector(
        requete_exemple,
        vocabulary,
        index_inverse,
        N
    )

    print("\n" + "=" * 65)

    print("EXERCICE 7 : VECTEUR DE LA REQUETE")

    print("=" * 65)

    print("Requete :", requete_exemple)

    print([
        (terme, round(poids, 5))
        for terme, poids in zip(vocabulary, vecteur_q)
        if poids != 0
    ])

    # Exercice 8
    print("\n" + "=" * 65)

    print("EXERCICE 8 : SIMILARITE COSINUS")

    print("=" * 65)

    print(
        "Exemple de similarite :",
        cosine_similarity(vecteur_q, vecteur_q)
    )

    # Exercice 9 : cinq requêtes de test
    requetes_test = [
        "information retrieval model",
        "inverted index search",
        "TF IDF ranking documents",
        "dense retrieval embeddings",
        "boolean retrieval AND operators"
    ]

    print("\n" + "=" * 65)

    print("EXERCICE 9 : CINQ REQUETES DE TEST TOP-5")

    print("=" * 65)

    for query in requetes_test:

        resultats = search(
            query,
            documents,
            index_inverse,
            vocabulary,
            N,
            k=5
        )

        afficher_resultats(
            query, resultats, documents
        )

    # Exercice 10
    experience_stemming(documents)

    # Exercice 11
    comparer_and_vectoriel(
        "information retrieval model",
        documents,
        index_inverse,
        vocabulary,
        N
    )

    # Exercice 12
    moteur_interactif(
        documents,
        index_inverse,
        vocabulary,
        N
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
