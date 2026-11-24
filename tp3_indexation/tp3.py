import math
import re
from collections import Counter, defaultdict


# ============================================================
# FIGL SEARCH ENGINE V2 - TP3 : Moteur de recherche TF-IDF
# ============================================================

# Les mots vides anglais sont retires avant la construction
# de l'index. Le meme pretraitement est applique aux requetes.
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


# ------------------------------------------------------------
# EXERCICE 1 - Chargement du corpus
# ------------------------------------------------------------

def charger_documents(nom_fichier="documents.tsv"):
    """Lit un fichier TSV : DocID, tabulation, texte."""
    documents = {}

    with open(nom_fichier, "r", encoding="utf-8") as fichier:
        for numero, ligne in enumerate(fichier, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue

            parties = ligne.split("\t", 1)
            if len(parties) != 2:
                print(f"Ligne {numero} ignoree : format TSV incorrect.")
                continue

            doc_id = parties[0].strip()
            contenu = parties[1].strip()

            if doc_id and contenu:
                documents[doc_id] = contenu

    return documents


# ------------------------------------------------------------
# EXERCICE 2 - Pretraitement
# ------------------------------------------------------------

def tokeniser(texte):
    """Met en minuscules et extrait les mots."""
    return re.findall(r"[a-z]+", texte.lower())


def stemmer_simple(mot):
    """
    Petit stemmer pedagogique pour illustrer model/models.
    Ce n'est pas un stemmer linguistique complet.
    """
    if len(mot) > 4 and mot.endswith("ies"):
        return mot[:-3] + "y"
    if len(mot) > 4 and mot.endswith("es"):
        return mot[:-2]
    if len(mot) > 3 and mot.endswith("s"):
        return mot[:-1]
    return mot


def preprocess(texte, use_stemming=False):
    """Tokenisation, suppression des stopwords et stemming optionnel."""
    tokens = tokeniser(texte)
    tokens = [mot for mot in tokens if mot not in STOPWORDS]

    if use_stemming:
        tokens = [stemmer_simple(mot) for mot in tokens]

    return tokens


def afficher_pipeline_exemples(documents, use_stemming=False):
    """Affiche les etapes du pretraitement pour trois documents."""
    print("\n--- Exemple du pipeline de pretraitement ---")
    for doc_id in list(documents.keys())[:3]:
        texte = documents[doc_id]
        tokens = tokeniser(texte)
        sans_stopwords = [t for t in tokens if t not in STOPWORDS]
        termes_finaux = preprocess(texte, use_stemming)

        print(f"\n{doc_id} - Texte original : {texte}")
        print("Tokens :", tokens)
        print("Apres suppression des stopwords :", sans_stopwords)
        print("Termes finalement indexes :", termes_finaux)


# ------------------------------------------------------------
# EXERCICE 12 - Moteur interactif
# ------------------------------------------------------------

def moteur_interactif(documents, index_inverse, vocabulary, N):
    print("\n========================================")
    print("       FIGL SEARCH ENGINE V2")
    print("========================================")
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

        choix = input("Nombre de resultats (3, 5 ou 10) [5] : ").strip()
        if choix in {"3", "5", "10"}:
            k = int(choix)
        else:
            k = 5

        resultats = search(
            query, documents, index_inverse, vocabulary, N, k=k
        )
        afficher_resultats(query, resultats)


# ------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ------------------------------------------------------------

# Utilise par afficher_poids_document pour afficher le texte.
documents_actuels = {}


def main():
    global documents_actuels

    try:
        documents = charger_documents("documents.tsv")
    except FileNotFoundError:
        print("Erreur : documents.tsv est introuvable.")
        print("Placez documents.tsv dans le meme dossier que main.py.")
        return

    if not documents:
        print("Le corpus est vide ou son format est incorrect.")
        return

    documents_actuels = documents
    N = len(documents)

    print("=" * 65)
    print("TP3 - MOTEUR DE RECHERCHE VECTORIEL TF-IDF")
    print("=" * 65)
    print("Nombre de documents :", N)

    # Indexation effectuee une seule fois au lancement.
    index_inverse = construire_index_enrichi(documents)
    vocabulary = sorted(index_inverse.keys())

    afficher_pipeline_exemples(documents)
    afficher_index_enrichi(index_inverse)
    afficher_table_idf(index_inverse, N)

    # Exemple de poids TF-IDF sur D8 si disponible.
    doc_exemple = "D08" if "D08" in documents else list(documents.keys())[0]
    afficher_poids_document(doc_exemple, index_inverse, N)

    # Vecteurs des trois premiers documents.
    print("\n--- VECTEURS DOCUMENTS ---")
    for doc_id in list(documents.keys())[:3]:
        afficher_vecteur_document(
            doc_id, vocabulary, index_inverse, N
        )

    # Exemple de vecteur de requete.
    requete_exemple = "information retrieval model"
    vecteur_q = query_vector(
        requete_exemple, vocabulary, index_inverse, N
    )
    print("\n--- VECTEUR DE LA REQUETE ---")
    print("Requete :", requete_exemple)
    print([
        (terme, poids)
        for terme, poids in zip(vocabulary, vecteur_q)
        if poids != 0
    ])

    # Cinq requetes de test.
    requetes_test = [
        "information retrieval model",
        "inverted index search",
        "TF IDF ranking documents",
        "dense retrieval embeddings",
        "boolean retrieval AND operators"
    ]

    print("\n--- CINQ REQUETES DE TEST (TOP-5) ---")
    for query in requetes_test:
        resultats = search(
            query, documents, index_inverse, vocabulary, N, k=5
        )
        afficher_resultats(query, resultats)

    # Comparaison boolenne / vectorielle.
    comparer_and_vectoriel(
        "information retrieval model",
        documents, index_inverse, vocabulary, N
    )

    # Experience de normalisation morphologique.
    experience_stemming(documents)

    # Mode interactif final.
    moteur_interactif(documents, index_inverse, vocabulary, N)
