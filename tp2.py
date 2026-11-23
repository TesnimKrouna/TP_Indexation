import re
import math
 
# ---------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------
documents = {}
 
stopwords = {
    "le", "la", "les", "un", "une",
    "de", "des", "du", "et", "dans",
    "par", "pour", "d", "l"
}
 

with open("documents.tsv", "r", encoding="utf-8") as fichier:
    for ligne in fichier:
        ligne = ligne.strip()
        if not ligne:
            continue
        doc_id, contenu = ligne.split("\t", 1)
        documents[doc_id] = contenu
 
print("=" * 70)
print(f"Nombre de documents chargés : {len(documents)}")
for doc_id, contenu in documents.items():
    print(doc_id, ":", contenu)
 
 
# ---------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------
def tokeniser(texte):
    """Met le texte en minuscules et extrait les mots. Complexité O(L)."""
    return re.findall(r"\b\w+\b", texte.lower())
 
 
def supprimer_stopwords(tokens):
    """Supprime les mots vides d'une liste de tokens. Complexité O(L)."""
    resultat = []
    for token in tokens:
        if token not in stopwords:
            resultat.append(token)
    return resultat
 
 
def preprocess(texte):
    """Tokenisation + suppression des mots vides. Complexité O(L)."""
    tokens = tokeniser(texte)
    tokens = supprimer_stopwords(tokens)
    return tokens
 
 
# ---------------------------------------------------------------------
# Construction de l'index inversé
# ---------------------------------------------------------------------
def cle_tri_docid(doc_id):
    """
    Clé de tri numérique pour des identifiants du type 'D1', 'D10', 'D2'...
    Sans cette clé, le tri par défaut (alphabétique) placerait "D10" avant
    "D2", ce qui casse l'algorithme d'intersection (qui suppose des listes
    triées dans le même ordre que les comparaisons '<').
    Si l'identifiant ne contient pas de partie numérique, on le laisse
    tel quel (tri alphabétique classique en repli).
    """
    partie_numerique = re.sub(r"\D", "", doc_id)
    return int(partie_numerique) if partie_numerique else doc_id
 
 
def construire_index(documents):
    """
    Construit l'index inversé {terme: [liste triée de DocID]}.
    Complexité : O(N * L) pour la construction, où N = nombre de
    documents et L = longueur moyenne d'un document, plus O(V * K log K)
    pour le tri final des listes de postings (V = nombre de termes,
    K = longueur moyenne d'une liste de postings).
    """
    index_inverse = {}
 
    for doc_id, texte in documents.items():
        termes = preprocess(texte)
        for terme in set(termes):
            if terme not in index_inverse:
                index_inverse[terme] = []
            index_inverse[terme].append(doc_id)
 
    # Tri numérique de chaque liste de postings (correction du bug D1/D10)
    for terme in index_inverse:
        index_inverse[terme].sort(key=cle_tri_docid)
 
    return index_inverse
 
 
def rechercher(terme, index_inverse):
    """Accès direct à la liste de postings d'un terme. Complexité O(1)."""
    return index_inverse.get(terme.lower(), [])
 
 
# ---------------------------------------------------------------------
# Intersection classique (sans skip pointers)
# ---------------------------------------------------------------------
def intersection(list1, list2):
    """
    Intersection de deux listes triées, algorithme de fusion.
    Complexité : O(len(list1) + len(list2)).
    """
    i, j = 0, 0
    resultat = []
 
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            resultat.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1
 
    return resultat
 
 
# ---------------------------------------------------------------------
# Skip pointers (exercice bonus)
# ---------------------------------------------------------------------
def creer_skips(liste):
    """
    Construit un dictionnaire {position: position_de_saut} avec un saut
    tous les floor(sqrt(n)) éléments, n étant la taille de la liste.
    C'est la fonction qui manquait dans le code d'origine.
    """
    n = len(liste)
    if n == 0:
        return {}
    pas = max(1, int(math.sqrt(n)))
    skips = {}
    i = 0
    while i < n:
        skips[i] = min(i + pas, n - 1)
        i += pas
    return skips
 
 
def intersection_avec_skips(list1, list2):
    """
    Même principe que intersection(), mais on utilise les pointeurs de
    saut pour avancer plus vite quand c'est possible.
 
    Important : on vérifie 'skips[i] > i' (respectivement 'skips[j] > j')
    avant de sauter. Sans cette vérification, en fin de liste le saut
    peut pointer sur la position courante elle-même : i (ou j) ne
    progresse plus et la boucle tourne indéfiniment. C'est le bug que
    contenait la version précédente.
    """
    i, j = 0, 0
    result = []
 
    skips1 = creer_skips(list1)
    skips2 = creer_skips(list2)
 
    while i < len(list1) and j < len(list2):
 
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
 
        elif list1[i] < list2[j]:
            if i in skips1 and skips1[i] > i and list1[skips1[i]] <= list2[j]:
                i = skips1[i]
            else:
                i += 1
 
        else:
            if j in skips2 and skips2[j] > j and list2[skips2[j]] <= list1[i]:
                j = skips2[j]
            else:
                j += 1
 
    return result
 
 
# =======================================================================
# Programme principal
# =======================================================================
if __name__ == "__main__":
 
    index_inverse = construire_index(documents)
 
    print("\n" + "=" * 70)
    print("Index inversé (ordre alphabétique des termes) :")
    for terme in sorted(index_inverse):
        print(f"{terme:15} -> {index_inverse[terme]}")
 
    # -------------------------------------------------------------
    # Au moins 5 requêtes à un terme
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Tests de requêtes à un terme :")
    termes_test = ["recherche", "documents", "index", "moteur", "indexation"]
    for terme in termes_test:
        print(f"  rechercher('{terme}') -> {rechercher(terme, index_inverse)}")
 
    # -------------------------------------------------------------
    # Au moins 3 requêtes AND (avec et sans skip pointers)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Tests de requêtes AND :")
    requetes_and = [
        ("recherche", "documents"),
        ("moteur", "recherche"),
        ("index", "recherche"),
    ]
    for t1, t2 in requetes_and:
        docs1 = rechercher(t1, index_inverse)
        docs2 = rechercher(t2, index_inverse)
        res_simple = intersection(docs1, docs2)
        res_skips = intersection_avec_skips(docs1, docs2)
        assert res_simple == res_skips, "Les deux méthodes doivent donner le même résultat !"
        print(f"  '{t1}' AND '{t2}' -> {res_simple}")
 
    # -------------------------------------------------------------
    # Requête interactive
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    terme1 = input("Entrez le premier terme : ").strip().lower()
    terme2 = input("Entrez le deuxième terme : ").strip().lower()
    docs1 = index_inverse.get(terme1, [])
    docs2 = index_inverse.get(terme2, [])
    print(f"Intersection : {intersection(docs1, docs2)}")