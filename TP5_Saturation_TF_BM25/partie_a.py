
# -*- coding: utf-8 -*-
"""
TP5 : Saturation de TF dans BM25
Partie A : A1, A2, A3, A4 et A5
"""

import matplotlib.pyplot as plt


# A1 : Fonction TF saturée de BM25

def tf_sat(tf, k1=1.2, b=0.75, dl=None, avgdl=None):
    """
    Composante TF saturée de BM25.

    Si dl et avgdl ne sont pas fournis,
    on suppose que dl = avgdl, donc K = k1.
    """

    if dl is None or avgdl is None:
        K = k1
    else:
        K = k1 * (1 - b + b * dl / avgdl)

    if tf == 0:
        return 0.0

    return (tf * (k1 + 1)) / (tf + K)


# A2 : Tableau des valeurs de TF saturé

print("\n========== A2 : TABLEAU TF SATURÉ ==========")

k1 = 1.2
liste_tf = [1, 2, 3, 5, 10, 20, 50]

borne = k1 + 1
precedent = 0

print(f"{'TF':<10}{'TF saturé':<20}{'Gain':<20}{'% de la borne':<20}")

for tf in liste_tf:
    valeur = tf_sat(tf, k1)
    gain = valeur - precedent
    pourcentage = (valeur / borne) * 100

    print(f"{tf:<10}{valeur:<20.4f}{gain:<20.4f}{pourcentage:<20.2f}")

    precedent = valeur


# A3 : Tracer les courbes de TF saturé

print("\n========== A3 : COURBES TF SATURÉ ==========")

tf_values = list(range(0, 31))
liste_k1 = [0.5, 1.2, 2.0, 10]

plt.figure(figsize=(10, 6))

# TF brut : y = tf
plt.plot(tf_values, tf_values, label="TF brut (y = tf)", linewidth=2)

for k1 in liste_k1:
    valeurs = [tf_sat(tf, k1) for tf in tf_values]

    plt.plot(
        tf_values,
        valeurs,
        label=f"TF saturé (k1={k1})"
    )

    # Borne théorique k1 + 1
    plt.axhline(
        y=k1 + 1,
        linestyle="--",
        label=f"Borne k1+1 ({k1 + 1})"
    )

plt.title("Comparaison entre TF brut et TF saturé")
plt.xlabel("Fréquence du terme (tf)")
plt.ylabel("Valeur")
plt.grid(True)
plt.legend()
plt.show()


# Vérification avec k1 = 0.01

print("\nVérification avec k1 = 0.01")

for tf in [1, 2, 5, 10, 20]:
    print(
        f"tf={tf} : TF saturé = "
        f"{tf_sat(tf, k1=0.01):.4f}"
    )


# A4 : Vérification mathématique

print("\n========== A4 : VÉRIFICATION MATHÉMATIQUE ==========")

k1 = 1.2
K = k1

# Pour p = 0.5
p = 0.5
tf_theorique = K * p / (1 - p)
valeur = tf_sat(tf_theorique, k1)

print("Pour p = 0.5 :")
print("TF théorique =", tf_theorique)
print("TF saturé =", valeur)
print("Pourcentage de la borne =", valeur / (k1 + 1) * 100, "%")

# Pour p = 0.9
p = 0.9
tf_theorique = K * p / (1 - p)
valeur = tf_sat(tf_theorique, k1)

print("\nPour p = 0.9 :")
print("TF théorique =", tf_theorique)
print("TF saturé =", valeur)
print("Pourcentage de la borne =", valeur / (k1 + 1) * 100, "%")


# A5 : Influence de la longueur du document

print("\n========== A5 : INFLUENCE DE LA LONGUEUR ==========")

k1 = 1.2
b = 0.75
tf = 3
avgdl = 100

# Document court
dl_court = 50

score_court = tf_sat(
    tf,
    k1=k1,
    b=b,
    dl=dl_court,
    avgdl=avgdl
)

# Document long
dl_long = 200

score_long = tf_sat(
    tf,
    k1=k1,
    b=b,
    dl=dl_long,
    avgdl=avgdl
)

print("Document court (dl = 50) :", round(score_court, 4))
print("Document long (dl = 200) :", round(score_long, 4))
