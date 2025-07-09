# Problèmes actuels

## Précision de l'estimation de profondeur

Pour l'instant, pas de garantie sur la précision atteignable. ça à l'air prometteur, mais rien n'est sûr pour l'instant

Solution actuelle -> vision stéréo -> résultats plus précis

Qu'est-ce qui mène à ceci : qualité de la syncronisation entre les caméras. Qualité de la calibration et de la camera en elle même.


## Qualité de la réidentification des personnes

Pas si importante si le reste marche bien. Mais pour l'instant ça ne marche pas bien, pas de piste majeure d'amélioration pour l'instant

Pas la plus grande priorité.

## Qualité de la détection et de la lecture des dossards

Fonctionne pas formidable pour l'instant. Beaucoup de pistes d'améliorations possible. Pas trop inquiétant pour l'instant

Peut être arcuo ? -> jusqu'à 10_000


## A quoi demain va servir

Récolte de donnée stéréo correctes afin d'avoir une base sur laquelle calculer la précision et la réussite de mon système. -> Est-ce que je peux avoir accès aux données de la courses de demain ? Si possible timestamp d'arrivée pour chaque dossard -> dans l'idée d'avoir le résultat à atteindre afin de pouvoir comparer les solutions et de mettre des chiffres sur les améliorations apportés au système.

Regarder brevet en général sur la solution technique

## Problème un peu plus profond

On va atteindre les limites du matériels à un moment. Les compromis sont :

- Précision du chronométrage (0.1s = au minium 20FPS)
- Précision des détections -> plus gros modèle = meilleure précision = plus lent
- Précision de la profondeur -> algorithmes plus précis = algorithmes plus complexes


Différentes optimisations sont possibles, mais longue à mettre en place

-> enregistrement séparé du traitement faisable -> raspberry pi avec la barre stéréo
