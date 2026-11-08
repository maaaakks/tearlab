# Campagne de généralisation — protocole initial, 8 novembre 2026

Objectif : progresser contre les tireurs puis transférer vers des compositions
réservées, en distinguant gain de vitesse, acquisition et généralisation.
GPU natif qualifié ; aucune interaction avec le vrai jeu Isaac.

## Contrats de tâche

- 1 ou 3 PV abstraits : un contact retire un PV, mort au premier ou au troisième
  contact. Avec 3 PV, une victoire peut donc comporter au plus deux dégâts.
  Immunité native existante de 30 ticks. Ce budget est une expérience, pas une
  calibration des cœurs du jeu. Deux features ajoutées :
  santé restante / 3 et immunité / 30. Toutes les variantes ont la même entrée.
- Victoire : tous les ennemis éliminés, santé strictement positive. Clear sans
  dégât mesuré séparément sur chaque politique, y compris celle à trois contacts.
- Horizon fini de 400 décisions (40 s). Timeout = échec, pénalisé, sans bootstrap
  de continuation. Ce choix diffère explicitement du pilote précédent.
- Récompense de base : +3 victoire, -3 mort, -2 timeout, -0,1 par dégât et
  -0,002 par décision. Aucun bonus de position, visée, mur ou action prédéfinie.
- Shaping potentiel : 1,5 × (gamma Phi_suivant - Phi_actuel), Phi = fraction de
  PV ennemis retirés ; Phi_suivant = 0 à TOUT terminal. À état initial Phi=0,
  son retour actualisé est exactement nul sur une trajectoire complète.
  Il redistribue le crédit sans créditer les dégâts partiels d'une défaite.
- Contrôles de reward : victoire, mort avec dégâts partiels, timeout avec dégâts
  partiels, immunité, reset, mort+clear. Tests CPU puis comparaison native/tensorielle.

## Expériences prévues

1. Acquérir la visée : un tireur statique à deux PV, attaques désactivées.
   C'est une pratique, pas un résultat combat. Passage après deux examens à >=90 %.
   Comparer aux actions constantes et aléatoires ; conserver le parent.
2. Ablation du budget : même parent, mêmes situations et mêmes updates ; 1 contre
   3 contacts, reward identique. Un tireur, PV 2 puis 4 puis 8 puis 12, sans aucun
   changement du réseau. Audits séparés sur les deux budgets.
3. Apprentissage de deux tireurs puis mélanges avec rétention. Éviter le passage
   forcé à une difficulté non maîtrisée ; conserver meilleurs et derniers poids.
   Les changements suivant les diagnostics seront consignés AVANT leur run.
4. Comparer reward sparse et potentiel au budget retenu à exposition égale.
5. Reproduire le résultat retenu sur une autre graine d'apprentissage et auditer
   indépendamment les compositions, PV et géométrie réservés.

## Séparation des examens

- Développement et rétention : graines 910001–910099 ; ces scores peuvent choisir
  les poids et le passage de phase. Chaque phase enregistre ses propres critères.
- Audits finaux : graines 920001–920099. Ils ne servent pas à sélectionner un
  checkpoint. Les résultats inspectés ne deviennent pas des tests vierges pour
  une nouvelle itération : employer de nouvelles graines et le signaler.
- Jamais entraînées : 2 poursuivants + 1 tireur, 2 poursuivants + 2 tireurs,
  1 poursuivant + 3 tireurs ; poursuivants à 2/4/8 PV, tireurs à 12 PV.
- Audits distincts : poursuivants à 12 PV, rectangle 640×360, permutations des
  slots. Pas de revendication sur tous les ennemis, items ou obstacles d'Isaac.
- Promesse initiale : chercher >=30 % de clears sans dégât sur les compositions
  réservées, ou >=60 % de clears avec trois contacts, ET un progrès sur les
  tireurs connus. Un résultat n'atteignant que la seconde condition devra
  rester présenté comme une généralisation avec tolérance aux dégâts.

Les expériences sont bornées et enregistrent durée, transitions, phases,
rewards, entropie, KL, clipping, shots, dégâts, timeouts, scores par famille,
checkpoints avec sidecars, sources et artefacts W&B hors ligne. Une graine
unique ne démontre pas la robustesse. Les échecs restent dans le bilan.
