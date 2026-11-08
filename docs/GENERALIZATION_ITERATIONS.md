# Décisions d'expérience consignées avant les runs suivants

## Après la visée et pendant l'ablation du budget

La pratique atteint 100 % sans dégât sur deux checks de développement ; parent
conservé à l'update 400. Le meilleur contrôle constant fait 10,9 % dans cette
pratique sans dégât ; le contrôle aléatoire 61,7 %. Ce dernier montre que le test
facile sert à amorcer la compétence, pas à démontrer le combat.

Sur l'ablation à exposition identique, à l'update 600, le bras 1-contact fait
96,5 % contre un tireur à douze PV ; le bras 3-contacts fait 65,2 % avec tolérance,
et 4,7 % sur son examen 1-contact. Le choix final attend le run complet. Cette
observation réfute l'hypothèse simple « autoriser des dégâts aide forcément ».

## Passage aux deux tireurs — règle avant lancement

Chaque run de curriculum traite UN palier ; aucune promotion forcée au palier
suivant lorsqu'un budget de temps expire. Le mélange garde des scénarios précédents.

- `two4` : 6 parts deux tireurs à 4 PV, 2 parts un tireur à 12 PV, 1 part pratique.
  Passage après deux examens à >=85 % sur deux tireurs 4 PV et >=80 % sur un 12 PV.
- `two12` : 8 parts deux tireurs 12 PV, 2 parts un 12 PV, 1 part deux tireurs 4 PV.
  Passage après deux examens à >=70 % sur deux 12 PV et >=80 % sur un 12 PV.
- `mixed` : familles 1P1T, 1P2T, 3P1T avec rétention un/deux tireurs et poursuivants.
  Passage après deux examens : >=70 % de moyenne des trois mélanges, >=50 % pour
  chacun, rétention >=80 % sur un tireur et >=60 % sur deux.

Les poids sont choisis sur qualité équilibrée (minimum des taux rapportés à ces
seuils), puis réussite de la tâche, sans dégât et durée. Les compositions réservées
ne sélectionnent jamais les poids.

La comparaison reward sparse/potentiel utilise exactement le même mélange que
`two12`, depuis le même parent et le même optimiseur initial, avec exposition égale.
Les checkpoints choisissent le score des deux tireurs, pas celui d'un tireur.
La sélection tient aussi compte de la rétention : minimum du taux un-tireur / 0,8
et du taux deux-tireurs / 0,7, puis score deux-tireurs, no-hit et durée.
La qualité est plafonnée à 1 : une fois les seuils de rétention franchis, un
gain de quelques points sur un tireur ne doit pas primer sur le progrès de la
nouvelle tâche. Les gates de réussite restent inchangées.

## Après le premier run deux tireurs à 4 PV

Le budget de 1 600 updates s'arrête à 83,6 % sans dégât sur deux tireurs 4 PV,
avec 98,4 % de rétention et 53,5 % sur deux tireurs 12 PV. Le seuil 85 % répété
n'est pas franchi : aucune promotion. Continuer ce même palier depuis le meilleur
parent, paramètres inchangés, au plus 800 updates ou 120 s. Les épisodes et RNG
redémarrent ; les poids et l'optimiseur sont repris, ce n'est pas une reprise bit exacte.

## Après la continuation deux tireurs 4 PV

Gates franchies : 85,9 % puis 88,7 % sans dégât, avec rétention 98 % ; 66,8 %
sur deux tireurs à 12 PV avant entraînement explicite de ce cas. Prochaine
comparaison : sparse contre potentiel, budget 1-contact, 600 updates chacun
depuis ce même parent, mélange deux-tireurs12/rétention identique, plafond 240 s.
Les compositions réservées restent fermées.

## Compétence manquante avant les mélanges

Les examens de rétention montrent que les poursuivants n'ont pas été acquis :
deux poursuivants restent à quelques pourcents pour le parent strict, malgré les
progrès contre deux tireurs. Ajouter un palier `chasers` avant le mélange :
1P à deux PV, 2/3/4P à quatre PV, avec rétention un/deux tireurs. Gate avant
promotion : moyenne poursuivants >=85 %, chaque cas >=70 %, un tireur >=80 %,
deux tireurs >=60 %, sur deux checks. Aucun test de composition réservée utilisé.

## Après l'ablation des rewards sur deux tireurs natifs

Le plafond de 240 s donne 436 updates par bras (au lieu des 600 prévues).
Exposition identique conservée. Meilleurs examens ID : sparse 75,0 %, potentiel
74,2 % sans dégât ; rétention un-tireur 99,6 % / 99,2 %. Pas d'avantage probant
du potentiel sur cette seule graine. Choix sparse par simplicité et score, puis
palier poursuivants depuis le meilleur sparse : jusqu'à 1 200 updates / 300 s.
Les seuils et scènes sont ceux consignés ci-dessus. Les audits compositionnels
restent fermés. Un score absent (`null`) dans les métadonnées `latest` signifie qu'il n'a pas
été réexaminé ; les anciens runs écrivaient à cet endroit le score du meilleur,
à ne pas interpréter comme une mesure des derniers poids. Les audits utilisent `best`.

Le futur palier mélange conserve aussi 1/3/4 poursuivants, en plus des deux
poursuivants déjà prévus, et exige >=70 % de moyenne sur ce groupe de rétention.
Les seuils de mélanges et de tireurs restent ceux annoncés.

## Poursuivants : plateau local, pas de promotion

Au check 1 000, 2P=96,9 %, 3P=87,1 %, 4P=69,9 %, un tireur=100 %, deux=69,5 %.
Le groupe moyen passe 85 %, mais 4P manque le seuil individuel et aucune paire
de checks ne valide la gate. Tester uniquement une redistribution d'exemples :
`chasers-focus`, 50 % quatre poursuivants, 10 % chacun sur 3P,2P,1P,1T,2T.
Même reward sparse, même LR, même réseau, mêmes seuils, <=800 updates / 240 s.
Cette expérience ne change pas la définition de réussite pour obtenir un passage.
