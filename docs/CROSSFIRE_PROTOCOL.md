# IsaacRL — tirs croisés, 9 novembre 2026

Périmètre autorisé : diagnostic, curriculum/placements, représentations. Aucun jeu réel.
Les anciens checkpoints restent figés et servent de parents/baselines. Reward sparse,
1 PV abstrait, horizon fini 400, gamma .995, PPO/rollout inchangés.

1. Diagnostiquer les deux parents sur 2/3/4 tireurs et anciens mélanges, en séparant
   générateur historique et placement contrôlé commun à tous les effectifs. Enregistrer
   projectile/contact, bord/coin, projectiles actifs, dégâts infligés et durée ; les
   corrélations aux morts ne sont pas des preuves causales. Sauver les derniers états
   diagnostiques, sans choisir les épisodes réussis.
2. Entraîner depuis chaque parent : 3 tireurs 4 PV puis 12 PV, 4 tireurs 4 PV puis
   12 PV ; distribuer les placements centre/libre/bord/coin et conserver un/deux
   tireurs, poursuivants et mélanges précédents. Première comparaison à exposition
   fixe : baseline pooling, branche de cinématique, branche d'attention, mêmes
   parents, scènes, RNG et nombres d'updates. Les branches sont initialement nulles,
   donc les trois politiques commencent avec les mêmes sorties. Pas de teacher.
3. Développement : deux cohortes fixes 960101/960201, 64 essais/cellule chacune.
   Critère d'éligibilité : rétention un tireur >=80%, deux >=60%, poursuivants >=65%.
   Puis score cible moyen 3/4T (sans masquer chaque cellule). Checkpoint choisi
   uniquement sur ces examens ; toutes les variantes gardées, même décevantes.
   Budget initial : 1600 updates/bras (400 par palier), 1024 voies, 32 décisions,
   donc 52 428 800 transitions/bras ; métriques vitesse/paramètres/coût incluses.
   Continuer le curriculum gagnant uniquement si le diagnostic justifie un nouveau
   palier, décision consignée avant lancement, sans consulter le test final.

Test final fermé jusqu'au gel des poids : graines 970101 et 970201, 128 essais/cellule,
compositions anciennes réservées devenues benchmark, placements groupés non entraînés,
tirs initialement synchronisés et intervalles ensuite communs (non entraînés), et
dimensions 640x360 (non entraînées). Rapporter chaque déplacement de distribution
séparément ; pas de score moyen qui masque un échec. Les seeds seules ne rendent
pas une composition inédite. Le nouveau test porte sur ces facteurs réservés.

Les placements contrôlés sont valides, espacés et indépendants de l'effectif :
générer quatre positions puis activer les premières. Les phases ordinaires sont
indépendantes et la synchronisation ne modifie ni vitesses ni dommages. Vérifier
reset partiel, terminaux/observations, masque des projectiles et parité native/tensor.
La cinématique est dérivée des observations déjà disponibles (vitesse relative,
temps/distance au plus proche passage à vitesse constante), sans information future.
L'attention reçoit les mêmes projectiles, sans ces features supplémentaires.

Hypothèses à challenger : avantage réel des features ou capacité supplémentaire ;
oubli des compétences anciennes ; performance d'un début au centre vs bord ; perte
de débit causée par réseau/reset ; vie stricte utile à l'esquive mais pas preuve
d'optimalité dans le jeu ; aucun résultat live revendiqué.
