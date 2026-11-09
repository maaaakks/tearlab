# Démonstration automatique

Relance Isaac et entre dans une partie avec Isaac : la démo doit démarrer sans
console ni activation manuelle. L'agent Python est préparé en attente.

Onze scénarios tournent en boucle : 1/2/3/4 Horfs, 1/2/3/4 Gapers et trois
mélanges. Après victoire, premier contact ou 40 secondes, le suivant démarre.
Les monstres gardent leur IA et leurs PV vanilla. Les placements tournent.

Les cœurs sont protégés pendant la boucle, mais chaque contact compte comme un
échec immédiat. C arrête ou reprend la démo. Une déconnexion ou un retard arrête
la boucle. Le modèle reste figé, sur CPU, à vitesse normale et 10 décisions/s.

Le lanceur Bureau « IsaacRL - Jouer agent.cmd » relance le processus si nécessaire.
« IsaacRL - Arrêter agent.cmd » l'arrête sans utiliser le clavier dans le jeu.
Les résultats et actions sont dans le dossier indiqué par current-session.json.

Les tests de syntaxe, de projection et de la boucle avec API simulées passent.
Le lancement effectif sera à confirmer après ton rechargement ; aucun contrôle
de l'écran n'est utilisé pour cette dernière préparation.
