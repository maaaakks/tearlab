# Essai temps réel de l'agent IsaacRL

Lanceur sur le Bureau : **IsaacRL - Jouer agent.cmd**.

1. Démarrer une **nouvelle partie avec Isaac**, de préférence sans objets ajoutés.
2. **C** active l'agent dans la salle courante ; **C** rend immédiatement le contrôle.
3. À chaque dégât, changement de salle, fin du combat, pause ou perte de connexion,
   le contrôle revient au joueur. **C** permet de réessayer. Aucun reset automatique.
4. Plus de quatre ennemis, huit larmes ou 32 projectiles : retour manuel, capacités
   du modèle dépassées. L'agent ne navigue pas entre salles et n'utilise pas d'objet.

Pour un essai simple, ouvrir la console de débogage (touche de console du jeu) :

- `rlpreview horf 1`, puis `rlpreview horf 2` ou `rlpreview horf 4` : vrais Horfs.
- `rlpreview gaper 1` ou `rlpreview gaper 4` : vrais Gapers.

**Ces commandes remplacent les ennemis de la salle courante et placent Isaac au
centre.** Elles sont exécutées seulement si tu les tapes ; rien ne se lance seul.
Les monstres gardent leur IA vanilla : ce ne sont pas les ennemis synthétiques du
simulateur. Les obstacles/objets du jeu restent présents, donc le test peut être
hors distribution. L'agent n'a pas encore été validé sur le vrai jeu.

Checkpoint figé : `isaac-crossfire/focus-seed129773/pool-best.pt`, choisi avant cet
essai (meilleure première graine). Inférence CPU, actions déterministes à 10 Hz,
vitesse de jeu normale. Aucun optimiseur ni apprentissage en arrière-plan.

Projection : unités natives, vitesses converties comme dans le banc physique,
PV ennemis exprimés en coups de dégâts courants. Préavis d'attaque inconnu pour
les IA vanilla : entrée à zéro et manque déclaré dans les traces. Durée des larmes
estimée par leur âge. Vie d'observation = contexte abstrait un contact du modèle,
pas les cœurs réels ; arrêt au premier dégât.

`current-session.json` indique le dossier des traces : état moteur, action demandée,
action appliquée, dégâts et latence d'inférence. `status.json` indique la connexion.
Pour arrêter le processus Python : exécuter `stop_preview.ps1` ou fermer via Ctrl+C
si lancé dans un terminal. Pour revenir aux anciens liens/options : jeu fermé,
`restore_preview.ps1`. Les sauvegardes ont été copiées avant installation et ne
sont jamais restaurées automatiquement par-dessus une nouvelle progression.
