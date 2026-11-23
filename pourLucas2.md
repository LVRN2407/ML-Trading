Bon Lucas je continue un peu aujourd'hui, 
Aujourd'hui je t'explique le principe de multiprocessing, vu que 
t'as pas mal avancé sur la conception dure du modèle de machine learning 
je fais peaufiner les détails pour le choix des etf, les méthodes de visualisation 
et les data que tu as garder plus tard (même si en vrais c'est un peu le plus important)

Quand tu arrives à la fin d'un projet de machine learning, le plus important est de pouvoir faire 
de nombreux benchmark en variant un peu les hyperparamètres pour trouver les paramètres qui marcherait 
de la meilleur manière (évitant le overfitting et le underfitting), pour faire ça tu peux lancer ton programme 
et changer les hyperparamètres mais le problème principale c'est le temps ! on a besoin de beaucoup de temps 
surtout qu'on va surement continuer à complexifier les datasets, ajouter d'autres etf, varier les modèles qui peuvent prendre plus de temps... Donc une question important qu'on doit se poser à cette étape c'est comment optimiser le temps qu'on a pour pouvoir lancer 
tout le processus et recevoir les résultats en un minimum de temps. 

Pour ça, on utilise du multiprocessing: les processeurs qu'on utilise on tous un nombre de coeur, par exemple, pour toi, 
on peut lancer le gestionnaire de tâche et aller dans performance puis clique sur processeur et tu peux voir le nombre de coeur que 
tu as: par exemple j'en ai 8 (normalement ça va de 4 pour les processeurs entrée de gamme jusqu'a 16 environ). L'avantage des coeurs,
c'est que tu peux lancer plusieurs programme en parallèle sur chacun des coeurs, ce qu'on va utiliser pour optimiser encore plus notre programme. Le point qui est important c'est de choisir où paralélliser notre programme pour optimiser notre vitesse de résultat. 

déjà pour récupérer le nombre de coeurs que tu as dans ta machine tu peux utliser : 
 available_cpus = os.process_cpu_count() ou os.sched_getaffinity(0) (mais celui marche que sur linux)--> te donne le nombre de coeur valable pour ton progamme 
 après souvent c'est mieux de pas tous les utiliser parceque windows à aussi besoin de tes coeurs pour 
 fonctionner, si tu mets les 8 au max tu vas avoir des latences de fou (tkt j'ai déjà essayer). 

 Ici vu que j'ai décidée de paralléliser sur les etf on obtient ça 
 ici worker = coeurs (plus ou moins)
workers = max(1, min(max_workers, available_cpus, len(DATASETS)))
 où len de datasets = 4 (4 etf en même temps )
donc au minimum on a un coeur et au maximum on a le minimum entre le nombre d'etf, le nombre max théorique de workers et 
le nombre de coeurs disponible (dans mon cas vu que j'ai 8 coeurs ce sera le nombre d'etf là limite)



Donc là t'as le nombre de coeurs disponible mainteant tu va les paralléliser sur un tâche précise : 
par exemple : 

 with ProcessPoolExecutor(max_workers=workers) as executor:
        scores = list(executor.map(evaluate_dataset, jobs))
on les lances tous sur evaluate_dataset 

de tel manière quand tu lances le processur t'obtient plus ou moins 
ça : 

Processus 1 → SPY → XGBoost → Random Forest → LightGBM
Processus 2 → DIA → XGBoost → Random Forest → LightGBM
Processus 3 → SPLV → XGBoost → Random Forest → LightGBM
Processus 4 → TLT → XGBoost → Random Forest → LightGBM


maintenant si tu veux aller plus dans les détails tu peux t'amuser à compter le 
temps que tu gagnes en faisant ça: 
ici théoriquement tu te dis que ça va 4x plus vite vu que tu lances les 4 opérations en même temps 
en réalité c'est moins vu que les 4 processus ne se termine pas en même temps 

par exemple imagine dia est un dataset 2x plus grand que les autres, et bah il va se terminer plus tard 
si les autres prennent 5 min lui il prendra 10 min à se terminer 

Si on est dans le modèle linéaire de base ça donnerait ça : 
Spy (5min) + dia (10min) + splv(5min) + tlt(5min) = 25 minutes au total 

Si on parallélise on obtient : 

Processus 1 → SPY : 5min
Processus 2 → DIA : 10min
Processus 3 → SPLV : 5 min
Processus 4 → TLT : 5 min 

Donc au total on aura le temps du processus le plus long soit 10 min dans ce cas 

Dans ce cas on ne sera allé que 2.5x plus vite (le terme des pros ici c'est le speedup)
donc la diff sera moins grand 



donc là techniquement si tu veux optimiser au grand grand max tu te dis quoi: 
bah quitte à optimiser au max et si on a beaucoup beacoup de coeur autant créer 12 processus de 
la manière suivante : 

Processus 1 → SPY → XGBoost 
Processus 2 → SPY→Random Forest 
Processus 3 → SPY→ LightGBM
Processus 4 → DIA → XGBoost
Processus 5 → DIA  → Random Forest 
Processus 6 → DIA  → LightGBM
Processus 7 → SPLV → XGBoost
Processus 8 → SPLV → Random Forest
Processus 9 → SPLV  → LightGBM
Processus 10 → TLT → XGBoost
Processus 11 → TLT→ Random Forest
Processus 12 → TLT  → LightGBM

Donc là t'aurais 12 worker (environ 12 coeurs) qui travailleraient en même temps 
et qui ferait un gain théorique de x12 

ça marche pas vraiment parceque chacun de nos modèles xgboost, random forest... 
font déjà du multi-threading en interne donc si on fait ça on risque de ralentir le modèle en question 

donc ça devient une bataille entre : 
oui d'un côté on en fait plus en même temps 
mais de l'autre ils vont tous plus lentement --> à part si on précise pour chaque modèle qu'ils ont au max 
n_job = 1 (donc dans ce cas, si on fait ça on peut s'attendre à des gains rééls, il faudrait check en temps réel pour 
être sûr)



Bon bref, j'ai fais des test avec le premier modèle que j'ai implémenté et c'est pas du tout aussi fameux que la théorie on est plutôt sur un gain réel de 20% avec le programme sans multiprocessing qui prend environ 83 sec et celui avec qui prend 66, donc bon c'est très utile mais ça fonctionnera encore mieux quand on aura clairement optimisé tous les coeurs et tout le processus ! 

