# DAS-Project
Project of Distributed Autonomous Systems course

Modifiche del 2023/05/10
- Implementato Gradient Tracking

Modifiche del 2023/05/30
- Implementato nuovo DataSet bilanciato
- Completato Task 1_2


TO DO:
- Inserire il Save di uu in Distirbuted Algorithm
- Plottare l'evoluzione della norma di SS e UU
- Salvare per mostrare al prof al ricevimento la UU e relativi grafici
- Creare il file .py per solo i Test Set
- Plottere la differenza tra la Media e i singoli agenti di UU, questo deve andare a 0

- CAPIRE PERCHÈ NON CONVERGONO LE UU





###### DOMANDE PER PICHIERRI  #############
# Neural Network
- Nell'inizializzazione del Gradient Trackind come calcolo ss[0]?
    La nostra idea era runnare tutta la rete e trovare la ss, cioè il gradient per tutti i layer e per tutti i neuroni
    Inizializziamo su una singola immagine o su tutto un minibatch per ogni agente?
- Come mostrare se le UU vanno al consenso o meno?
- Come modificare la struttura della rete per gestire le immagini 28X28

# ROS
- Come definire le distanze in 3D
- Come mai il potenziale non va a 0?
- Come mai i follower non vanno nel convex hull

