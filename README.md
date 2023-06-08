# DAS-Project

TO DO:
- Plot di tutte le uu che devono andare al consenso
- Pescare un neurone a caso dalla rete e vedere le uu di tutti gli agenti per quel neurone. Devono andare al consenso.

- Fare lo stopping criteria quando: Derivata = 0 e distanza dalla media = 0
- Nuova dinamica, i leader si muovono solo con Input
- Giocare con gli input
- Aggiungere un agente definito come "the_obstacole"





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
- Come gestire la comunicazione (launch file e agente) affinche sia realmente Distribuito?



###### RICEVIMENTO PICHIERRI  #############
# NN
- Salavare la ss per i mini_batch

# ROS
- Aggiungere il movimento dei leader direttamente nella formula
- Il codice dei grafici è fatto male, va salvato prima tutti i vicini posizione per potenziale
- Aggiungere un agente definito come "the_obstacole"
- Conteinmnet da vedere la Laplaciana
- Salvare la config di Rviz2
- shape 3D