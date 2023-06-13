# DAS-Project

TO DO:
- Giocare con gli input, tipo cerchio 
- Implementare la tecnica 'Bearing'
- Muovere gli obstacole 
- Implementare il 'Creazy choir'



###### RICEVIMENTO PICHIERRI  #############
# NN
- Salavare la ss per i mini_batch

# ROS
- Conteinmnet da vedere la Laplaciana
- shape 3D



##### TEST SIGNIFICATIVI PER REPORT #####
# NN
- Modificare codice per (Stopping criteria = 10 senza improuvement)
- Da controllare se la SS va aggiornata per batch (come il gradiente) o ogni volta

- Salavare i plot di: [J, Norm_J, UU_mean, UU, UU_single_neurons, SS]
- Salvare i plot di [UU, UU_single_neaurons] zommate sulle prime iterazioni significative per aprezzare il grafico.
- Salvare in un file .txt il risultati a terminale

- Test {1} con:
    TARGET              = 8
    SIZE                = (28X28)
    AGENT               = 5
    SAMPLES_PER_AGENT   = 256
    CostFunct = "BinaryCrossEntropy"

    EPOCHS      = 1000 or (10 senza impuvement)
    STEP_SIZE   = 1e-1
    BATCH_SIZE  = 8


- Test {2} con: Per vedere la differenza col target
    TARGET              = 7
    SIZE                = (28X28)
    AGENT               = 5
    SAMPLES_PER_AGENT   = 256
    CostFunct = "BinaryCrossEntropy"

    EPOCHS      = 1000 or (10 senza impuvement)
    STEP_SIZE   = 1e-1
    BATCH_SIZE  = 8


- Test {3} con: Differenza della cost function
    TARGET              = 8
    SIZE                = (28X28)
    AGENT               = 5
    SAMPLES_PER_AGENT   = 256
    CostFunct = "Quadratic"

    EPOCHS      = 1000 or (10 senza impuvement)
    STEP_SIZE   = 1e-1
    BATCH_SIZE  = 8


- Test {4} con: Togliendo i mini batch
    TARGET              = 8
    SIZE                = (28X28)
    AGENT               = 5
    SAMPLES_PER_AGENT   = 32
    CostFunct = "BinaryCrossEntropy"

    EPOCHS      = 2*1000 or (10 senza impuvement)
    STEP_SIZE   = 2*1e-1
    BATCH_SIZE  = 32