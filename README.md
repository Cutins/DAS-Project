# DAS-Project

TO DO:
- Implementare la tecnica 'Bearing'
- Muovere gli obstacole, contro gli altri obstacle
- Diminiscing step-size (cambiare la batch-size)

- Test_3
- Ripulire / commentare i codici

- 



###### RICEVIMENTO PICHIERRI  #############
# NN
- Salavare la ss per i mini_batch

# ROS
- Conteinmnet da vedere la Laplaciana
- shape 3D



##### TEST SIGNIFICATIVI PER REPORT #####
# NN
- Salvare in un file .txt il risultati a terminale
- inserire il config.txt

- Test {1} con:
# SEED
SEED                = 25

# GRAPH SETTINGS
GRAPH_TYPE          = "Cycle"                    # {"Cycle", "Path", "Star"}

# DATAFRAME SETTINGS
TARGET              = 8
SIZE                = (28, 28)
N_AGENTS            = 5
SAMPLES_PER_AGENT   = 256                        # Multiple of Minibatch Size
SAMPLES = N_AGENTS*SAMPLES_PER_AGENT

# NETWORK SETTINGS
INPUT_SIZE          = SIZE[0]*SIZE[1]
NETWORK             = [INPUT_SIZE, 28 , 1]
ACTIVATION_TYPE     = "Sigmoid"                 # {"Sigmoid", "ReLu", "HyTan"}
LOSS_TYPE           = "BinaryCrossEntropy"      # {"Quadratic", "BinaryCrossEntropy"}

# TRAINING SETTINGS
EPOCHS              = 10
STEP_SIZE           = 1e-2
BATCH_SIZE          = 8                         # Dimension of the Minibatch 
N_BATCH             = int(np.ceil(SAMPLES_PER_AGENT / BATCH_SIZE))

# SAVE & PLOT OPTIONS
SAVE_WEIGHTS        = True
PLOT_FOLDER         = 'Test_1_test'




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
    STEP_SIZE   = 1e-2
    BATCH_SIZE  = 8


- Test {3} con: Differenza della cost function
    TARGET              = 7
    SIZE                = (28X28)
    AGENT               = 5
    SAMPLES_PER_AGENT   = 256
    CostFunct = "Quadratic"

    EPOCHS      = 1000 or (10 senza impuvement)
    STEP_SIZE   = 1e-2
    BATCH_SIZE  = 8


- Test {4} con: Togliendo i mini batch
    TARGET              = 8
    SIZE                = (28X28)
    AGENT               = 5
    SAMPLES_PER_AGENT   = 32
    CostFunct = "BinaryCrossEntropy"

    EPOCHS      = 2*1000 or (10 senza impuvement)
    STEP_SIZE   = 2*1e-
    BATCH_SIZE  = 32


- Test {5} con: "Star"
    TARGET              = 7
    SIZE                = (28X28)
    AGENT               = 5
    SAMPLES_PER_AGENT   = 256
    CostFunct = "BinaryCrossEntropy"
    GRAPH_TYPE = "Star" 

    EPOCHS      = 400
    STEP_SIZE   = 1e-2 #Costant
    BATCH_SIZE  = 32


- Test riducendo lo stepsize (diminishing da 1e-1 a 1e-3) circa 200 epoche a occhio
- Test sui diversi grafi (cycle, star e path) e commentare la "velocità" di consenso
-


  

# ROS
- Fare il plot_csv opne
- Fare Task_2.2 dove si vede bene il collision avoidance (Posizione iniziale in asse)
    Testarla con molte shape diverse
- Task 2.3 provare movimento a cerchio e wavesAMENT_PREFIX_PATH 


- Aggiungere le legende mancanti ai plot
Plot:
con grid e righe spesse