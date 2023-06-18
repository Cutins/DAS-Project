import numpy as np

# SEED
SEED                = 25

# GRAPH SETTINGS
GRAPH_TYPE          = "Cycle"                    # {"Cycle", "Path", "Star"}

# DATAFRAME SETTINGS
TARGET              = 6
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
EPOCHS              = 300
STEP_SIZE           = 1e-2
BATCH_SIZE          = 8                         # Dimension of the Minibatch 
N_BATCH             = int(np.ceil(SAMPLES_PER_AGENT / BATCH_SIZE))

# SAVE & PLOT OPTIONS
SAVE_WEIGHTS        = True
PLOT_FOLDER         = 'Test_digit_6'