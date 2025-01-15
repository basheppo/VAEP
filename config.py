class Config:
    BATCH_SIZE = 32
    EPOCHS = 3000
    LATENT_DIM = 64
    INPUT_SHAPE = (100, 100, 100, 1)
    INITIAL_LEARNING_RATE = 0.0001
    DATASET_PATH = "/home/basheppo/dataset"
    PROPERTIES_FILE = "numres.csv"
    CHECKPOINT_PATH = "./VAEP"
    LOG_FILE = 'VAEP_logs.csv'