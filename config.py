# global
SEED = 2023
DATADIR = './data/'
DATASET = 'ml-1m'
OUTDIR = './output/'
ATTACKS = ['random', 'average']
RS_MODELS = ['mfcf', 'ibcf', 'ubcf']
DETECTION = []
TRAIN_SIZE = 0.8
ATTACK_SIZE_PERCENTAGE = 0.1    # 10% of the target user/item ratings will be attacked
PUSH = True     # True: push the target user/item rating to the maximum rating, 
                # False: push the target user/item rating to the minimum rating
TOP_N = 50      # top-N recommendation

rating_range = {
    'ml-1m': (1, 5),
}

R_MIN, R_MAX = rating_range[DATASET]
LOG_FILE = OUTDIR + 'log.txt'
FILLER_SIZE_PERCENTAGE = 1 # fraction of average number of ratings per user

# ----------------------------------------------- Data Loader -----------------------------------------------
# ml-1m
COLD_START_THRESHOLD = 5 # to avoid cold start drop users with less than 5 ratings and items with less than 5 ratings [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]
SAMPLE_FRAC = 0.25 # randomly sample 25% of the data [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]


# ----------------------------------------------- RS models -----------------------------------------------
# mfcf: matrix factorization
ALPHA = 0.001
K = 2
BETA = 0.02
MAX_ITER = 100

# ibcf: item-based collaborative filtering
IKNN = 10

# ubcf: user-based collaborative filtering
UKNN = 10


# ----------------------------------------------- Attacks -----------------------------------------------
# random: random attack
# R_MAX = 5
# R_MIN = 1
# ATTACK_SIZE_PERCENTAGE = 0.1
# PUSH = True


# ----------------------------------------------- Detection -----------------------------------------------
# mlp: multi-layer perceptron
# MLP_EPOCHS = 100
# MLP_BATCH_SIZE = 32
# MLP_HIDDEN_SIZE = 32
# MLP_DROPOUT = 0.2
# MLP_LR = 0.001
