# global
SEED = 2023
DATADIR = './data/'
# DATASETS = ['dummy']
DATASETS = ['yahoo_movies']
# DATASETS = ['ml-1m']
# DATASETS = ['yahoo_movies', 'SmallLibraryThing']
OUTDIR = './output/'
# ATTACKS = ['random']
ATTACKS_BASE = ['random', 'average']
ATTACKS_SEMANTIC = ['sasha_random']
# ATTACKS_SEMANTIC = ['sasha_random', 'sasha_average']
ATTACKS = ATTACKS_BASE + ATTACKS_SEMANTIC

print(ATTACKS)
# RS_MODELS = ['mfcf', 'ubcf']
RS_MODELS = ['mfcf', 'ibcf', 'ubcf']
SIMILARITY_MEASURES = ['cosine']
# SIMILARITY_MEASURES = ['cosine', 'jaccard', 'pearson', 'adjusted_cosine']
EVALUATIONS = ['hit_ratio', 'pred_shift']
DETECTIONS = []
TRAIN_SIZE = 0.8
PUSH = True     # True: push the target user/item rating to the maximum rating, 
                # False: push the target user/item rating to the minimum rating
TOP_N = 50      # top-N recommendation
TOP_Ns = [10, 20, 30, 40, 50]   # top-N recommendation


FILLER_SIZE_PERCENTAGE = 1       # fraction of average number of ratings per user
ATTACK_SIZE_PERCENTAGE = 0.03    # 3% of the target user/item ratings will be attacked
# ATTACK_SIZES = [0.01, 0.02, 0.03, 0.04, 0.05] # 1%, 2%, 3%, 4%, 5% of the target user/item ratings will be attacked
# FILLER_SIZES = [0.5, 1, 1.5, 2, 2.5]       # 0.5, 1, 1.5, 2, 2.5, 3 times of the average number of ratings per user

ATTACK_SIZES = [ATTACK_SIZE_PERCENTAGE]
FILLER_SIZES = [FILLER_SIZE_PERCENTAGE]


NUM_TARGET_ITEMS = 50   

RATING_RANGE = {
    'ml-1m': (1, 5),
    'dummy': (1, 10),
    'yahoo_movies': (1, 5),
}

try:
    R_MIN, R_MAX = RATING_RANGE[DATASETS[0]]
except:
    R_MIN, R_MAX = 1, 5

LOG_FILE = 'log.txt'
EXP_NO = 0

# ----------------------------------------------- send mail -----------------------------------------------
# SUBJECT = 'SAShA detection'
BODY = 'Experiment done. sending a copy of the log file'

# ----------------------------------------------- Data Loader -----------------------------------------------
# ml-1m
COLD_START_THRESHOLD = 5 # to avoid cold start drop users with less than 5 ratings and items with less than 5 ratings [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]
SAMPLE_FRAC = 0.25 # randomly sample 25% of the data [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]
# SAMPLE_FRAC = 0.01 # randomly sample 25% of the data [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]


# ----------------------------------------------- RS models -----------------------------------------------
# mfcf: matrix factorization
ALPHA = 0.001
K = 5   # number of latent features
BETA = 0.02
MAX_ITER = 100

# ibcf: item-based collaborative filtering
# IKNN = 2
IKNN = 10

# ubcf: user-based collaborative filtering
# UKNN = 2
UKNN = 10


# ----------------------------------------------- Attacks -----------------------------------------------
from utils.similarity_measures import *
KG_SIMILARITY = adjusted_cosine_similarity
SAMPLE = 0.25 
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
