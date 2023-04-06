DATADIR = './data/'
DATASET = 'ml-1m'
OUTDIR = './output/'

# matrix factorization
ALPHA = 0.001
K = 2
BETA = 0.02
MAX_ITER = 100

# data loader

COLD_START_THRESHOLD = 5 # to avoid cold start drop users with less than 5 ratings and items with less than 5 ratings [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]
SAMPLE_FRAC = 0.01 # randomly sample 25% of the data [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]
# SAMPLE_FRAC = 0.25 # randomly sample 25% of the data [ref: https://link.springer.com/chapter/10.1007/978-3-030-49461-2_18]