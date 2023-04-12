# On Detection Of Knowledge Graph Based Shilling Attacks On Collaborative Filtering Recommendation Systems

This repository contains the code for our undergraduate thesis work titled "On Detection Of Knowledge Graph Based Shilling Attacks On Collaborative Filtering Recommendation Systems".

## Abstract

Collaborative filtering (CF) is a widely used technique for recommender systems. It is based on the assumption that users with similar preferences tend to rate items similarly. However, this assumption is violated in the presence of shilling attacks. 

## Requirements

- Python 3.6
- numpy 1.18.1

## Datasets

- [MovieLens 1M]( https://grouplens.org/datasets/movielens/1m/ )
- [MovieLens 20M]( https://grouplens.org/datasets/movielens/20m/ )

## Usage

### Data Preprocessing

- Download the datasets from the links provided above.
- Run the following command to preprocess the datasets.

```bash

```

### Generating Recommendations

- Run the following command to generate recommendations.
- see [generate_recommendations_ubcf.py](generate_recommendations_ubcf.py) for more details about the command line arguments.

```bash
    python generate_recommendations_ubcf.py

```

### Generating Shilling Profiles

- Run the following command to generate shilling attacks.
- see [generate_shilling_profiles_base_attacks.py](generate_shilling_profiles_base_attacks.py) for more details about the command line arguments.
- The following attacks are currently supported:

    - `random`: Randomly select items for filler and rated by sampling from the normal distribution with mean and standard deviation of the user's ratings.
    - `average`: Selected items randomly and rated based on the average rating of the items.

```bash
    python generate_shilling_profiles_base_attacks.py --attack=<attack_name>

```

### Training

- Run the following command to train the model.

```bash

```

### Evaluation

- Currently the following metrics are supported:

    - `prediction shift`: The prediction shift is a way of quantifying how well a pushed (or nuked) item has been shifted in a direction favoring its goal.
    - `hit ratio`: Hit ratio is a measure of whether the pushed item made it to the top-k list (or was removed from the top-k list).

```bash

```

## Citation

If you find this repository useful, please cite our paper.

```bibtex
```