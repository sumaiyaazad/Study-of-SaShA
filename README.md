# On Detection Of Knowledge Graph Based Shilling Attacks On Collaborative Filtering Recommendation Systems

This repository contains the code for our undergraduate thesis work titled "On Detection Of Knowledge Graph Based Shilling Attacks On Collaborative Filtering Recommendation Systems".

## Abstract

Collaborative filtering (CF) is a widely used technique for recommender systems. It is based on the assumption that users with similar preferences tend to rate items similarly. However, this assumption is violated in the presence of shilling attacks. In this paper, we propose a novel approach to detect shilling attacks in CF-based recommendation systems. We use the knowledge graph (KG) to detect shilling attacks. We propose a novel KG-based shilling attack detection (KG-SAD) approach that uses the KG to detect shilling attacks. We evaluate our approach on two real-world datasets. The results show that our approach outperforms the state-of-the-art approaches in terms of precision, recall, and F1-score.

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
- see [generate_recommendations_ubcf.py](generate_recommendations_ubcf.py) for more details.

```bash
    python generate_recommendations_ubcf.py

```

### Training

- Run the following command to train the model.

```bash

```

### Evaluation

- Run the following command to evaluate the model.

```bash

```

## Citation

If you find this repository useful, please cite our paper.

```bibtex
```