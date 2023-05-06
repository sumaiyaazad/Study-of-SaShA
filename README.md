# On Detection Of Knowledge Graph Based Shilling Attacks On Collaborative Filtering Recommendation Systems

This repository contains the code for our undergraduate thesis work titled "On Detection Of Knowledge Graph Based Shilling Attacks On Collaborative Filtering Recommendation Systems".

## Abstract

Collaborative filtering (CF) is a widely used technique for recommender systems. It is based on the assumption that users with similar preferences tend to rate items similarly. However, this assumption is violated in the presence of shilling attacks. 

## Requirements

- Python 3.6
- numpy 1.18.1

## Datasets

- [MovieLens 1M]( https://grouplens.org/datasets/movielens/1m/ ) (not processed for semantic aware attacks)
- [Yahoo Movies](https://github.com/sisinflab/LinkedDatasets/tree/master/yahoo_23MB)
- [Small Library Things](https://github.com/sisinflab/LinkedDatasets/tree/master/LibraryThing)

## Usage

### Data Preprocessing

- Most of the preprocesing specific to datasets is done in the data_loader.py file.
- Download the datasets from the links provided above. You can find more compatible preprocessed datasets [here](https://github.com/sisinflab/LinkedDatasets/tree/master).

- NEED TO ADD MORE DESCRIPTION ABOUT THE FILE CONTENTS

<!-- ```bash
https://drive.google.com/u/0/uc?id=1iKxaYhd_33yH0LtcZuO7Nf0yFcHFQXmI&export=download
``` -->

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


### Evaluation

- Currently the following metrics are supported:

    - `prediction shift`: The prediction shift is a way of quantifying how well a pushed (or nuked) item has been shifted in a direction favoring its goal.
    - `hit ratio`: Hit ratio is a measure of whether the pushed item made it to the top-k list (or was removed from the top-k list).
<!-- 
```bash

``` -->


## Experiments

### Experiment flow

    -choose dataset
        -load data
        -list most popular items
        -list most unpopular items

        - for each target item
            -choose similarity measure
            -generate similar items : save to file

        -choose similarity measure
            -generate pre-attack similarities : save to file

            -choose recommender system
                -generate pre-attack recommendations : save to file
                -calculate pre-attack hit ratio : save to result

                -choose attack
                    -for each attack size (and fixed filler size)
                        -for each filler size (and best attack size)
                            -generate attack profiles : save to file
                            -generate post-attack similarities : save to file
                            -generate post-attack recommendations : save to file
                            -calculate post-attack hit ratio : save to result
                            -calculate prediction shift with pre-attack recommendations : save to result
                    -generate graph of (prediction shift, hit ratio) vs attack size : save to result
                    -generate graph of (prediction shift, hit ratio) vs filler size : save to result

                    -choose best attack size and best filler size
                    -choose detection method (using best attack and filler size)
                        -generate detected attack profiles : save to file
                        -generate post-detection similarities : save to file
                        -generate post-detection recommendations : save to file
                        -calculate post-detection hit ratio : save to result
                        -calculate prediction shift with pre-attack recommendations : save to result
                        -calculate detection accuracy : save to result

### Results File Structure

    {OUTDIR}
        experiment_results_{exp no.}
            log.txt
            {dataset}
                {NUM_TARGET_ITEMS}_popular_items.csv                > (item, avg_rating)
                {NUM_TARGET_ITEMS}_unpopular_items.csv              > (item, avg_rating)
                similarities
                    kg_item_similarity_matrix.csv                    > (item1, item2, similarity)
                    pre_attack
                        {item_item or user_user}_{similarity measure}.csv   > (item1, item2, similarity) or (user1, user2, similarity)
                    post_attack
                        {attack}
                            {item_item or user_user}_{similarity measure}.csv
                            {item_item or user_user}_{similarity measure}_{attack size}_{filler size}.csv    
                            {item_item or user_user}_{similarity measure}_{attack size}_{filler size}.csv
                    post_detection
                        {item_item or user_user}_{similarity measure}.csv
                attack_profiles
                    {attack}
                        shilling_profiles_{attack size}_{filler size}.csv            > (user, item, rating)
                        shilling_profiles_{attack size}_{filler size}_detected.csv   > tbd
                {recommender system}
                    recommendations      
                        pre_attack_{similarity measure}_recommendations.csv          > (user, item, rating)
                        {attack}
                            post_attack_{similarity measure}_recommendations.csv
                            post_attack_{similarity measure}_{attack size}_{filler size}_recommendations.csv    
                            post_attack_{similarity measure}_{attack size}_{filler size}_recommendations.csv
                            post_detection_{similarity measure}_{attack size}_{filler size}_{detection}_recommendations.csv
                    detections
                        {detector}
                            {attack}_attack_{similarity measure}_{attack size}_{filler size}_detected_profiles.csv
                    graphs
                        {attack}_attack_size_vs_hit_ratio.png
                        {attack}_attack_size_vs_pred_shift.png
                        {attack}_filler_size_vs_hit_ratio.png
                        {attack}_filler_size_vs_pred_shift.png
                    results
                        hit_ratio
                            pre_attack_{similarity measure}_hit_ratio.csv       > (among_first, hit_ratio)
                            post_attack_{similarity measure}_hit_ratio.csv      > (among_first, hit_ratio, attack_size, filler_size, attack)
                            post_detection_{similarity measure}_hit_ratio.csv   > (among_first, hit_ratio, attack_size, filler_size, attack, detection)
                        pred_shift  > tbd
                

- Run the following command to run the experiments.
- Use breakpoint and version arguments to run the experiments in parts.

```bash
    python experiment.py
```

## Citation

If you find this repository useful, please cite our paper.

```bibtex
```