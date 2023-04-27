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

test
