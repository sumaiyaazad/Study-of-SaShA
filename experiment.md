
# Experiment flow

    -choose dataset
        -load data
        -list most popular items
        -list most unpopular items

        -choose similarity measure
            -generate pre-attack similarities : save to file

            -choose recommender system
                -generate pre-attack recommendations : save to file
                -calculate pre-attack hit ratio : save to result

                -choose attack
                    -for each attack size (and fixed filler size)
                        -generate attack profiles : save to file
                        -generate post-attack similarities : save to file
                        -generate post-attack recommendations : save to file
                        -calculate post-attack hit ratio : save to result
                        -calculate prediction shift with pre-attack recommendations : save to result
                    -generate graph of (prediction shift, hit ratio) vs attack size : save to result
                    -choose best attack size

                    -for each filler size (and best attack size)
                        -generate attack profiles : save to file
                        -generate post-attack similarities : save to file
                        -generate post-attack recommendations : save to file
                        -calculate post-attack hit ratio : save to result
                        -calculate prediction shift with pre-attack recommendations : save to result
                    -generate graph of (prediction shift, hit ratio) vs filler size : save to result
                    -choose best filler size 

                    -choose detection method (using best attack and filler size)
                        -generate detected attack profiles : save to file
                        -generate post-detection similarities : save to file
                        -generate post-detection recommendations : save to file
                        -calculate post-detection hit ratio : save to result
                        -calculate prediction shift with pre-attack recommendations : save to result
                        -calculate detection accuracy : save to result


# File Structure

    {OUTDIR}
        experiment_results_{exp no.}
            log.txt
            {dataset}
                {NUM_TARGET_ITEMS}_popular_items.txt
                {NUM_TARGET_ITEMS}_unpopular_items.txt
                similarities
                    pre_attack
                        {item_item or user_user}_{similarity measure}.csv
                    post_attack
                        {item_item or user_user}_{similarity measure}.csv
                    post_detection
                        {item_item or user_user}_{similarity measure}.csv
                {recommender system}
                    recommendations
                        pre_attack_{similarity measure}_recommendations.csv
                        {attack}
                            post_attack_{similarity measure}_recommendations.csv
                            attack_size_stat
                                post_detection_{similarity measure}_{attack size}_{filler size}_recommendations.csv
                            filler_size_stat
                                post_detection_{similarity measure}_{attack size}_{filler size}_recommendations.csv
                            post_detection_{similarity measure}_{attack size}_{filler size}_recommendations.csv
                    attack_profiles
                        {attack}_{attack size}_{filler size}.csv
                        {attack}_{attack size}_{filler size}_detected.csv
                    graphs
                        {attack}_attack_size_vs_hit_ratio.png
                        {attack}_attack_size_vs_pred_shift.png
                        {attack}_filler_size_vs_hit_ratio.png
                        {attack}_filler_size_vs_pred_shift.png
                    results
                        hit_ratio.csv   : tbd
                        pred_shift.csv  : tbd
                