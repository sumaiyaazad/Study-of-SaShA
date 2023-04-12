from utils.data_loader import *
from attacks.base_attack import *
from attacks.random import *
from attacks.average import *
from utils.notification import *
import config as cfg
import argparse

def main():
    
    if args.dataset == 'ml-1m':
        data, user_data, item_data = load_data_ml_1M()
    else:
        raise ValueError('Dataset not found.')
    
    
    data = data.reset_index(drop=True)
    user_data = user_data.reset_index(drop=True)
    item_data = item_data.reset_index(drop=True)

    
    print('*'*10, 'Starting experiment', '*'*10)
    print('Dataset: {}'.format(args.dataset))
    print('Number of users: {}'.format(len(user_data)))
    print('Number of items: {}'.format(len(item_data)))
    print('Attack: {}'.format(args.attack))
    print()


    # rand_attack = RandomAttack(data, cfg.R_MAX, cfg.R_MIN)

    if args.attack == 'random':
        attack = RandomAttack(data, cfg.R_MAX, cfg.R_MIN)
    elif args.attack == 'average':
        attack = AverageAttack(data, cfg.R_MAX, cfg.R_MIN)
    else:
        raise ValueError('Attack not found.')

    # output = OUTDIR + 'attack_name_dataset_target_item_id.csv'
    output = OUTDIR + 'shilling_profiles/{0}_{1}_{2}.csv'.format(args.attack, args.dataset, args.target_id)

    # create directory if it doesn't exist
    if not os.path.exists(OUTDIR + 'shilling_profiles/'):
        os.makedirs(OUTDIR + 'shilling_profiles/')

    # generate shilling profiles
    attack.generate_profile(args.target_id, 0, output)

    print()
    print('Experiment completed.')
    balloon_tip('SAShA Detection', 'shilling profiles generation done.\n{0}\n{1}\n{2}'.format(args.attack, args.dataset, args.target_id))


if __name__ == '__main__':
    
    # main command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset to use')
    parser.add_argument('--attack', type=str, default='random', help='attack to use')
    parser.add_argument('--target_id', type=int, default=1, help='target item id')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose mode')
    parser.add_argument('--not_level', type=int, default=0, help='notification level, 0: no notification, 1: only at the end, 2: at verbose mode')
    parser.add_argument('--sep', type=str, default=',', help='separator for output file')
    args = parser.parse_args()

    main()

