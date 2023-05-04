from utils.data_loader import *
from attacks.semantic_attack import *
from attacks.sasha_random import *
from attacks.sasha_average import *
from attacks.sasha_segment import *
from utils.notification import *
from utils.misc import *
from utils.sendmail import *
import config as cfg
import argparse

def main():
    
    if args.dataset == 'yahoo_movies':
        data, user_data, item_data = load_data_yahoo_movies()
        kg, features, _ = load_kg_yahoo_movies(item_data)
    elif args.dataset == 'SmallLibraryThing':
        data, user_data, item_data = load_data_SmallLibraryThing()
        kg, features, _ = load_kg_SmallLibraryThing(item_data)
    else:
        raise ValueError('Dataset not found.')

    
    print('*'*10, 'Starting experiment', '*'*10)
    print('Dataset: {}'.format(args.dataset))
    print('Number of users: {}'.format(len(user_data)))
    print('Number of items: {}'.format(len(item_data)))
    print('Attack: {}'.format(args.attack))
    print()


    # output = OUTDIR + 'attack_name_dataset_target_item_id.csv'
    output = OUTDIR + 'shilling_profiles/{0}_{1}_{2}.csv'.format(args.attack, args.dataset, args.target_id)

    # create directory if it doesn't exist
    if not os.path.exists(OUTDIR + 'shilling_profiles/'):
        os.makedirs(OUTDIR + 'shilling_profiles/')

        
    (r_min, r_max) = RATING_RANGE[args.dataset]
    similarity_filelocation = 'output/shilling_profiles/' + args.similarity_filelocation
    item_feature_matrix = get_item_feature_matrix(kg, item_data, features)

    # create attack object
    if args.attack == 'sasha_random':
        attack_generator = SAShA_RandomAttack(  data=data, 
                                                r_max=r_max,
                                                r_min=r_min,
                                                similarity=KG_SIMILARITY,
                                                kg_item_feature_matrix=item_feature_matrix,
                                                similarity_filelocation=similarity_filelocation,
                                                attack_size_percentage=args.attack_size,
                                                filler_size_percentage=args.filler_size)

    elif args.attack == 'sasha_average':
        attack_generator = SAShA_AverageAttack( data=data,
                                                r_max=r_max,
                                                r_min=r_min,
                                                similarity=KG_SIMILARITY,
                                                kg_item_feature_matrix=item_feature_matrix,
                                                similarity_filelocation=similarity_filelocation,
                                                attack_size_percentage=args.attack_size,
                                                filler_size_percentage=args.filler_size)
    elif args.attack == 'sasha_segment':
        attack_generator = SAShA_SegmentAttack( data=data,
                                                r_max=r_max,
                                                r_min=r_min,
                                                similarity=KG_SIMILARITY,
                                                kg_item_feature_matrix=item_feature_matrix,
                                                similarity_filelocation=similarity_filelocation,
                                                attack_size_percentage=args.attack_size,
                                                filler_size_percentage=args.filler_size,
                                                select_size_percentage=args.selected_size)
    else:
        raise ValueError('Attack not found.')

    # generate shilling profiles
    attack_generator.generate_profile([args.target_id], 0.25, output, verbose=args.verbose)

    print()
    sendmail('SAShA Detection test mail on new machine', 'shilling profiles generation done.\n{0}\n{1}\n{2}'.format(args.attack, args.dataset, args.target_id))  
    print('Experiment completed.')
    balloon_tip('SAShA Detection', 'shilling profiles generation done.\n{0}\n{1}\n{2}'.format(args.attack, args.dataset, args.target_id))


if __name__ == '__main__':
    
    # main command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SmallLibraryThing', help='dataset to use')
    parser.add_argument('--attack', type=str, default='sasha_random', help='attack to use')
    parser.add_argument('--target_id', type=int, default=1, help='target item id')
    parser.add_argument('--attack_size', type=float, default=0.1, help='attack size')
    parser.add_argument('--filler_size', type=float, default=1, help='filler size')
    parser.add_argument('--selected_size', type=float, default=0.8, help='selected size')
    parser.add_argument('--similarity_filelocation', type=str, default='kg_similarity.csv', help='similarity file location')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose mode')
    parser.add_argument('--not_level', type=int, default=0, help='notification level, 0: no notification, 1: only at the end, 2: at verbose mode')
    parser.add_argument('--sep', type=str, default=',', help='separator for output file')
    args = parser.parse_args()

    main()

