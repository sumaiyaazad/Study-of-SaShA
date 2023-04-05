import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)  
sys.path.append(parent_directory)


from utils.data_loader import *

# create a folder to save the results
OUTPUT_PATH = 'data_statistics'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Load data
data, users, items = load_data_ml_1M()
print('data', data.shape)
print('users', users.shape)
print('items', items.shape)

print('-'*50, 'data loaded', '-'*50)

# users vs ratings distribution --------------------------------------------------- #

temp = data.groupby('user_id')['rating'].count()
temp.name = 'num_ratings'
users = users.merge(temp, left_on='user_id', right_on='user_id')

# sort users by number of ratings
users_sorted = users.sort_values(by=['num_ratings'], ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(range(len(users['num_ratings'])), users_sorted['num_ratings'])
plt.xlabel('Users')
plt.ylabel('Number of ratings')
plt.title('Users vs Number of ratings')
# plt.show()
plt.savefig(OUTPUT_PATH + '/users_vs_ratings.png')
print('-'*50, 'users vs ratings saved', '-'*50)

# items vs ratings distribution --------------------------------------------------- #

temp = data.groupby('item_id')['rating'].count()
temp.name = 'num_ratings'
items = items.merge(temp, left_on='item_id', right_on='item_id')

# sort items by number of ratings
items_sorted = items.sort_values(by=['num_ratings'], ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(range(len(items['num_ratings'])), items_sorted['num_ratings'])
plt.xlabel('Items')
plt.ylabel('Number of ratings')
plt.title('Items vs Number of ratings')
# plt.show()
plt.savefig(OUTPUT_PATH + '/items_vs_ratings.png')
print('-'*50, 'items vs ratings saved', '-'*50)

# ratings distribution --------------------------------------------------- #

plt.figure(figsize=(10, 5))
plt.hist(data['rating'], bins=5)
plt.xlabel('Rating')
plt.ylabel('Number of ratings')
plt.title('Ratings distribution')
# plt.show()
plt.savefig(OUTPUT_PATH + '/ratings_distribution.png')
print('-'*50, 'ratings distribution saved', '-'*50)

# generate a list of least and most rated 50 items --------------------------------------------------- #

temp = data.groupby('item_id')['rating'].mean()
temp.name = 'avg_rating'
items = items.merge(temp, left_on='item_id', right_on='item_id')
items_sorted = items.sort_values(by=['avg_rating'], ascending=True)

# save the list of least rated 50 items
LEAST_RATED_NUM = 50
items_sorted.head(LEAST_RATED_NUM).to_csv(OUTPUT_PATH + '/least_rated_' + str(LEAST_RATED_NUM) + '_items.csv', index=False)
print('-'*50, 'least rated', LEAST_RATED_NUM, 'items saved', '-'*50)

# save the list of most rated 50 items
MOST_RATED_NUM = 50
items_sorted.tail(MOST_RATED_NUM).to_csv(OUTPUT_PATH + '/most_rated_' + str(MOST_RATED_NUM) + '_items.csv', index=False)
print('-'*50, 'most rated', MOST_RATED_NUM, 'items saved', '-'*50)

