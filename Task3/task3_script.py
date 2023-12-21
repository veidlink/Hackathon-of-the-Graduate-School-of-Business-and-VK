import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

data_train = pd.read_csv('train (3).csv')
data_test = pd.read_csv('test (3).csv')

user_item_matrix = data_train.pivot(index = 'user_id', columns = 'item_id', values = 'like')
user_item_matrix = user_item_matrix.fillna(-1) # filling with -1 empirically gives the best result

# As a first step we normalize the user vectors to unit vectors.

# magnitude = norm of each vector
magnitude = np.sqrt(np.square(user_item_matrix).sum(axis=1))

# normalizing item vectors
user_item_matrix = user_item_matrix.divide(magnitude, axis='index')

def calculate_similarity(user_item_matrix):
    """Calculate the column-wise cosine similarity for a sparse
    matrix. Return a new dataframe matrix with similarities.
    """
    data_sparse = sparse.csr_matrix(user_item_matrix)
    similarities = cosine_similarity(data_sparse.T) # also tried spearman and kendall correlation - gave worse results
    sim = pd.DataFrame(data=similarities, index = user_item_matrix.columns, columns= user_item_matrix.columns)
    return sim


def recommend(user_item_matrix, user_index):
    # Get the artists the user has liked.
    known_user_likes = user_item_matrix.iloc[user_index]
    known_user_likes = known_user_likes[known_user_likes >= 0].index.values

    # Users likes for all items as a sparse vector.
    user_rating_vector = user_item_matrix.iloc[user_index]

    # Calculate the score.
    score = data_matrix.dot(user_rating_vector).div(data_matrix.sum(axis=1))

    # Remove the known likes from the recommendation.
    score = score.drop(known_user_likes)
    return score.nlargest(20).index

# Build the similarity matrix
data_matrix = calculate_similarity(user_item_matrix)

recommendations = {}
for index, row in data_test.iterrows():
    user_id = row['user_id']
    recommendations[user_id] = list(recommend(user_item_matrix, user_id))

recs_item = pd.DataFrame(recommendations).T.reset_index().rename({'index': 'user_id'}, axis = 1)
recs_item.to_csv('submission3_10.csv', index = False)






