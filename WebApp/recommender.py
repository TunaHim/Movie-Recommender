# %
""" Contains various recommendation implementations all algorithms return a list of movieids """

# Import dependencies
import pickle
import pandas as pd
import numpy as np
from utils import movies
from sklearn.metrics.pairwise import cosine_similarity

# %
# All these files are created from the EDA.ipynb jupyter notebook. 
# Open the Saved Model
with open('../savedFiles/nmf_MovieModel.pkl','rb') as file:
    loaded_MovieModel = pickle.load(file)

# Open mean file
with open('../savedFiles/mean_movies.pkl','rb') as file:
    mean_movies = pickle.load(file)

# Open the merged movies and rating file
df_MovRat = pd.read_csv('../savedFiles/ratmov.csv', sep=';')

# Define functions for the recommender systems
# All the recommenders suggests 3 movies (k=3) only. One can change it too to less/more recommendations.

# Function: 'Random' Recommender  
def recommend_random(k=3):
    return movies['title'].sample(k).to_list()

# Function: 'Non Negative Matrix Factorization' Recommender
def recommend_with_NMF(query, k=3):

    """Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids."""

    # 1. candidate generation
    # 2. construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(query, columns=mean_movies.index, index=["new_user"])
    new_user_dataframe.fillna(mean_movies, inplace=True)
    
    # 3. scoring
    Q_matrix = loaded_MovieModel.components_                                            # Get the movie-feature dataframe/matrix 'Q' from the model components
    #Q = pd.DataFrame(data=Q_matrix)#, columns=title)                                   # to dataframe for better visualization

    P_matrix = loaded_MovieModel.transform(new_user_dataframe)                          # Get the user-feature matrix/dataframe 'P' by using the model method 
                                                                                        # transform on the full imputed user/movie dataframe/matrix
    
    R_hat_matrix = np.dot(P_matrix, Q_matrix)                                           # Get the dot product of P and Q

    R_hat =pd.DataFrame(R_hat_matrix, columns=mean_movies.index, index=["new_user"])    # Get the Ratings_reconstructed dataframe
   
    # 4. ranking
    sorted_list = R_hat.transpose().sort_values(by="new_user", ascending=False).index.to_list()
    
    # filter out movies already seen by the user
    rated_movies = list(query.keys())
    recommendations = [movie for movie in sorted_list if movie not in rated_movies]
    
    # return the top-k highest rated movie ids or titles    
    return recommendations[:k]
   
ratings = df_MovRat[['title','rating','userId']].pivot_table(index='userId', columns='title', values='rating')

# Function: 'Cosine Similarity' Recommender
def recommend_neighborhood(query,k=3):

    # transpose ratings matrix
    initial = ratings.T

    # add new query to user item matrix
    query_df = pd.DataFrame(query.values(), index=query.keys(),columns=['new_user'])
    user_item = initial.merge(query_df,how='left',left_index=True,right_index=True)

    # unseen movies
    unseen_movies = user_item[user_item['new_user'].isna()].index
    
    # fill na for cosine similarity
    user_item = user_item.fillna(0)
    
    # create cosine similarity matrix
    user_user = cosine_similarity(user_item.T)
    user_user = pd.DataFrame(user_user, columns = user_item.columns, index = user_item.columns).round(2)
    
    # select 5 users with highest similarity (one can select more users too)
    top_five_users = user_user['new_user'].sort_values(ascending=False).index[1:6]
    movie_rec_list = list()
    ratio_list = list()
    for movie in unseen_movies:
        other_users = initial.columns[~initial.loc[movie].isna()]
        other_users = set(other_users)
        num = 0
        den = 0
        for other_user in other_users.intersection(set(top_five_users)):
            rating = user_item[other_user][movie]
            sim = user_user['new_user'][other_user]
            num = num + (rating*sim)
            den = den + sim + 0.0001
        ratio = num/(den + 0.0001)
        movie_rec_list.append(movie)
        ratio_list.append(ratio)
    out = pd.DataFrame({
        'movie':movie_rec_list,
        'ratio':ratio_list
    })
    out.sort_values('ratio',ascending=False,inplace=True)

    return list(out.iloc[:k]['movie'])