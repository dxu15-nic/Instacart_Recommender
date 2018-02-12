# Hide deprecation warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import itertools
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as linalg

def product_recommender(df_order_products__prior, df_order_products__train, df_orders, df_products):

    print("Starting Recommender systems...")

    # Data Preparation
    df_orders_test = df_orders.loc[(df_orders.eval_set == "train")].reset_index()
    df_orders_test = df_orders_test[["order_id", "user_id", 'cluster']]
    df_test = df_order_products__train[["order_id", "product_id"]]
    df_test = df_test.groupby("order_id")["product_id"]\
                         .apply(list).reset_index().rename(columns={"product_id": "products"})
    df_test = pd.merge(df_orders_test, df_test, on="order_id")
    df_test = df_test[["user_id", "products", "cluster"]]

    # Users prior purchases per product
    df_orders_train = df_orders.loc[df_orders.eval_set == "prior"]
    df_orders_train = df_orders_train[["order_id", "user_id", "cluster"]]
    df_train = pd.merge(df_orders_train, df_order_products__prior[["order_id", "product_id"]],\
                                 on="order_id")
    df_train = df_train[["user_id", "product_id", "cluster"]]
    df_train = df_train.groupby(["user_id", "product_id", "cluster"])\
                                                      .size().reset_index().rename(columns={0:"quantity"})

    # Utility Matrices
    clusternumber = len(df_train.cluster.unique())
    cluster = []

    for i in range(clusternumber):
        cluster.append(df_train.loc[df_train['cluster'] == i].drop('cluster',axis=1))

    for i in range(clusternumber):
        cluster[i]["user_id"] = cluster[i]["user_id"].astype("category")
        cluster[i]["product_id"] = cluster[i]["product_id"].astype("category")

        utility_matrix = []

    for i in range(clusternumber):
        utility_matrix.append(coo_matrix((cluster[i]["quantity"],
                                         (cluster[i]["product_id"].cat.codes.copy(),
                                          cluster[i]["user_id"].cat.codes.copy()))))

    for i in range(clusternumber):
        print("Utility matrix {} shape: {}".format(i,utility_matrix[i].shape))

    utility_matrix_T = []

    for i in range(clusternumber):
        utility_matrix_T.append(utility_matrix[i].T.tocsr())

    # Let's create users and products dictionaries for future ease of use
    users = []

    for i in range(clusternumber):
        users.append({uid:i for i, uid in enumerate(cluster[i]["user_id"].cat.categories)})

    products = []

    for i in range(clusternumber):
        products.append(dict(enumerate(cluster[i]["product_id"].cat.categories)))

    # Popular products
    popular_products = list(df_order_products__prior["product_id"].value_counts().head(10).index)
    print("Most popular products:")
    print(df_products.product_name.loc[df_products.product_id.isin(popular_products)].reset_index(drop=True))

    # Recommendation with user to user similarity
    # We will use an example user: User ID 1
    user_ex = 1
    cluster = df_train.cluster.loc[df_train.user_id == user_ex].unique()[0]

    # Get top similar users
    similarities = cosine_similarity(utility_matrix_T[cluster][users[cluster][1]],utility_matrix_T[cluster])

    ids = np.argpartition(similarities[0], -11)[-11:]
    best = sorted(zip(ids, similarities[0][ids]), key=lambda x: -x[1])[1:]

    # Let's check if they're really similar
    ex_user_products = set(utility_matrix_T[cluster][ids[-1]].nonzero()[1])
    print("User products history:")
    print(df_products.product_name.loc[df_products.product_id.isin(ex_user_products)].reset_index(drop=True))

    similar_user_products = set(utility_matrix_T[cluster][ids[-2]].nonzero()[1])
    print("Most similar user products history:")
    print(df_products.product_name.loc[df_products.product_id.isin(similar_user_products)].reset_index(drop=True))

    print("Recall:",len(similar_user_products.intersection(ex_user_products)) / len(similar_user_products))

    # Let's get now the product recommendations
    ids = ids[:-1]
    if len(df_test.products.loc[df_test.user_id == user_ex])>0:
        products_in_basket = df_test.products.loc[df_test.user_id == user_ex].tolist()[0]
    else:
        products_in_basket = []
    final_recommendations = []
    final_valuation = []

    for i in range(len(ids)):
        similar_users_products = utility_matrix_T[cluster][ids[i]].nonzero()[1]

        #Mask to filter products already in the user's cart
        mask = np.isin(similar_users_products, products_in_basket, invert=True)
        for j in range(len(similar_users_products[mask])):
            if np.isin(similar_users_products[mask][j], final_recommendations, invert=True):
                final_recommendations.append(similar_users_products[mask][j])
                final_valuation.append(best[-(i+1)][1])
            else:
                index = final_recommendations.index(similar_users_products[mask][j])
                final_valuation[index]+= best[-(i+1)][1]

    final_recommendations = np.asarray(final_recommendations)
    final_valuation = np.asarray(final_valuation)

    ind = heapq.nlargest(min(10,len(final_recommendations)), range(len(final_valuation)), final_valuation.take)
    final_recommendations = final_recommendations[ind]

    print("Recommended products:")
    print(df_products.product_name.loc[df_products.product_id.isin(final_recommendations)].reset_index(drop=True))

    # Let's do it now for the rest of the users, or a sample of them
    subset = 0.05 #We will make the predictions only in 5% of the data
    df_test = df_test.sample(n=int(len(df_test) * subset)).reset_index(drop=True)

    def rec_user2user(row):
        cluster = row['cluster']
        similarities = cosine_similarity(utility_matrix_T[cluster][users[cluster][row["user_id"]]]\
                                         ,utility_matrix_T[cluster])
        ids = np.argpartition(similarities[0], -11)[-11:]
        best = sorted(zip(ids, similarities[0][ids]), key=lambda x: -x[1])[1:]

        ids = ids[:-1]

        if len(df_test.products.loc[df_test.user_id == row['user_id']])>0:
            products_in_basket = df_test.products.loc[df_test.user_id == row['user_id']].tolist()[0]
        else:
            products_in_basket = []

        final_recommendations = []
        final_valuation = []

        for i in range(len(ids)):
            similar_users_products = utility_matrix_T[cluster][ids[i]].nonzero()[1]
            #Mask to filter products already in the user's cart
            mask = np.isin(similar_users_products, products_in_basket, invert=True)
            for j in range(len(similar_users_products[mask])):
                if np.isin(similar_users_products[mask][j], final_recommendations, invert=True):
                    final_recommendations.append(similar_users_products[mask][j])
                    final_valuation.append(best[-(i+1)][1])
                else:
                    index = final_recommendations.index(similar_users_products[mask][j])
                    final_valuation[index]+= best[-(i+1)][1]

        final_recommendations = np.asarray(final_recommendations)
        final_valuation = np.asarray(final_valuation)

        ind = heapq.nlargest(min(10,len(final_recommendations)), range(len(final_valuation)), final_valuation.take)
        final_recommendations = set(final_recommendations[ind])

        return final_recommendations

    print("Running the recommendations (User to user model) ...")
    df_test['Recommendations'] = df_test.apply(rec_user2user, axis=1)
    df_test = df_test[['user_id','cluster','products','Recommendations']]
    df_test.columns = ['User','Cluster','Products in basket','U2U Recommendations']

    print(df_test.sort_values('User').head())

    # SVD Factorization
    user_ex = 1
    cluster_ex = df_train.cluster.loc[df_train.user_id == user_ex].unique()[0]

    # We'll start by factorizing the utility matrix using SciPy's SVD
    user_factors = []
    product_factors = []
    singular_values = []

    for cluster in range(clusternumber):
        utility_matrix_T[cluster] = utility_matrix_T[cluster].astype(np.float32)
        user_factor, singular_value, product_factor = linalg.svds(utility_matrix_T[cluster], 10)

        # User factored stored directly with a user*factor format
        user_factors.append(user_factor*singular_value)
        product_factors.append(product_factor)
        singular_values.append(singular_value)

    scores =  user_factors[cluster_ex][users[cluster_ex][user_ex]].dot(product_factors[cluster_ex])
    best = np.argpartition(scores, -10)[-10:]
    recommendations_all = sorted(zip(best, scores[best]), key=lambda x: -x[1])

    print("Recommended products:")
    print(df_products.product_name.loc[df_products.product_id.isin(best)].reset_index(drop=True))

    # But some of those products might be already in the users basket, so we should get rid of them
    bought_indices = utility_matrix_T[cluster_ex][users[cluster_ex][user_ex]].nonzero()[1]
    count = 10 + len(bought_indices)
    ids = np.argpartition(scores, -count)[-count:]
    best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])

    recommendations_new = list(itertools.islice((rec for rec in best if rec[0] not in bought_indices), 10))

    print("Recommended products:")
    recommendations = []
    for recommendation in recommendations_new:
        recommendations.append(recommendation[0])

    print(df_products.product_name.loc[df_products.product_id.isin(recommendations)].reset_index(drop=True))

    def rec_SVD(row):
        cluster = row['Cluster']

        scores =  user_factors[cluster][users[cluster][row['User']]].dot(product_factors[cluster])
        bought_indices = utility_matrix_T[cluster][users[cluster][row['User']]].nonzero()[1]
        count = 10 + len(bought_indices)
        ids = np.argpartition(scores, -count)[-count:]
        best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])

        recommendations_new = list(itertools.islice((rec for rec in best if rec[0] not in bought_indices), 10))

        recommendations = []
        for recommendation in recommendations_new:
            recommendations.append(recommendation[0])

        final_recommendations = set(recommendations)

        return final_recommendations

    # Now, let's do it for the already sampled portion of the dataset, df_test
    print("Running the recommendations (SVD model) ...")
    df_test['SVD Recommendations'] = df_test.apply(rec_SVD, axis=1)
    print(df_test.head())

    # Recall between user to user recommendation and SVD matrix factorization
    def methods_recall(row):
        return len(row['U2U Recommendations'].intersection(row['SVD Recommendations'])) / len(row['U2U Recommendations'])
        
    df_test['Methods Recall'] = df_test.apply(methods_recall, axis=1)
    print("U2U and SVD recommendations recall: {:.2f}%".format(df_test['Methods Recall'].mean() * 100))
