Instacart Recommender & Clustering Study
==============================

Recommending new products to Instacart clients

Project Organization
------------

    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── raw
    │   └── interim
    │
    ├── notebooks
    │
    ├── reports
    │   └── figures 
    │	    ├── Clustering
    │       └── EDA
    │
    └── src
        ├── data
        ├── models
        └── visualization 


--------
# Instacart Recommender Project Walkthrough

## Introduction

The goal of this project is to recommend the best possible products for each Instacart client, making use of their previous purchase history and the similarity to other Instacart clients. That will be performed first by clustering the clients in their similarity groups and via user to user cosine similarities and SVD similarity factorization. The number of the different client clusters will have to be found, and the number of recommended products will be 10 for each system and client, avoiding products already bought by the user in the past as well as products already in the client's cart at the momment.

## Getting Data

The data for this project is readily available at the [Instacart dataset website](https://www.instacart.com/datasets/grocery-shopping-2017). This dataset is provided for non-commercial use and was the base of a Kaggle comeptition on August 2017, although the goal of that competition was to predict the client's next purchase, which is a supervised machine learning problem, as opposed to this recommender system, which is an unsupervised machine learning problem.

There is a [GitHub data dictionary page](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b) available for a better understanding of the meaning of each column and table, as well as the table size. Nevertheless, I've created a simple database diagram for a quick review of the tables relationship:

![Database Diagram](reports/figures/database_diagram.jpg)

## Clients Clustering

After the dataset with the previous client purchases is merged with the products dataset, we can create a crosstab of the sum of purchases that each client has made in each department, aisle or even for each product. Due to memory allocation efficiency, it was chosen to crosstab with department ID, since there are only 21 departments and the resulting table was a manageable 4 million positions. After that, a PCA was performed on the crosstab to keep only the most important features, with the resulting explained variance results:

![PCA variance](reports/figures/Clustering/PCA_variance.png)

The first component dominates the rest with as much as 70.87% of the explained variance. After testing a different number of components, there were no clear clustes within the dataset using only 2 to 3 components that could be visually evident, as can be seen in the next graph:

![Components Plots](reports/figures/Clustering/Components_plots.png)

Hence, all the PCA components were used to make the best possible cluster separations, so a total of 97.59% of the variance can be explained with the 10 components used. With this, we can scan for distances between the different number of clusters and their own acceleration as the second derivative of that same distance:

![Elbow Finding](reports/figures/Clustering/elbow_finding.png)

Now we can see how the distance has a clear elbow at 3 clusters, where the acceleration has also a clear spike, so that is a possible number of clusters to be used in this dataset.

Just to be sure, a few other methods were used to corroborate for the optimum number of clusters, among them the inertia using the KMeans algorithm, with the inertia being the sum of squared distances to the centroid (or within-cluster sum of squares). Here we can see something similar to the previous analysis:

![KMeans Inertia](reports/figures/Clustering/kmeans_inertia.png)

We can also see the actual distances of the different cluster numbers and how 3 clusters seem a reasonable fit in this case by painting the dendogram, which encodes the order of consecutive merge-operations as well as the distance that separated the merged clusters:

![Dendogram](reports/figures/Clustering/dendogram.png)

Finally, having decided that there are different 3 clusters, we can see how the particular points are clustered within the three first PCA components. This graph also shows us that the first component, PCA 0, is dominating the rest of the components, since the division is mostly vertical to that component:

![Cluster Centers 3D](reports/figures/Clustering/cluster_centers_3D.png)

This clustering is stored and used for the EDA and recommender systems.

## Exploratory Data Analysis

The previous section has let us with the original data untouched and a new feature for each client, which is the client's cluster. We can perform the EDA using all this information to see how is the data distributed and what are the differences among the different clusters.

First, it's important to note that not all clusters are equal in length, with the Cluster 0 having most of the orders (1.75M), Cluster 1 being the smallest (under 500k orders) and Cluster 2 being in the middle (a little over 1M orders).

A number of visualizations can be seen within the notebooks, but let's show here too the most interesting ones, with the number of orders per client being one of the most informatives graph:

![Orders per User](reports/figures/EDA/orders_per_user.png)

The number of orders per client seems to be capped at 100 orders max. The shape of the distribution could be one of an exponential distribution, where lambda would be more than one based only on the form of the figure.

The different times to purchase used by the clients is also worth showing:

![Orders time heatmaps](reports/figures/EDA/orders_time_heatmaps.png)

We can see how Saturday and Sunday morning are the most common times for shopping, although Cluster 0 has a slightly bigger bias towards shoping on Sunday afternoons. But it doesn't seem that this habit clearly differs within clusters.

We could see next the number of days since the previous purchase:

![Days since previous order clusters](reports/figures/EDA/days_since_previous_order_clusters.png)

In this case there is a clear distinction within clusters. Cluster 0 consits in a bigger proportion of users that tend to have a wider time gap between purchases, and it could be argued that users in Cluster 1 are the most recurrent buyers based on the smaller tail and higher concentration towards lower number of days.

If we've seen that the client's cluster affects on their purchase frequency, does it also affect on their purchase size?

![Products per order clusters](reports/figures/EDA/products_per_order_clusters.png)

There seems to be a big difference indeed in basket size, with users from Cluster 1, which are those who almost never get to 30 days between ordres, also being the users with a bigger basket size on average at the same time, while less frequent users from Cluster 0 also have the smallest orders (in means of different products per order).

So clients from different clusters shop with different frequencies and have different cart sizes, altough they tend to buy more or less at the same times (mostly weekends), but do they buy the same products? We can see here, for the top 6 departments, how is the cluster distribution:

![Top Departments Cluster Pie](reports/figures/EDA/top_departments_cluster_pie.png)

Remembering that, roughly, Cluster 0 accounts for 50% of the product purchases, Cluster 1 for 15% and Cluster 2 for the remaining 35%, we can understand the big differences in products bought by the clients of each cluster. For instance, in dairy eggs, we can even find more purchases for Cluster 2 than for Cluster 1, being Cluster 2 much smaller. That is actually consistent with the more frequent purchases of Cluster 1 and 2, since it seems that they are buying more daily products. At the same time, clients from Cluster 0 that, remember, are the ones with a wider gap between purchases, seem to be very interested in products for personal care.

Lastly, we can see that all of this is consistent with the reorder ratio of the different clusters, as it shows that clients from Clusters 1 and 2, which consume more frequent products, have a higher reorder ratio than clients from Cluster 0:

![Reorder Ratio Cluster](reports/figures/EDA/reorder_ratio_cluster.png)

## Recommender Systems

The datasets provided and the clustering info was merged and a utility matrix with the info of number of products bought by each user was created for each cluster, to be able to use them separately and make better predictions for clients in each clusters, since as we saw, if clients from Cluster 0 make a lot of purchases from the personal care departments, it would be wiser to recommend them that kind of products than others from dairy eggs because of a skew made from clients from different clusters.

Two different methods were used to create a recommender system:

* Cosine similarity: Using the pairwise functions from SKLearn, and based on the utility matrices.
* SVD Factorization: Using SciPy's linalg function within the sparse function group.

We can show an example of the recommendations made, for the client with user ID = 1. This client was recommended these products with the first recommender:

- Grade A Pasteurized 2% Milkfat Lowfat Cottage ...
- Original String Cheese
- Red Velvet Cupcake Cake & Frosting Mix
- Riserva Ducale Chianti Classico
- Skin Trip Coconut Moisturizer
- Organic Vegetarian Pho Soup Starter
- San Marzano Tomatoes With Basil
- Organic Green Tea With Pomegranate & Acai
- Marshmallow Creme
- Whole Grain Hot Dog Buns

While with the SVD factorization, the recommended showed these results:

- Strawberry Squeeze Fruit Spread
- Original String Cheese
- Simply Organic Garlic Salt
- Sprouted Watermelon Seeds
- Beggin' Strips Bacon Flavor Dog Snacks
- Chicken Thigh
- Olive Bruschetta
- Organic Green Tea With Pomegranate & Acai
- Real Semi-Sweet Chocolate Chips
- Classic Mouthwash

We can see how the recall between both methods is, in this case, 10%, since the product "Original String Cheese" was recommended by both of them, out of a total of 49688 different products. On average, we can see that the recall between both methods is 9.03% indeed.
