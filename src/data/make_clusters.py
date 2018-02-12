# Hide deprecation warnings
import warnings
warnings.filterwarnings('ignore')

# Common imports
import pandas as pd
import numpy as np

#Clustering imports
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# To plot pretty figures
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def createClusters(df_aisles, df_orders, df_products, df_order_products__prior):

    print("Creating clusters...")

    data = pd.merge(pd.merge(pd.merge(df_order_products__prior, df_products, on="product_id"), df_orders, on="order_id"), df_aisles, on="aisle_id")

    data_user_dept = pd.crosstab(data['user_id'], data['department_id'])

    #Using the Principal Component Analysis in the crosstab dataframe will reduce the number of aisles to any given number.
    pca = PCA(n_components=10)
    pca.fit(data_user_dept)
    pca_samples = pca.transform(data_user_dept)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    var_exp = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    plt.bar(range(10), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(10), cumsum, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show();

    ps = pd.DataFrame(pca_samples)

    plt.figure(figsize=(16,16))

    i=1
    for column1 in range(0,5):
        for column2 in range(0,5):
            if column1!=column2:
                ax = plt.subplot(5,4,i)
                ax.scatter(ps[column1], ps[column2], c='dimgrey', s=20)
                plt.axis('off')
                plt.title('PCA {}/PCA {}'.format(column1,column2))
                i+=1

    plt.show();

    PCA3D = plt.figure(figsize=(10, 8))
    ax = PCA3D.add_subplot(111, projection='3d')

    ax.scatter(ps[0], ps[1], ps[2], c='dimgray')
    plt.show();

    # Using all the 10 PCA components to capture 96% of the variance.
    #Reduce the matrix for the sake of memory
    matDendogram = ps[:10000]

    link_mat = linkage(matDendogram, 'ward')
    # print("link_mat shape",link_mat.shape)
    clusters = fcluster(link_mat, 10, criterion='maxclust')

    # "Elbow" method for Hierarchical Clustering
    last_merges = link_mat[-10:, 2]
    last_merges_rev = last_merges[::-1]
    idx = np.arange(1, len(last_merges) + 1)
    plt.plot(idx, last_merges_rev, label='Distance')

    acceleration = np.diff(last_merges, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idx[:-2] + 1, acceleration_rev, label='Acceleration')
    plt.legend()
    plt.show();

    # Plotting a dendrogram
    plt.figure(figsize=(12,10))

    ddata = dendrogram(link_mat, leaf_rotation=90., no_labels=True, color_threshold=50)
    dist_cut = findDendrogramCut(ddata, 3)
    plt.axhline(y=dist_cut, linestyle='--', c='dimgray')
    plt.text(np.min(ddata['icoord'])*100, dist_cut+2, 'n_clusters = 3', size=18, color='dimgray')
    plt.title('HAC Dendrogram',)
    plt.xlabel('Data points')
    plt.ylabel('Distance')
    plt.show();

    kRange = range(1,12)
    testKmean = test(matDendogram, kRange)

    plt.figure(figsize=(12,10))

    plt.plot(kRange, testKmean, 'o-', color='darkblue', alpha=0.8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia vs KMean Parameter')
    plt.annotate('Knee at k=3', xy=(3, testKmean[2]), xytext=(5,testKmean[1]),
                 size=14, weight='bold', color='dimgray',
                 arrowprops=dict(facecolor='dimgray', shrink=0.05))
    plt.show();

    # Balancing Inertia with Entropy
    testRes = newTest(matDendogram, range(1,30))

    plt.figure(figsize=(16,12))

    norm_inertias = testRes['inertias'][0]
    norm_entropies = - np.log2(1./len(matDendogram))

    ax1 = plt.subplot('221')
    ax1.plot(range(1,30),testRes['inertias'], 'o-', color='darkblue', alpha=0.8)
    plt.title('Inertia')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum-of-Squares')

    ax2 = plt.subplot('222')
    ax2.plot(range(1,30), testRes['entropies'], 'o-', color='darkblue', alpha=0.8)
    plt.title("Shannon's Entropy")
    plt.xlabel('Number of clusters')
    plt.ylabel('Entropy')

    ax3 = plt.subplot('223')
    ax3.plot(range(1,30), testRes['inertias']/norm_inertias + testRes['entropies']/norm_entropies, \
             'o-', color='darkblue', alpha=0.8)
    plt.title('Combined Cost Function')
    plt.xlabel('Number of clusters')
    plt.ylabel('Cost')

    plt.annotate('Possible clusters, 3 & 5', xy=(3.5, 0.48), xytext=(10, 0.6),
                 size=14, weight='bold', color='dimgray',
                 arrowprops=dict(facecolor='dimgray', shrink=0.05))
    plt.annotate('', xy=(5.2, 0.43), xytext=(10, 0.6),
                 size=14, weight='bold', color='dimgray',
                 arrowprops=dict(facecolor='dimgray', shrink=0.05))

    plt.show();

    # From now on, we'll stick with 3 clusters, which was the clearer number of clusters based on the previous analysis.
    clusterer = KMeans(n_clusters=3).fit(ps)
    centers = clusterer.cluster_centers_
    c_preds = clusterer.predict(ps)

    colors = ['blue','green','yellow']
    colored = [colors[k] for k in c_preds]

    fig = plt.figure(figsize=(12,10))
    colors = ['blue','green','yellow']
    colored = [colors[k] for k in c_preds]

    matXY = ps[[0,1,2]].as_matrix()

    plt.scatter(matXY[:50000,0], matXY[:50000,1],  color = colored)
    for ci,c in enumerate(centers):
        plt.plot(c[0], c[1], 'o', markersize=8, color='red')

    plt.xlabel('PCA 0')
    plt.ylabel('PCA 1')
    plt.show();

    PCA3D = plt.figure(figsize=(10, 8))
    ax = PCA3D.add_subplot(111, projection='3d')

    matXYZ = ps[[0,1,2]].as_matrix()

    ax.scatter(matXYZ[:10000,0], matXYZ[:10000,1], matXYZ[:10000,2], alpha=0.3,color = colored[:10000])

    ax.set_xlabel('PCA 0')
    ax.set_ylabel('PCA 1')
    ax.set_zlabel('PCA 2')

    for c in centers:
        ax.scatter(c[0], c[1], c[2], 'o', color='red')

    plt.show();

    #Let's finally return the predictions for all the users
    print("Obtaining cluster for all users...")

    df_orders['cluster'] = df_orders.apply(lambda row: c_preds[row.user_id-1], axis=1)

    return df_orders

# Computing Akaike and Bayes Information Criteria
def kmeansIC(model):
    k, m = model.cluster_centers_.shape
    n, = model.labels_.shape
    D = model.inertia_
    return {'AIC': D + 2*m*k, 'BIC': D + np.log(n)*m*k}

# Computing Renyi and Shannon Entropies
def kmeansRenyi(model):
    n, = model.labels_.shape
    nClusters, m = model.cluster_centers_.shape
    max_p = np.max([ model.labels_[model.labels_ == k].shape[0] for k in range(nClusters)]) / float(n)
    return - np.log2(max_p)

def kmeansEntropy(model):
    n, = model.labels_.shape
    nClusters, m = model.cluster_centers_.shape
    p = np.array([ model.labels_[model.labels_ == k].shape[0] for k in range(nClusters)]) / float(n)
    return - np.sum(p * np.log2(p))

# Small test function to calculate all our model estimators
# in function of the number of cluster in k-means.
def newTest(data, nClusterRange):
    nClusters = len(nClusterRange)
    inertias = np.zeros(nClusters)
    BICs = np.zeros(nClusters)
    AICs = np.zeros(nClusters)
    Hrenyis = np.zeros(nClusters)
    entropies = np.zeros(nClusters)
    for i in range(len(nClusterRange)):
        model = KMeans(n_clusters=i+1, init='k-means++').fit(data)
        inertias[i] = model.inertia_
        AICs[i] = kmeansIC(model)['AIC']
        BICs[i] = kmeansIC(model)['BIC']
        Hrenyis[i] = kmeansRenyi(model)
        entropies[i] = kmeansEntropy(model)
    return {'inertias': inertias, 'AICs': AICs, 'BICs': BICs, 'Hrenyis': Hrenyis, 'entropies': entropies}

# inspect the inertia of the different number of clusters.
def test(data, nClusterRange):
    inertias = np.zeros(len(nClusterRange))
    for i in range(len(nClusterRange)):
        model = KMeans(n_clusters=i+1, init='k-means++').fit(data)
        inertias[i] = model.inertia_
    return inertias

# Find the distance at which to cut off the dendrogram for
# a given number of clusters
def findDendrogramCut(ddata, n_cluster):
    hierarch_dist = np.matrix(ddata['dcoord'])
    sorted_hierarch_dist = hierarch_dist[np.argsort(hierarch_dist.A[:,1])]
    d_cut = sorted_hierarch_dist[-n_cluster:-n_cluster+2][:,1].sum() / 2.
    return d_cut
