# Hide deprecation warnings
import warnings
warnings.filterwarnings('ignore')

# Common imports
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
import missingno as msno
from statsmodels.graphics.mosaicplot import mosaic

#Clustering imports
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Load the csv files that we'll need for the EDA phase into Pandas dataframes
def visualize(df_aisles, df_departments, df_orders, df_products, df_order_products__train, df_order_products__prior):

    print("Starting EDA...")

    print(df_aisles.head())
    print(df_aisles.info())

    print(df_departments.head())
    print(df_departments.info())

    print(df_orders.head())
    print(df_orders.info())
    print(df_orders.describe())

    orders_split = df_orders.groupby("eval_set")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
    orders_split['Ratio'] = orders_split["Total_orders"].apply(lambda x: x /orders_split['Total_orders'].sum())
    print(orders_split)

    # Cluster order distribution
    cluster_number = df_orders["cluster"].value_counts()

    plt.figure(figsize=(8,6))
    sns.barplot(cluster_number.index, cluster_number.values, alpha=0.8, color='darkblue')
    plt.ylabel("Number of orders")
    plt.xticks(rotation="vertical")
    plt.title('Cluster distribution')
    plt.show();

    print(df_products.head())
    print(df_products.info())
    print(df_products.describe())

    print(df_order_products__train.head())
    print(df_order_products__train.info())
    print(df_order_products__train.describe())

    print(df_order_products__prior.head())
    print(df_order_products__prior.info())
    print(df_order_products__prior.describe())

    # Data preparation
    # Let's merge the prior orders information dataset with product, aisle and department information to get the whole info of what kind of products people buy.
    df_past_orders_info = pd.merge(pd.merge(pd.merge(df_order_products__prior, df_products, on="product_id",\
        how="left"), df_aisles, on="aisle_id", how="left"), df_departments, on="department_id", how="left")

    # We should also add to that dataset the cluster number of each order
    df_past_orders_cluster = df_past_orders_info.join(df_orders[['order_id','cluster']]\
                                                  .set_index('order_id'), on='order_id')

    # Data visualization
    # Orders per user
    orders_per_user = df_orders.groupby("user_id")["order_number"].aggregate(max).reset_index()\
                  .order_number.value_counts()

    plt.figure(figsize=(12,10))
    plt.bar(orders_per_user.index, orders_per_user.values, alpha=0.8, color='darkblue')
    plt.ylabel("Number of users")
    plt.title('Distribution of number of orders per user')
    plt.xlabel("Orders per user")
    plt.show()

    # Time of purchase
    fig, ax = plt.subplots(figsize=(12,10))
    sns.countplot(x="order_dow", data=df_orders, alpha=0.8, color='darkblue')
    plt.ylabel("Orders")
    plt.xlabel("Day of week")
    plt.title("Frequency distribution of orders per day of the week")
    labels = ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']
    ax.set_xticklabels(labels)
    plt.show()

    plt.figure(figsize=(12,10))
    sns.countplot(x="order_hour_of_day", data=df_orders, alpha=0.8, color='darkblue')
    plt.ylabel("Orders")
    plt.xlabel("Hour of day")
    plt.title("Frequency distribution of orders per hour of day")
    plt.show()

    number_figures = 3
    titles = ['Cluster 0','Cluster 1','Cluster 2']

    grid = GridSpec(3, 1)
    plt.figure(figsize=(8,12))

    for row in range(3):
        plt.subplot(grid[row, 0])
        heatmap_df = df_orders[df_orders.cluster==row].groupby(["order_dow", "order_hour_of_day"])\
                              ["order_number"].aggregate("count").reset_index()
        heatmap_df = heatmap_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

        sns.heatmap(heatmap_df)
        plt.title(titles[row])
        plt.ylabel("")
        plt.xlabel("")

    plt.xlabel('Hour of day')
    plt.show()

    # Days since prior order
    plt.figure(figsize=(12, 8))
    sns.countplot(x="days_since_prior_order", data=df_orders, alpha=0.8, color='darkblue')
    plt.ylabel("Orders")
    plt.xlabel('Days since last order')
    plt.xticks(rotation="vertical")
    plt.title("Days since last order distribution")
    plt.show()

    plt.figure(figsize=(10,8))
    sns.violinplot(x="cluster", y="days_since_prior_order", data=df_orders, split=True, inner="quart")
    plt.show();

    # Number of products per order
    plt.figure(figsize=(16,8))
    plt.subplot(1, 2, 1)

    # Train set
    order_product_count = df_order_products__train.groupby("order_id")["add_to_cart_order"]\
                          .aggregate(max).reset_index().add_to_cart_order.value_counts()
    plt.bar(order_product_count.index, order_product_count.values, alpha=0.8, color='darkblue')
    plt.ylabel('Orders')
    plt.title("Products per order in test set")
    plt.xlabel('Number of products')

    # Prior set
    plt.subplot(1, 2, 2)
    order_product_count = df_order_products__prior.groupby("order_id")["add_to_cart_order"]\
                          .aggregate(max).reset_index().add_to_cart_order.value_counts()
    plt.bar(order_product_count.index, order_product_count.values, alpha=0.8, color='darkblue')
    plt.ylabel('Orders')
    plt.title("Products per order in train set")
    plt.xlabel('Number of products')

    plt.show()

    baskets = []
    for i in range(len(cluster_number)):
        baskets.append(df_past_orders_cluster[df_past_orders_cluster.cluster==i]\
                       .groupby(["order_id"])["product_id"].aggregate("count").reset_index())
        baskets[i]['cluster'] = i
        print("Average number of products in baskets in cluster {}: {}".format(i,baskets[i].product_id.mean()))

    df_basket_size = pd.concat([baskets[0],baskets[1],baskets[2]])

    plt.figure(figsize=(10,8))
    sns.violinplot(x="cluster", y="product_id", data=df_basket_size, split=True, inner="quart")
    plt.show();

    #Most common products / departments / aisles:
    print(df_past_orders_info['product_name'].value_counts()[0:10])

    # Top 10 products distribution
    top_10_products = df_past_orders_info["product_name"].value_counts().head(10)

    plt.figure(figsize=(12,10))
    sns.barplot(top_10_products.index, top_10_products.values, alpha=0.8, color='darkblue')
    plt.ylabel("Orders")
    plt.xticks(rotation="vertical")
    plt.title('Top 10 Products')
    plt.show()

    # Most popular products per cluser
    number_figures = 3
    labels = ['Cluster 0','Cluster 1','Cluster 2']

    top_10_products = []
    for figure in range(number_figures):
        top_10_products.append(df_past_orders_cluster["product_name"][df_past_orders_cluster.cluster==figure]\
                               .value_counts().head(10))

    grid = GridSpec(1, 3)
    plt.figure(figsize=(16,4))

    for column in range(3):
        plt.subplot(grid[0, column])
        sns.barplot(top_10_products[column].index, top_10_products[column].values, alpha=0.8, color='darkblue')
        if column == 0:
            plt.ylabel("Orders")
        plt.xticks(rotation="vertical")
        plt.title(labels[column])

    plt.show()

    # Top 10 Aisles
    top_10_aisles = df_past_orders_info["aisle"].value_counts().head(10)

    plt.figure(figsize=(12,10))
    sns.barplot(top_10_aisles.index, top_10_aisles.values, alpha=0.8, color='darkblue')
    plt.ylabel("Orders")
    plt.xticks(rotation="vertical")
    plt.title('Top 10 Aisles')
    plt.show()

    # Most common departments
    top_10_departments = df_past_orders_info["department"].value_counts().head(10)

    plt.figure(figsize=(12,10))
    sns.barplot(top_10_departments.index, top_10_departments.values, alpha=0.8, color='darkblue')
    plt.ylabel("Orders")
    plt.xticks(rotation="vertical")
    plt.title('Top 10 Departments')
    plt.show()

    plt.figure(figsize=(10,10))
    dept_distn = df_past_orders_info['department'].value_counts()
    labels = (np.array(dept_distn.index))
    sizes = (np.array((dept_distn / dept_distn.sum())*100))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=200)
    plt.title("Departments distribution")
    plt.show()

    # Products per department
    products_department  = pd.merge(left=pd.merge(left=df_products, right=df_departments, how='left'), \
                                right=df_aisles, how='left')
    products_grouped = products_department.groupby("department")["product_id"]\
          .aggregate({'Total_products': 'count'}).reset_index()

    products_grouped['Ratio'] = products_grouped["Total_products"].apply(lambda x: \
                                                                         x /products_grouped['Total_products'].sum())
    products_grouped.sort_values(by='Total_products', ascending=False, inplace=True)

    plt.figure(figsize=(12,10))
    sns.barplot(products_grouped.department, products_grouped.Total_products, alpha=0.8, color='darkblue')
    plt.ylabel('Products')
    plt.xlabel('Department')
    plt.xticks(rotation='vertical')
    plt.title("Products per department")
    plt.show()

    def orders_ratio(row):
        return row.Orders_dept/common_departments[common_departments.department==row.department].Orders_dept.sum()

    # Most common departments per cluster
    common_departments = df_past_orders_cluster.loc[df_past_orders_cluster['department'].isin(products_grouped.department.head(6).values)]
    common_departments = common_departments.groupby(["department",'cluster'])["order_id"].aggregate({'Orders_dept': 'count'}).reset_index()
    common_departments['orders_ratio'] = common_departments.apply(orders_ratio, axis=1)

    number_figures = 6
    labels = ['Cluster 0','Cluster 1','Cluster 2']
    titles = products_grouped.department.head(number_figures)

    figures = []
    for figure in range(number_figures):
        fracs = []
        for cluster in range(3):
            fracs.append(common_departments[common_departments.department==titles.tolist()[figure]].orders_ratio.tolist()[cluster])
        figures.append(fracs)

    # Make square figures and axes
    rows, columns = 2, 3
    grid = GridSpec(rows, columns)
    plt.figure(figsize=(12,12))

    for row in range(rows):
        for column in range(columns):
            plt.subplot(grid[row, column], aspect=1)
            plt.pie(figures[column+3*row], labels=labels, autopct='%1.1f%%', shadow=True)
            plt.title(titles.tolist()[column+3*row], fontsize=16)

    plt.show()

    plt.figure(figsize=(10,8))
    sns.barplot(x="department", y="orders_ratio", hue="cluster", data=common_departments)
    plt.show();

    # Reorder ratios
    print(reorders_calc(df_order_products__prior, 'train'))
    print(reorders_calc(df_order_products__train, 'test'))

    reorders = []
    for cluster in range(len(cluster_number)):
        reorders.append(reorders_cluster(df_past_orders_cluster[df_past_orders_cluster.cluster==cluster], cluster))

    plt.figure(figsize=(10,8))
    sns.barplot([0,1,2], reorders, alpha=0.8, color='darkblue')
    plt.ylabel("Percentage of reorders")
    plt.xticks(rotation="vertical")
    plt.title('Cluster reorder ratio')

    for cluster in range(len(cluster_number)):
        plt.annotate(str(round(reorders[cluster],2))+' %', xytext=(-0.1+cluster, reorders[cluster]-0.04), \
                     xy=(0,0), size=14, weight='bold', color='lightblue')

    plt.show()

    df_reordered = df_past_orders_info.groupby(["department"])["reordered"].aggregate("mean").reset_index()

    plt.figure(figsize=(12,10))
    sns.pointplot(df_reordered['department'].values, df_reordered['reordered'].values, alpha=0.8, color='darkblue')
    plt.ylabel('Reorder ratio')
    plt.xlabel('Department')
    plt.title("Reorder ratio per department")
    plt.xticks(rotation='vertical')
    plt.show()

    # Most reordered products
    reorder_prob = df_past_orders_info.groupby("product_id")["reordered"]\
               .aggregate({'reordered_times': sum,'total_orders': 'count'}).reset_index()

    reorder_prob['reorder_prob'] = reorder_prob['reordered_times'] / reorder_prob['total_orders']
    reorder_prob = pd.merge(reorder_prob, df_products[['product_id', 'product_name']], how='left', on=['product_id'])
    reorder_prob = reorder_prob.sort_values(['reorder_prob'], ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12,10))
    sns.barplot(reorder_prob.product_name, reorder_prob.reorder_prob, alpha=0.8, color='darkblue')
    plt.xticks(rotation='vertical')
    plt.ylabel('Reorder probability')
    plt.title('Most reordered products')
    ax.set_ylim(0.8, 1)
    plt.show()

def reorders_calc(df, name):
    reorder = df.reordered.sum() / df.shape[0]
    print ("Percentage of reorders in {} set: {}".format(name,reorder))

def reorders_cluster(df, cluster):
    return df.reordered.sum() / df.shape[0]
