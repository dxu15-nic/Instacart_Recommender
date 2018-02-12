from data.make_clusters import *
from visualization.visualize import *
from models.recommender import *
import pandas as pd

print("Loading datasets...")
df_aisles = pd.read_csv("../data/raw/aisles.csv")
df_orders = pd.read_csv("../data/raw/orders.csv")
df_products = pd.read_csv("../data/raw/products.csv")
df_departments = pd.read_csv("../data/raw/departments.csv")
df_order_products__prior = pd.read_csv("../data/raw/order_products__prior.csv")
df_order_products__train = pd.read_csv("../data/raw/order_products__train.csv")

df_orders = createClusters(df_aisles, df_orders, df_products, df_order_products__prior)

visualize(df_aisles, df_departments, df_orders, df_products, df_order_products__train, df_order_products__prior)

product_recommender(df_order_products__prior, df_order_products__train, df_orders, df_products)
