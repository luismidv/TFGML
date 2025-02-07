import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import euclidean

def data_preparing():
    dataframe = pd.read_csv('./data/users_dataframe.csv')
    print(dataframe.describe())
    
    categorical_columns = [ 'Worktimes', 'Schedules', 'Studies level', 'Pets', 'Cooking', 'Sport', 'Smoking', 'Organized']
    numerical_columns = ['Age']
    dataframe = dataframe[categorical_columns + numerical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), categorical_columns),
            ('num', StandardScaler(), numerical_columns)

        ]
    )

    X_transform = preprocessor.fit_transform(dataframe)




    return X_transform

def data_checking(dataframe):
    for col in dataframe.columns:
        if dataframe[col].isnull().sum() > 0:
            print(f"Missing values in {col} column")
        else:
            print(f"No missing values in column {col}")

def reshape_playground(data):
    print(f"Data shape {data.shape}")
    data[50].reshape(17,1)
    print(data.shape)

def forward_algorithm(dataframe,cluster_spec):

    cluster_spec = dataframe[0]
    kmeans = KMeans(n_clusters=4, random_state=42)
    result = kmeans.fit_predict(dataframe)
    print(result[1])
    print(result)

def set_specific_cluster(dataframe,cluster_spec):
    cluster_spec = dataframe[cluster_spec]
    distances = [euclidean(cluster_spec,point) for point in dataframe]
    print(distances)

def specific_cluster_kmeans(dataframe, cluster_spec):
    cluster_spec = dataframe[cluster_spec]
    cluster_spec = cluster_spec.reshape(1,17)
    print(cluster_spec.shape)
    kmeans = KMeans(n_clusters=1, init = cluster_spec, n_init = 1, random_state=42)
    data = kmeans.fit(dataframe)
    print(data)





dataframe= data_preparing()
print(dataframe)
#forward_algorithm(dataframe,50)
#set_specific_cluster(dataframe,50)
specific_cluster_kmeans(dataframe,50)
