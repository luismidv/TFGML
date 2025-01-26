import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import resultview as rw
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

def data_preparing():
    dataframe = pd.read_csv('./MLSystem/data/users_dataframe.csv')
    print(dataframe.describe())
    
    categorical_columns = [ 'Worktimes', 'Schedules', 'Studies level', 'Pets', 'Cooking', 'Sport', 'Smoking', 'Organized']
    numerical_columns = ['Age']
    dataframe = dataframe[categorical_columns + numerical_columns]
    
    return dataframe,categorical_columns,numerical_columns

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
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('cat', OneHotEncoder(sparse_output=False), categorical_columns),
    #         ('num', StandardScaler(), numerical_columns)
            
    #     ]
    # )
    encoder = OneHotEncoder(sparse_output=False)
    encoded_df = encoder.fit_transform(dataframe)
    #pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
    print(f"Kmeans receive data. \n Actual length {dataframe.shape}")
    #X = pipeline.fit_transform(dataframe)
    print(encoded_df)
    print(f"Kmeans specific cluster: {encoded_df[cluster_spec]} | {encoded_df[cluster_spec].shape}" )
    cluster_center = encoded_df[cluster_spec]
    k_means = KMeans(n_clusters=1, init = cluster_center , max_iter=300, random_state=42)
    k_means.fit(cluster_center)
    # results = k_means.predict(encoded_df)
    # centers =  k_means.cluster_centers_
    # print(f"Cluster results \n {results}")    
    # print(f"Cluster centers \n {centers}")

    #rw.view_kmeans_results(results, centers)

def forward_algorithm2(dataframe):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_df = encoder.fit_transform(dataframe)
    print(encoded_df)
    k_means = KMeans(n_clusters=2, n_init = 10, max_iter=300, random_state=42)
    k_means.fit(encoded_df)
    predict = k_means.predict(encoded_df)
    print(predict)

dataframe, categorical_columns, numerical_columns= data_preparing()
forward_algorithm(dataframe,50)