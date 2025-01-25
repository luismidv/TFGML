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



    return dataframe,categorical_columns,numerical_columns, preprocessor

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

def forward_algorithm(dataframe,cluster_spec, preprocessor):

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
    ])

    scaled_data = pipeline.fit(dataframe)
    cluster_spec = scaled_data[0]
    print(cluster_spec.shape)

    distances = [euclidean(cluster_spec, dataframe) for person in scaled_data]


    #dataframe['cluster'] = pipeline.named_steps['clustering'].labels_

    #pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
    #X = pipeline.fit_transform(dataframe)





dataframe, categorical_columns, numerical_columns, preprocessor= data_preparing()
print(dataframe)
forward_algorithm(dataframe,50, preprocessor)
