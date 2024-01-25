from datetime import datetime
import pickle
from random import randint

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from sqlalchemy import create_engine

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

PK = 'Customer_ID'
Y_COL = 'churn'
TABLE_NAME = 'churn_prediction_dataset'

MODELS_PATH = f"/opt/airflow/data/models"

def _read_config() -> dict:
    """
    Read config from Postgres
    """
    pass

def _get_data() -> pd.DataFrame:
    """
    Read dataset from Postgres
    """
    select_cols:list = _read_config()['features']
    select_cols.append(Y_COL)
    
    select_cols:str = ', '.join(select_cols)
    
    with PostgresHook(postgres_conn_id=ID_CONNECTION) as hook:
        df = hook.get_pandas_df(f"SELECT {select_cols} FROM {TABLE_NAME}")
        
    return df

def _remove_outliers(X:pd.DataFrame) -> tuple:
    """
    """
    features = _read_config()['preprocess']['outliers']
    df = X.copy()
    
    indices = [x for x in df.index]    
    out_indexlist = []
        
    for col in features:       
        Q1 = np.nanpercentile(df[col], 25.)
        Q3 = np.nanpercentile(df[col], 75.)
        
        cut_off = (Q3 - Q1) * 1.5
        upper, lower = Q3 + cut_off, Q1 - cut_off
                
        outliers_index = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
        out_indexlist.extend(outliers_index)
        
    #using set to remove duplicates
    out_indexlist = list(set(out_indexlist))
    
    clean_data = np.setdiff1d(indices,out_indexlist)

    return X.loc[clean_data]

def _fill_missing_values(X:pd.DataFrame) -> pd.DataFrame:
    """
    """
    config = _read_config()['preprocess']['missing_values']
    df = X.copy()
    
    for col in df.columns:
        if col in config['mean']:
            df[col].fillna(df[col].mean(), inplace=True)
        elif col in config['literal']:
            df[col].fillna(config['literal'][col], inplace=True)
        
    return df

def _build_preprocessor(df:pd.DataFrame) -> Pipeline:
    """
    Build preprocessor pipeline
    """
    cathegorical_features = list(df.select_dtypes(include='object').columns)
    numerical_features = list(df.select_dtypes(exclude='object').columns)
    
    print(f'Categorical features({len(cathegorical_features)}): {cathegorical_features}')
    print(f'Numerical features({len(numerical_features)}): {numerical_features}')
    
    # missing values
    missing_values_transformer = Pipeline(steps=[
        ('fillNaN', FunctionTransformer(_fill_missing_values))
    ])
    
    # outliers
    outliers_transformer = Pipeline(steps=[
        ('removeOutliers', FunctionTransformer(_remove_outliers))
    ])
    
    # transformers
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # column transformer
    preprocessor = ColumnTransformer(transformers=[
        missing_values_transformer,
        outliers_transformer,
        ('cat', cat_transformer, cathegorical_features),
        ('num', num_transformer, numerical_features)
    ])
    
    return preprocessor

def preprocess_data() -> None:
    """
    Preprocess dataset and save to Postgres
    
    :return: None
    """
    df = _get_data()
    preprocessor = _build_preprocessor(df)
    
    X = df.drop(Y_COL, axis=1)
    y = df[Y_COL]
    
    X = preprocessor.fit_transform(X)
    
    df = pd.DataFrame(X, columns=df.drop(Y_COL, axis=1).columns)
    df[Y_COL] = y
    
    pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
    engine = create_engine(pg_hook.get_uri())
    
    df.to_sql(f"{TABLE_NAME}_gold", engine, index=False, if_exists='replace')

def train_model() -> dict:
    """
    Train model
    """
    seed = randint(0, 100)
    
    df = _get_data()
    
    # train test split
    X = df.drop([Y_COL, PK], axis=1)
    y = df[Y_COL]
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # build model using pipeline
    model = Pipeline(steps=[
        ('classifier', RandomForestClassifier())
    ])
    
    model.fit(X_train, y_train)
    
    # save model
    model_name = f"model_{seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    model_path = f"{MODELS_PATH}/{model_name}"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return {
        "seed":seed, 
        "model_name":model_name
    }

def evaluate_model(seed:int, model_name:str) -> None:
    """
    Evaluate model
    
    :param split_seed: seed used to split dataset
    :type split_seed: int
    
    :return: None
    """
    df = _get_data()
    
    # train test split
    X = df.drop([Y_COL, PK], axis=1)
    y = df[Y_COL]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # load model
    model_path = f"{MODELS_PATH}/{model_name}"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # evaluate model
    y_pred = model.predict(X_test)
    
    # save results
    results = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })
    
    pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
    engine = create_engine(pg_hook.get_uri())
    
    results.to_sql(f"{TABLE_NAME}_results", engine, index=False, if_exists='replace')

with DAG(
    dag_id=DAG_ID,
    start_date=datetime.now()
    ):
    
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )
    
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
    
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        op_kwargs={'seed': train_model.output['seed'], 'model_name': train_model.output['model_name']}
    )
    
    preprocess_data >> train_model >> evaluate_model