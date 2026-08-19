import pandas as pd
def load_data(file_path):
    """
    Load csv file

    """
    df=pd.read_csv(file_path)
    print("Data Loaded Successfully")
    print("Shape:",df.shape)
    return df

def prepare_feature(
        df,
        target_column="Churn",
        id_column="CustomerID"
):
    """
    Prepare input Feature X and target Y

    """
    # remvoe the id and target col 
    X=df.drop(
        [id_column,target_column],axis=1
    )
    # convert target in 1 or 0
    y=df[target_column].apply(lambda x:1 if x=="Yes" else 0)
    return X,y

def preprocess_data(
        file_path,
        target_column="Churn",
        id_column="CustomerID"
):
    """
    Complete preprocessing function
    """
    df= load_data(file_path)
    # prepare the x and y 
    X,y=prepare_feature(df,target_column,id_column)
    return X,y
