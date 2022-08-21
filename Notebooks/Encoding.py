import numpy as np
from sklearn.impute import SimpleImputer

def feature_engineering(dataframe, useless_columns):
    """Remove unnecessary columns from the dataframe

    Args:
        dataframe (DataFrame): input dataframe
        useless_columns (list): list of useless column names
                    that should be removed from the dataframe

    Returns:
        dataframe: returns cleaned up dataframe
    """    
    for col in useless_columns:
        if col in dataframe:
            dataframe = dataframe.drop(col,axis=1)
    return dataframe

def categorical_encoding(dataframe, column_name):
    """Converts categorical values into numbers and returns
    the dataframe as well as the code with which the values were
    encoded

    Args:
        dataframe (DataFrame): input dataframe with categorical values
        column_name (str): colum name with categorical values

    Returns:
        dataframe: redurns encoded dataframe
        dict: code with the dataframe columns were encoded
    """
    code = {}
    
    for idx, category_name in enumerate(dataframe[column_name].unique()):
        code[category_name] = idx

    dataframe[column_name] = dataframe[column_name].map(code)
    
    return dataframe, code
    
def imputation(dataframe, strategy='mean'):
    """Fills missing values 

    Args:
        dataframe (dataframe): input dataframe with missing values
        strategy (str): string specifying the imputation strategy

    Returns:
        dataframe: return dataframe with no missing values
    """
    for col in dataframe.columns:
        imputer = SimpleImputer(strategy=strategy, missing_values=np.nan)
        imputer = imputer.fit(dataframe[[col]])
        dataframe[col] = imputer.transform(dataframe[[col]])
    return dataframe

def preprocessing(dataframe, useless_columns):
    """Returns dataframe with feature_engineering, categorical_encoding
    and imputation done with single call

    Args:
        dataframe (DataFrame): input dataframe with categorical values
        column_name (list): list of colums with categorical values

    Returns:
        dataframe(DataFrame): processed dataframe
        X(DataFrame): dataframe containting data features
        y(DataFrame): dataframe containting target value
        mapping_code(dict): dictionary of encoding code for future 
                    references
    """
    dataframe = feature_engineering(dataframe, useless_columns)

    mapping_code = {}
    for col in dataframe.select_dtypes('object'):
        dataframe, mapping_code[col] = categorical_encoding(dataframe, col)

    dataframe = imputation(dataframe)
    
    X = dataframe.drop('Class',axis=1)
    y = dataframe['Class'].astype(int)
      
    return dataframe,X,y, mapping_code