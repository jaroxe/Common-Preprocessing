import numpy as np

def strings_to_cats(df):
    """
    df: Pandas dataframe

    Converts all string variables in the dataframe ('object' type) into
    categorical variables ('category' type)
    """
    for n,c in df.items():
        if c.dtype == 'O':
            df[n] = c.astype('category').cat.as_ordered()

def build_cat_code_dict(df, cat_col_name):
    """
    df: pandas dataframe
    cat_col_name: String, name of categorical column

    Builds dictionary with categories as keys and associated codes + 1 as values
    This way NAs will be assigned to 0
    Creates an additional category 'other'
    Returns the dictionary d = {'cat1': <code1>+1, 'cat2':<code2>+1...}
    """
    d = {}
    cat_list = list(df[cat_col_name].drop_duplicates())
    code_list = list(df[cat_col_name].drop_duplicates().cat.codes)
    for i in range(len(cat_list)):
        d[cat_list[i]] = code_list[i] + 1
    d['other'] = i+2
    return d

def build_cat_dicts(df):
    """
    df: Pandas dataframe

    Creates a dictionary containing mappings for all categorical variables in the dataframe
    Returns dictionary of dictionaries {'col_name': {'cat1': <code1>+1, 'cat2':<code2>+1...}}
    """
    d = {}
    for n,c in df.items():
        if c.dtype.name == 'category':
            mapping = build_cat_code_dict(df, n)
            d[n] = mapping
    return d

def new_to_other(df, d):
    """
    df: Pandas dataframe, test set or previously unseen data
    d: Dictionary, mappings for categorical variables generated from the training set

    The function goes through all categorical variables in the test set
    Changes categories that didn't appear in training set to 'other'
    Returns the resulting dataframe
    """
    df_res = df.copy()
    
    for n,c in df_res.items():
        if c.dtype.name == 'category':
            cats_in_dict = set(d[n].keys())
            cats_in_df = set(df_res[n])

            for cat in (cats_in_df - cats_in_dict):
                df_res[n] = df_res[n].replace(cat, 'other').astype('category')
                
    return df_res

def cats_to_codes_plus1(df, d={}):
    """
    df: Pandas dataframe
    d: dictionary of dictionaries --> d = {'col_name': <mapping of cats to codes>}
    
    Same function can be applied to train and test sets
    If applied to training set, empty dictionary will be passed by default
        The function will add the mapping info (categories to codes) into the dictionary
    If applied to test set, pass in the mappings dictionary generated from the training set

    Returns:
        df_res: Original dataframe with numericalized categorical variables
        d: Mappings dictionary for categorical variables
                        {'col_name': {'cat1': <code1>+1, 'cat2':<code2>+1...}}
    """
    df_res = df.copy()
    if d == {}:
        d = build_cat_dicts(df)
    for n,c in df_res.items():
        if c.dtype.name == 'category':
            df_res[n] = c.map(d[n])
    return df_res, d

def nan_to_median(df, na_dict={}):
    """
    df: Pandas dataframe
    na_dict: Dictionary

    Searches for missing values in each column in the data frame
    If a given column has missing values:
        1. NAs will be filled with value in na_dict corresponding to that column
        2. If 1. doesn't apply:
            2.1: Fill NAs with column median
            2.2: Adds {'col_name': <median>} to na_dict 
    Returns dataframe with filled NAs, na_dict
    """
    df_res = df.copy()
    if len(na_dict) > 0:
        for key in na_dict:
            df_res[key] = df_res[key].fillna(na_dict[key])
    for n,c in df_res.items():
        if c.isnull().any():
            df_res[n] = c.fillna(c.median())
            na_dict[n] = c.median()
    return df_res, na_dict

def df_to_X_y(df, y_name):
    """
    df: Pandas dataframe
    y_name: String, name of response variable

    Separates the response variable from the dataframe
    Returns:
        X: Original dataframe without the response column
        y: Numpy array of the response variable
    """
    X = df.copy()
    y = np.array(df[y_name])
    X = X.drop([y_name], axis=1)
    return X, y

def process_df(df, mappings={}, na_dict={}, y_name=None, fill_na=True):
    """
    df: Pandas dataframe
    mappings: Dictionary, {'categorical_col': {'col1': <num1>, 'col2':<num2>...}...}
    na_dict: Dictionary, {'column_name': <value to fill NAs>} (usually median)
    y_name: String, name of response variable
    fill_na: bool, if True function will replace missing values in dataframe

    Function performs basic pre-processing on dataframe
    Performs same pre-processing on training data (labels provided) and unseen data
    If applied on train data, mappings and na_dict will be empty dictionaries
        these will be filled in by the function
    If applied on test data, mappings and na_dict from train should be passed as arguments

    Returns:
        processed dataframe
        numpy array of response (if y_name provided)
        mappings dictionary
        na_dict dictionary
    """
    df_res = df.copy()
    strings_to_cats(df_res) # Convert string columns into categorical
    if y_name is None:      # If df is test set, set unseen categories to 'other'
        df_res = new_to_other(df_res, mappings)
    df_res, mappings = cats_to_codes_plus1(df_res, mappings) # Numericalize categorical variables
    if fill_na:             # Fill NAs with column median and save info
        df_res, na_dict = nan_to_median(df_res, na_dict)
    if y_name is None:      # If df is test set, no response to separate
        return df_res, mappings, na_dict
    else:                   # If df is not test set, separate response variable
        X, y = df_to_X_y(df_res, y_name)
        return X, y, mappings, na_dict