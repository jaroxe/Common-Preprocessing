# Consistent-Preprocessing
Consistent preprocessing for training and test sets

Sometimes, after training a Machine Learning model, we would like to test this model on previously unseen data. Before feeding this new data into our model we have to make sure that we apply the same preprocessing steps as we did to the original training set (when we first created our model).

In particular, we have to make sure that our numericalization of categorical variables (the conversion of categories into numbers) is identical to that of the original training data.
We also have to make sure that we replace missing values consistently in both training and test sets.

`consistent_preprocessing.py` contains functions that will perform these steps adequately.

How it works (refer to step 4 for simple use):

### 1. Numericalize categorical variables

Let’s say we have a pandas dataframe `train_df` containing our training set. Some of the variables in this data aren’t yet numerical. This means that the data cannot be employed to train a model in its current state. The first step is to convert any string variables into categorical variables:

<!-- -->

    strings_to_cats(train_df)

Now that all our non-numerical variables are categorical, we can proceed to convert the categorical variables into numerical, by assigning a number to each category:

    train_df, mappings = cats_to_codes_plus1(train_df)

`mappings` is a dictionary that contains the relationship between categories and numbers (which number has been assigned to each category) for each categorical variable. Using this dictionary we can now numericalize the test set `test_df` (unseen data) with identical mappings:

    strings_to_cats(test_df)
    test_df = new_to_other(test_df, mappings)
    test_df, _ = cats_to_codes_plus1(test_df, mappings)

The second line of code in the above snippet makes sure that any categories that did not appear in the training set are grouped into a category named 'other', which our function will be able to handle.

### 2. Filling NAs

Once all the data in our dataframe is numerical, we can proceed to fill missing values:

    train_df, na_dict = nan_to_median(train_df)

The above line of code replaces all NAs in a given column by the median of that column. It also stores this median in the dictionary `na_dict`.
With this dictionary we can now fill the missing values in the test set (unseen data) consistently:

    test_df, _ = nan_to_median(test_df, na_dict)

The missing values in the test set will be replaced with the column median from the training set. In the rare case that a test column presents NAs without that column presenting NAs in the training set, the NAs will be replaced by the column median in the test set.

### 3. Separating the response variable from the main dataframe

Very likely, our response variable (what we would like to predict) will be contained as a column in our training dataframe. The following code will separate the response variable 'resp' from the dataframe (as a numpy array):

    X_train, y_train = df_to_X_y(train_df, 'resp')

### 4. Doing it the simple way

The `process_df` function will perform all the above steps automatically. However, in some cases, it will be preferable to perform each step separately, in order to have more control over the process:

    X_train, y_train, mappings, na_dict = process_df(train_df, y_name='resp')
    X_test, _, _ = process_df(test_df, mappings, na_dict)

**Note**: Some of the functions in this repository were inspired by the fastai library.
