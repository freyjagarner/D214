# import statements
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import itertools

from IPython.display import Markdown as md
from IPython.core.magic import register_cell_magic
# from IPython.display import HTML
# import random

# setting the path to the csv files
path = os.path.join(os.path.abspath(os.getcwd()), 'csvs\\')
dataframe = pd.DataFrame

@register_cell_magic
def markdown(line, cell):
    return md(cell.format(**globals()))   

# function to style dataframes as first column being bold and hiding the index
# to bring attention to the important data
def boldify(df):
    df = df.style.set_table_styles([{'selector': 'td.col0', 'props': 'font-weight:bold'}]).hide_index()
    return df

def null_vals(df):
    # if the dataframe of nulls is not empty return the null values as a dataframe
    if not df[df.isnull()].empty:
        df_nulls = dataframe(df.isnull().sum(), columns=['Null Values'])
        df_nulls = df_nulls.reset_index()
        df_nulls.rename(columns={'index':'Variable'})
        return boldify(df_nulls)
    else:
        return md("No Null Values")

# function to make a dataframe of a dataframes columns datatypes
def get_dtypes(df):
    # make a dataframe out of datatypes
    df_dtypes = dataframe(df.dtypes, columns=['Type'])
    # index is automatically the column variables
    # make index into a range index so Variable can be a column
    df_dtypes = df_dtypes.reset_index()
    # set index column name to variable
    df_dtypes.rename(columns={'index':'Variable'})
    # return data types dataframe
    return boldify(df_dtypes)

# function to get the row and column counts of a dataframe
def get_size(df):
    # get the count of dataframe columns
    col_size = len(df.columns)
    # get the count of the dataframe rows
    row_size = len(df)
    # create dataframe of columns and rows with their count
    df_sizes = dataframe(columns=['Count'], index=['Columns', 'Rows'], data=[col_size, row_size])
    # return the dataframe
    return df_sizes

#function to get duplicate rows in a dataframe
def get_dupes(df):
    # if the dataframe of duplicated values is not empty return the dataframe of duplicate values
    if not df[df.duplicated()].empty:
        df_dupes = df[df.duplicated()]
        return df_dupes
    # else return No Duplicate Values
    else:
        return md("There are no Duplicate Values")


# function to makea  dataframe of the counts of unique values in columns
def count_unique(df):
    # create dataframe of unique value counts per column
    unique_counts = dataframe(df.nunique(), columns=['Count'])
    # make a range index to make the variable names a column
    unique_counts = unique_counts.reset_index()
    # return dataframe of unique column counts
    return boldify(unique_counts)

# function to print the unique values of each categorical column in a dataframe
def unique_vals(df):
    only_categories = df.select_dtypes(include=['string', 'object', 'category']).columns
    # iterate through the dataframe columns
    for i in only_categories:
        # make a list of lists of unique values in relevant columns less than 25 items in length
        unique_val_list = [[list(df[i].explode().unique()) for i in only_categories if len(df[i].explode().unique()) < 10000]]
        # make a list of the column names for the dataframe index
        idx = [i for i in only_categories if len(df[i].explode().unique()) < 10000]
        # create a df using a dict mapping the values of the unique value list to a column vs a row
        unique_vals_df = dataframe(dict(zip(["Values"], unique_val_list)), index=idx)
        # add a range index to put the variables into a row instead of the index
        unique_vals_df = unique_vals_df.reset_index()
        # rename index column to variable
        unique_vals_df.rename(columns={'index':'Variable'})
        # return the dataframe 
        return boldify(unique_vals_df)

def classify_scores(df, score_dict, score_column, column_name):
    df = df.assign(column_name=0)
    df[column_name] = [k for k, v in score_dict.items() if df[score_column] in v]
        

# a function to change dataframe column values based on a given dictionary
def change_col_val(val_dict, df):
    # val_dict is a dictionary of lists
    for k, v in val_dict.items():
        for i in v:
            # change the value of the cell to its index number in a list
            df.loc[df[k] == i, k] = v.index(i)


# a function to get a percentage
def percentage(part, whole):
    return round(100 * float(part) / float(whole), 2)

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, chi2
def get_kbest(x_train, y_train):
    selector = SelectKBest(f_classif, k='all')
    X_train_new = selector.fit_transform(x_train, y_train.values.ravel()) 
    mask = selector.get_support()    
    new_features = x_train.columns[mask]
    return(pd.DataFrame(sorted(zip(selector.pvalues_, new_features)), columns=['P-Value', 'Variable']))


from sklearn.model_selection import cross_val_score
# function to make dataframe of cross validations
def get_cross_val(clf, X, y):
    # fit data/classifier to 10 cross validations
    cross_val = cross_val_score(clf, X, y, cv=10)
    # sort scores in descending order
    sorted_cross_vals = sorted([i for i in cross_val], reverse=True)
    # make df and name column to match contents
    cv_df = dataframe(sorted_cross_vals).rename(columns={0:'cross_val_score'})
    return cv_df