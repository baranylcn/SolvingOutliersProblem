#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()


######################################
# OUTLIERS
######################################

sns.boxplot(x=df["Age"])
plt.show()
# As seen there are outliers in this dataset


# How to find outlier thresholds?
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr
print("Up threshold:", up)
print("Low threshold:", low)


# Let's see what are outliers :
df[(df["Age"] < low) | (df["Age"] > up)]
df[(df["Age"] < low) | (df["Age"] > up)].index


# Are there any outliers in dataset?
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)
df[(df["Age"] < low)].any(axis=None)


###################
# Functionalization
###################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
# A function created to find thresholds automatically.

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")
low, up = outlier_thresholds(df, "Fare")
df[(df["Fare"] < low) | (df["Fare"] > up)].head()
df[(df["Fare"] < low) | (df["Fare"] > up)].index
# Outliers found.


# Let's make it more functional :
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

#Examples
check_outlier(df, "Age")
check_outlier(df, "Fare")


###################
# Categorizing Columns
###################

dff = load_application_train()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the dataset.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                Class threshold for numeric but categorical variables
        car_th: int, optinal
                Class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                List of Categorical Variables
        num_cols: list
                List of Numerical Variables
        cat_but_car: list
                Cardinal variable liste with categorical view

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is in cat_cols.
        The sum of 3 lists with returns equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


num_cols = [col for col in num_cols if col not in "PassengerId"]
# Also we can see variables that are not num_cols and drop them.


for col in num_cols:
    print(col, check_outlier(df, col))
# Applied for all num_cols

cat_cols, num_cols, cat_but_car = grab_col_names(dff)
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))



###################
# Accessing Outliers
###################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

#Examples
grab_outliers(df, "Age")
grab_outliers(df, "Age", True)
age_index = grab_outliers(df, "Age", True)
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)



#############################################
# Solving the Outliers Problem
#############################################


###################
# Deletion
###################

low, up = outlier_thresholds(df, "Fare")
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
# Dataframe without outliers found.


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]


for col in num_cols:
    new_df = remove_outlier(df, col)

# Have many values deleted?
df.shape[0] - new_df.shape[0]


###################
# Data suppression (re-assignment with thresholds)
###################

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]
df.loc[(df["Fare"] > up), "Fare"] = up
df.loc[(df["Fare"] < low), "Fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
# Dataset reloaded
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]


for col in num_cols:
    print(col, check_outlier(df, col))
# There are outliers in dataset.
for col in num_cols:
    replace_with_thresholds(df, col)
# Suppression
for col in num_cols:
    print(col, check_outlier(df, col))
# No outliers anymore

###################
# Recap
###################

df = load()
outlier_thresholds(df, "Age")
# Outlier thresholds detected

check_outlier(df, "Age")
# Checked to see if there are outliers.

grab_outliers(df, "Age", index=True)
# Show which are outliers in dataframe.

remove_outlier(df, "Age").shape
# Remove the outliers

replace_with_thresholds(df, "Age")
# Suppression

check_outlier(df, "Age")




#############################################
# Multivariate Outlier Analysis: Local Outlier Factor (LOF)
#############################################
# When we looked at the variables separately, we detected outliers. So, if we look at the variables together, can we get outlier variables? For example, if a person was married 3 times at the age of 18.
# Being 18 years old or getting married 3 times are not problems, but being 18 years old and married 3 times can be an outlier.


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
# Only numeric variables are selected and missing values filled.


for col in df.columns:
    print(col, check_outlier(df, col))
# There are outliers in dataset.


low, up = outlier_thresholds(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape
low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape
# There are too many outliers in this dataset. If we do things in this way, it may cause data loss.


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
# Applied to the "LocalOutlierFactor" dataset.


df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# These values assigned to be traceable.

np.sort(df_scores)[0:5]
# If the values close to -1, it indicates that it is INLIER.


# Elbow Method :
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
# A graph was created according to the threshold values,
# and when we examined the graph, the point where the slope change was the hardest was determined.
# The determined slope change point was chosen as threshold.


# The value in the 3rd index is selected.
th = np.sort(df_scores)[3]

df[df_scores < th]
# Outliers
df[df_scores < th].shape


# The individual variables may appear as outliers,
# but we found outliers depending on the situation between the variables.

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)
# These values could be deleted.

# If working with tree methods, these values should not be changed.
