from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

# load the database
housing = pd.read_csv("housing.csv")
print(housing)

# create a stratified test set
housing["income_cat"] = pd.cut(housing["median_income"], bins = [0, 1.5, 3.0, 4.5, 6.0, np.inf], labels = [1, 2, 3, 4, 5])
print(housing)

sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in sss.split(housing, housing["income_cat"]) :
    strat_train_set = housing.loc[train_index].drop("income_cat", axis = 1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis = 1)
print(strat_test_set)
print(strat_test_set)

# seperate features and labels
housing = strat_train_set.copy()
housing_labels = housing["median_house_value"].copy()
print(housing_labels)
housing = housing.drop("median_house_value", axis = 1)
print(housing)

# seperate numerical and categorical columns
num_attributes = housing.drop("ocean_proximity", axis = 1).columns.tolist()
print(num_attributes)
cat_attributes = ["ocean_proximity"]
print(cat_attributes)

# making pipeline for numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scalar", StandardScaler())
])

# mkaing pipeline for categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown = "ignore"))
])

# construct the whole pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
])

# transform the data
housing_prepared = full_pipeline.fit_transform(housing)
housing = pd.DataFrame(housing_prepared, columns = housing.columns, index = housing.index)
print(housing_prepared)