from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

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
print(housing_prepared)

# train the model -> linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)
#print(f"The RMSE value of the Linear Regression model is {lin_rmse}")
print("This is for the linear regression model")
print(pd.Series(lin_rmses).describe())

# decision tree model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels,scoring = "neg_root_mean_squared_error", cv = 10)
#print(f"The RMSE value fo the Decision Tree Regression model is {dec_rmse}")
print("This is for the decision tree regression model")
print(pd.Series(dec_rmses).describe())

# random forest classifier model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
#random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_preds)
random_forest_rmses = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)
#print(f"The RMSE value of the Random Forest Regression model is {random_forest_rmse}")
print("This is for the random forest tree regression model")
print(pd.Series(random_forest_rmses).describe())

print("Thus I choose to select the RandomForest Regression model as the RMSE value after cross validation is the least amongst all")