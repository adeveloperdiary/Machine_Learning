from load_data import fetch_housing_data, load_gousing_data
from visualization import showHistogram
import numpy as np

fetch_housing_data()
housing = load_gousing_data()
# housing.head()
# housing.info()

print(housing["ocean_proximity"].value_counts())
# housing.describe()

showHistogram(housing, bins=50)

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

strat_train_set = []
strat_test_set = []

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing["income_cat"].value_counts() / len(housing)

housing_data = strat_train_set.copy()

corr_matrix=housing_data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
