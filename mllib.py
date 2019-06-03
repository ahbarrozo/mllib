import os
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(data_path, data_name):
    """Loads data frame from csv file.
    rtype: pd.DataFrame """
    csv_path = os.path.join(data_path, data_name)
    return pd.read_csv(csv_path)


def model_saver(my_model, model_name):
    joblib.dump(my_model, model_name + '.pkl')


def model_loader(model_name):
    return joblib.load(model_name)


class MLDataset:
    def __init__(self, data_frame):
        if type(data_frame) is not pd.DataFrame:
            sys.exit('Could not open the dataset as a pandas DataFrame. Aborting.')
        else:
            self._dataset = data_frame
            self._train = data_frame
            self._test = None
            self._labelstrain = None
            self._labelstest = None
            self._numattribs = None
            self._catattribs = None
            self._attribs = None

            print(f"Data frame dimensions: {data_frame.shape}\n")
            print('\nFeatures available: \n')
            [print(i, end='\t\n') for i in data_frame.columns.values]

    @property
    def df(self):
        return self._dataset

    @df.setter
    def df(self, data_frame):
        self._dataset = data_frame

    def add_feat(self, feat_name, data):
        """Adds a new column to data frame."""
        if self._dataset.shape[0] == len(data):
            self._dataset[feat_name] = data
        else:
            print('Length of new data is not compatible with data frame.')

    def check_empty_data(self):
        """Checks for empty or NaN data in the data frame, plotting those missing
        numbers."""
        print(self._dataset.isnull().sum())
        sns.heatmap(self._dataset.isnull(), cmap='viridis', cbar=False, yticklabels=False)

        plt.title('Missing data')
        plt.show()

    def check_cat_feat(self, cat_feat, label):
        """Plots three different types of graphs for a given categorical feature
        to help visualize the data in terms of counts and with respect to the label"""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self._dataset, x=cat_feat, palette='viridis')
        plt.plot()
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self._dataset, x=cat_feat, y=label, palette='viridis')
        plt.plot()
        plt.figure(figsize=(10, 6))
        sns.stripplot(data=self._dataset, x=cat_feat, y=label, palette='viridis', jitter=0.3)
        plt.plot()

    def check_feat(self, feature):
        """Plots a box graph to visualize variance in the feature, and prints the
        basic statistics associated with the feature."""
        plt.figure(figsize=(10, 5))
        sns.boxplot(y=feature, data=self._dataset)
        plt.title(f'{feature}')
        plt.show()
        print(self._dataset[feature].describe())

    def fill_gaps(self, feature, strategy=None):
        """Fill gaps in data with a strategy (mean, median, constant values, etc)"""

        if strategy is 'median':
            print('Filling gaps with median')
            self._dataset[feature] = self._dataset[feature].fillna(self._dataset[feature].median())
        elif strategy is 'mean':
            print('Filling gaps with mean')
            self._dataset[feature] = self._dataset[feature].fillna(self._dataset[feature].mean())
        else:
            print('Please specify a valid strategy.')

    def get_feat_import(self, grid_search):
        """Extracts feature importance from a GridSearchCV best estimator.
        :return Table of feature importance"""
        if type(grid_search) is not GridSearchCV:
            print("ERROR: grid_search is not a valid GridSearchCV object")
            return 1
        feature_importances = grid_search.best_estimator_.feature_importances_
        # extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
        attributes = self._numattribs  # + extra_attribs

        for cat_attrib in self._catattribs:
            cat_encoder = OneHotEncoder(categories='auto')
            cat_encoder.fit_transform(self._test[cat_attrib].values.reshape(-1, 1))
            cat_one_hot_attribs = cat_encoder.categories_[0].tolist()
            attributes += cat_one_hot_attribs
        return sorted(zip(feature_importances, attributes), reverse=True)

        # plt.figure(figsize=(12, 8))
        # data = pd.DataFrame({'feature': self._attribs,
        #                     "importance": feature_importances})
        # sns.barplot(data=data, y='feature', x='importance')
        # plt.title('feature importance')

    def get_labels(self, label_name):
        """Extracts the attribute to be used as label from the dataset, storing into
        a new variable: _labels."""

        if label_name not in self._dataset.columns:
            print(f'ERROR: {label_name} is not a valid attribute')
        else:
            if self._train is not None:
                self._labelstrain = self._train[label_name].copy()
                self._train.drop(label_name, axis=1, inplace=True)
                if self._test is not None:
                    self._labelstest = self._test[label_name].copy()
                    self._test.drop(label_name, axis=1, inplace=True)

    def plot_corr_mat(self):
        plt.figure(figsize=(11, 7))
        sns.heatmap(cbar=False, annot=True, data=self._dataset.corr(), cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def plot_hist(self, bins=50, fig_size=(20, 15)):
        """Plots histograms for all the attributes in the data set.
        rtype: None"""

        self._dataset.hist(bins=bins, figsize=fig_size)
        plt.show()

    def plot_pca(self, data):
        total = 0
        clist = []

        for i in np.arange(0, data.shape[1]):
            p = PCA(n_components=i + 1)
            p.fit(data)
            total = total + p.explained_variance_ratio_[i]
            clist.append(total)
        data_var = list(map(lambda data: data * 100, clist))
        plt.figure(figsize=(15, 10))
        plt.plot(np.arange(1, data.shape[1] + 1), data_var, marker='o', markerfacecolor='red', lw=6)
        plt.xlabel('number of components')
        plt.ylabel('cumulative variance %')
        plt.title('cumulative variance ratio of PCA components')

        mpl.rcParams.update({'font.size': 25})
        pca = PCA(n_components=None)
        pca.fit(data)
        plt.figure(figsize=(40, 25))
        sns.heatmap(pca.components_, annot=True,
                    xticklabels=self._attribs,
                    yticklabels=[str(i) for i in range(1, data.shape[1] + 1)])
        plt.xlabel('Features')
        plt.ylabel('PCA components')
        plt.title('Relation matrix for each feature')
        plt.show()
        mpl.rcParams.update({'font.size': 12})

    def plot_strat(self, attrib):
        """Plots histogram for the strata used to prevent bias.

        Example: housing.plot_strat('median_income')"""
        strat_name = attrib + '_strat'
        if strat_name not in self._dataset.columns:
            print(f'ERROR: {strat_name} is not a valid attribute.')
            return False
        else:
            bins_strat = sorted(list(self._dataset[strat_name].unique()))
            bins_strat_gap = bins_strat[-1] - bins_strat[-2]
            bins_strat.append(bins_strat[-1] + bins_strat_gap)
            self._dataset[strat_name].hist(bins=bins_strat,
                                           zorder=1,
                                           rwidth=bins_strat_gap * 0.5,
                                           align='left')
            plt.show()
            return True

    def set_attrib_types(self, cat_attribs=None):

        if cat_attribs is not None:
            if type(cat_attribs) is str:
                self._catattribs = [cat_attribs]
            elif type(cat_attribs) is list:
                self._catattribs = cat_attribs
            self._numattribs = list(
                [attrib for attrib in self._train.columns.tolist() if attrib not in self._catattribs])
            self._attribs = self._numattribs.copy()
            for attrib in self._catattribs:
                self._attribs.extend(list(self._dataset[attrib].unique()))
        else:
            self._numattribs = self._train.columns.tolist()
            self._attribs = self._numattribs.copy()

    def set_pipeline(self, filler="median"):
        """Creates a pipeline to transform the data to apply ML algorithms, handling numerical
        and categorical attributes separately. This allows for an imputer to fill missing values.
        In our case, we are using median as our strategy.
        rtype: FeatureUnion (one pipeline for categorical, and one for numerical attributes)"""

        if self._catattribs is not None:
            num_pipeline = Pipeline(
                [('selector', DataFrameSelector(self._numattribs)),  # Selects only numerical attributes
                 ('imputer', SimpleImputer(strategy=filler)),  # Fills NaN values
                 # ('attribs_adder', CombinedAttributesAdder()),      # Adds combined attributes
                 ('std_scaler', StandardScaler())])  # Normalizes the data

            cat_pipeline = Pipeline(
                [('selector', DataFrameSelector(self._catattribs)),  # Selects only categorical attributes
                 ('cat_encoder', OneHotEncoder(sparse=False, categories='auto'))])

            return FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                                  ("cat_pipeline", cat_pipeline)])
        else:
            pipeline = Pipeline([('selector', DataFrameSelector(self._numattribs)),  # Selects only numerical attributes
                                 ('imputer', SimpleImputer(strategy=filler)),  # Fills NaN values
                                 # ('attribs_adder', CombinedAttributesAdder()),      # Adds combined attributes
                                 ('std_scaler', StandardScaler())])

            return pipeline

    def strat_attrib(self, attrib, num_strat=5.0):
        """Creates a new attribute for stratification, making an inplace modification.
        rtype: None"""
        strat_name = attrib + '_strat'
        ratio = np.max(self._dataset[attrib]) / (0.5 * num_strat)
        self._dataset[strat_name] = np.ceil(self._dataset[attrib] / ratio)
        self._dataset[strat_name].where(self._dataset[strat_name] < num_strat, num_strat, inplace=True)

    def strat_train_test_split(self, attrib, num_strat=5.0, test_size=0.2, drop=True, seed=42):
        """Splits the data set into train and test sets using a StratifiedShuffleSplit. By
        default, it separates 20% of the data for test set. Method 'strat_attrib()' will be
        used to create the stratification. The 'drop' option can be used to drop the
        stratification attribute in the end.

        Example: housing.strat_train_test_split("median_income", 5)
        """
        strat_name = attrib + '_strat'
        if strat_name not in self._dataset.columns:
            self.strat_attrib(attrib=attrib, num_strat=num_strat)

        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

        for train_index, test_index in split.split(self._dataset, self._dataset[strat_name]):
            self._train = self._dataset.loc[train_index]
            self._test = self._dataset.loc[test_index]

        if drop is True:
            for set_ in (self._dataset, self._test, self._train):
                set_.drop(strat_name, axis=1, inplace=True)


"""The next two classes were created to be added as steps in the Pipeline creation. 
CombinedAttributesAdder() generates new attributes. DataFrameSelector() selects 
specific attributes to pass through the Pipeline. We can use it to distinguish 
between numerical and categorical attributes."""

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Selector created to separate the types of attributes to be used to build a
    pipeline."""

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values