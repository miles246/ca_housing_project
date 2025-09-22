#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump


class FeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 5]
        bedrooms_per_room = X[:, 4] / X[:, 3]
        population_per_household = X[:, 5] / X[:, 6]
        return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]


def main():
    input_path = "CMSE492/ca_housing_project/data/train/housing_train_processed.csv"
    output_path = "../data/train/train_final.csv"

    df = pd.read_csv(input_path)

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    numeric_features = [
        "longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population",
        "households", "median_income"
    ]

    categorical_features = ["ocean_proximity"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", FeatureAdder()),
        ("scaler", StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])

    X_processed = full_pipeline.fit_transform(X)
    final_df = pd.DataFrame(
        np.c_[X_processed, y.to_numpy()],
        columns=list(full_pipeline.get_feature_names_out()) + ["median_house_value"]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    dump(full_pipeline, "../data/preprocessing_pipeline.joblib")  # It allows to save and load scikit-learn models, pipelines, or any Python object efficiently.


if __name__ == "__main__":
    main()