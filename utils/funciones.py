import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly_express as px
import seaborn as sns
from PIL import Image
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def filtrar_por_distancia(df, distancia_km, lat_ref=41.1496, lon_ref=-8.6109):
    """
    Filters a DataFrame based on the distance from a reference point.

    Args:
        df (pandas.DataFrame): The DataFrame to filter.
        distancia_km (float): The maximum distance in kilometers.
        lat_ref (float, optional): The latitude of the reference point. Defaults to 41.1496.
        lon_ref (float, optional): The longitude of the reference point. Defaults to -8.6109.

    Returns:
        pandas.DataFrame: The filtered DataFrame containing only rows within the maximum distance.
    """
    # convert latitude and longitude columns to numeric
    df.loc[:, "latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df.loc[:, "longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # drop any rows with missing values in latitude or longitude columns
    df = df.dropna(subset=["latitude", "longitude"])

    # calculate distance from reference point to each row using Haversine formula
    R = 6371  # Earth's radius in km
    dlat = np.radians(df["latitude"].astype(float) - lat_ref)
    dlon = np.radians(df["longitude"].astype(float) - lon_ref)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat_ref))
        * np.cos(np.radians(df["latitude"]))
        * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    df["distancia"] = R * c

    # filter DataFrame to include only rows within maximum distance
    df_filtrado = df.loc[df["distancia"] < distancia_km]

    return df_filtrado


def fritas(df):
    """
    Given a pandas DataFrame, encodes all categorical (object) columns using
    Label Encoding and returns a copy of the encoded DataFrame.
    Parameters:
    - df: pandas DataFrame
    Returns:
    - df_encoded: pandas DataFrame
    """
    df_encoded = df.copy()  # Make a copy of the original DataFrame
    # Select the categorical columns of the DataFrame
    object_columns = df_encoded.select_dtypes(include=["object"]).columns
    encoder_info = []  # Initialize a list to store the encoder information

    for column in object_columns:
        le = LabelEncoder()  # Create a new LabelEncoder for each categorical column
        # Fit and transform the LabelEncoder on the column
        df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
        encoder_info.append(
            {  # Store the encoder information in a dictionary
                "column": column,
                "labels": list(le.classes_),  # List the original labels
                "codes": list(le.transform(le.classes_)),  # List the encoded codes
            }
        )

    # Return the encoded DataFrame and the encoder information
    return df_encoded, encoder_info


def bravas(df, target_column, min_k=2, max_k=15):
    """
    Given a pandas DataFrame, a target column name, a range of k values and a
    minimum number of samples per fold, performs K-NN regression using cross-validation
    to find the best value of k (number of neighbors) based on the mean squared error.
    Parameters:
    - df: pandas DataFrame
    - target_column: str, name of the target column
    - min_k: int, minimum number of neighbors to consider
    - max_k: int, maximum number of neighbors to consider
    Returns:
    - best_k: int, best value of k found
    """
    # Instantiate a LabelEncoder object
    le = LabelEncoder()
    # Make a copy of the input DataFrame
    df_encoded = df.copy()
    # Select object columns (categorical) of df_encoded
    object_columns = df_encoded.select_dtypes(include=["object"]).columns
    # Iterate over each categorical column and apply Label Encoding
    for column in object_columns:
        df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
    # Impute missing values using the mean of each column
    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_encoded), columns=df_encoded.columns
    )
    # Separate the predictors (X) from the target (y)
    X = df_imputed.drop(target_column, axis=1)
    y = df_imputed[target_column]
    # Define a pipeline for K-NN regression
    pipeline = Pipeline(steps=[("model", KNeighborsRegressor(n_neighbors=3))])
    # Set the hyperparameters to tune
    params = {
        "model__n_neighbors": [3, 5, 7],
        "model__weights": ["uniform", "distance"],
    }
    best_k = 0
    best_score = -np.inf
    # Iterate over a range of k values and perform cross-validation
    for k in range(min_k, max_k + 1):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, params, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
        )
        grid_search.fit(X, y)
        # Keep track of the best k and best score found so far
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_k = k
    # Return the best value of k found
    return best_k


def impute_missing_values_with_knn(df, n_neighbors):
    df_imputed = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    numeric_columns = df_imputed.select_dtypes(include=["int64", "float64"]).columns
    df_imputed[numeric_columns] = imputer.fit_transform(df_imputed[numeric_columns])
    return df_imputed


def desfritas(df_encoded, encoder_info):
    """
    Given a pandas DataFrame that has been encoded with the `fritas()` function and the encoder
    information dictionary returned by that function, decodes all categorical columns and returns
    a copy of the original DataFrame with the encoded columns replaced by their original values.
    This function replaces any codes that are not in the original list of labels with -1
    Parameters:
    - df_encoded: pandas DataFrame
    - encoder_info: list of dicts
    Returns:
    - df_decoded: pandas DataFrame
    """
    df_decoded = df_encoded.copy()  # Make a copy of the encoded DataFrame

    # Loop over each encoder in the encoder_info dictionary
    for encoder in encoder_info:
        column = encoder["column"]
        labels = encoder["labels"]
        codes = encoder["codes"]
        le = LabelEncoder()  # Create a new LabelEncoder for the column
        le.classes_ = np.array(labels)  # Set the original labels
        # Replace NaN values in the encoded column with -1
        df_decoded[column].fillna(-1, inplace=True)
        # Replace any codes that are not in the original list of labels with -1
        df_decoded[column].where(df_decoded[column].isin(codes), -1, inplace=True)
        # Inverse transform the codes
        df_decoded[column] = le.inverse_transform(df_decoded[column].astype(int))

    return df_decoded


def pure(df, method="minmax"):
    """
    Apply scaling to a DataFrame using either MinMaxScaler or StandardScaler.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be scaled.
        method (str, optional): The scaling method to be used. Default is "minmax".

    Returns:
        tuple: A tuple containing the scaled DataFrame and the scaler object used.

    Raises:
        ValueError: If the method is not "minmax" or "standard".

    """
    scaler = None
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()

    if scaler:
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled, scaler
    else:
        raise ValueError("method must be either 'minmax' or 'standard'")


def cortar_en_tiritas(df, target_column, test_size=0.2, random_state=None):
    """
    Split the input DataFrame into training and testing sets.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - target_column (str): The name of the target column to be dropped.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
    - X_train (DataFrame): The training set without the target column.
    - X_test (DataFrame): The testing set without the target column.
    - y_train (Series): The target column for the training set.
    - y_test (Series): The target column for the testing set.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def evaluar_papata(model, X_test, y_test):
    """
    Evaluate the performance of a model by calculating mean squared error (mse),
    mean absolute error (mae), and r-squared (r2) scores.

    Parameters:
    - model: The trained model object.
    - X_test: The input features for testing.
    - y_test: The target variable for testing.

    Returns:
    A dictionary containing the mse, mae, and r2 scores.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mse": mse, "mae": mae, "r2": r2}


def papas_perdidas(df):
    """
    Calculate the number and percentage of missing values in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    pandas.DataFrame: A DataFrame containing the number and percentage of missing values for each column.
    """
    missing_values = df.isnull().sum()
    missing_values_percentage = 100 * missing_values / len(df)
    missing_values_table = pd.concat(
        [missing_values, missing_values_percentage], axis=1
    )
    missing_values_table.columns = ["Missing Values", "Percentage"]
    return missing_values_table


def correlation_papa(df, annot=True, figsize=(10, 10), cmap="coolwarm"):
    """
    Plot a correlation heatmap for the numeric columns of a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - annot (bool, optional): Whether to annotate the heatmap with correlation values. Defaults to True.
    - figsize (tuple, optional): The size of the figure. Defaults to (10, 10).
    - cmap (str, optional): The color map to use for the heatmap. Defaults to "coolwarm".

    Returns:
    None
    """
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[numeric_columns].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap)
    plt.show()


def response_rate_chart(df):
    listings10 = df[df["number_of_reviews"] >= 10]
    feq1 = (
        listings10["host_response_rate"]
        .replace(0, np.nan)
        .dropna()
        .sort_values(ascending=True)
    )

    fig = px.histogram(feq1, nbins=35)
    fig.update_layout(
        title="Tasa de respuesta (al menos 10 reseñas)",
        xaxis_title="Porcentaje",
        yaxis_title="Número de anuncios",
        font=dict(size=20),
        xaxis=dict(range=[0, 100]),
    )
    return fig


def response_time_chart(df):
    listings10 = df[df["number_of_reviews"] >= 10].dropna(subset=["host_response_time"])
    # Eliminar los valores 0 del dataframe
    listings10 = listings10[listings10["host_response_time"] != 0]
    df_response_time = (
        listings10["host_response_time"].value_counts().reset_index(name="count")
    )
    fig = px.bar(
        df_response_time,
        x="host_response_time",  # Corrected 'x' argument
        y="count",
        labels={"host_response_time": "Tiempo de respuesta", "count": "Número"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(
        title="Tiempo de respuesta (al menos 10 reseñas)",
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        font=dict(size=16),
        yaxis=dict(range=[0, None]),
    )
    return fig


def response_charts(df):
    fig1 = response_rate_chart(df)
    fig2 = response_time_chart(df)

    # Create subplot with 1 row and 2 columns, with titles for each subplot
    figs = sp.make_subplots(
        rows=1, cols=2, subplot_titles=("Tasa de respuesta", "Tiempo de respuesta")
    )

    # Add each bar chart to the subplot
    figs.add_trace(fig1["data"][0], row=1, col=1)
    figs.add_trace(fig2["data"][0], row=1, col=2)

    # Update layout for the subplot
    figs.update_layout(
        title_text="Respuestas",
        height=400,
        width=1000,
        font_size=12,
        showlegend=False,
        template="plotly_dark",
    )
    return figs


def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
