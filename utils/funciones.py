import numpy as np
import pandas as pd


def filtrar_por_distancia(df, distancia_km, lat_ref=41.1496, lon_ref=-8.6109):
    # Calcular la distancia entre cada localización y las coordenadas de referencia
    R = 6371  # Radio medio de la Tierra en km
    dlat = np.radians(df["latitude"] - lat_ref)
    dlon = np.radians(df["longitude"] - lon_ref)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat_ref)) * \
        np.cos(np.radians(df["latitude"])) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distancia"] = R * c

    # Filtrar las localizaciones según la distancia
    df_filtrado = df.loc[df["distancia"] < distancia_km]

    return df_filtrado


# from sklearn.preprocessing import LabelEncoder
# from sklearn. impute import KNNImputer
# from sklearn. model_selection import GridSearchCV, KFold
# from sklearn. linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn. impute import SimpleImputer
# import pandas as pd
# import numpy as np


# def encode_categorical_columns(df):
#     df_encoded = df.copy()
#     object_columns = df_encoded.select_dtypes(include=["object"]).columns
#     for column in object_columns:
#         le = LabelEncoder()
#         df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
#     return df_encoded


# def best_k(df, target_column, min_k=2, max_k=10):
#     le = LabelEncoder()
#     df_encoded = df.copy()
#     object_columns = df_encoded.select_dtypes(include=['object']).columns
#     for column in object_columns:
#         df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
#     imputer = SimpleImputer(strategy='mean')
#     df_imputed = pd.DataFrame(imputer.fit_transform(
#         df_encoded), columns=df_encoded.columns)
#     X = df_imputed.drop(target_column, axis=1)
#     y = df_imputed[target_column]
#     pipeline = Pipeline(steps=[('model', KNeighborsRegressor(n_neighbors=3))])
#     params = {'model__n_neighbors': [3, 5, 7],
#               'model__weights': ['uniform', 'distance']}
#     best_k = 0
#     best_score = -np.inf
#     for k in range(min_k, max_k+1):
#         kf = KFold(n_splits=k, shuffle=True, random_state=42)
#         grid_search = GridSearchCV(
#             pipeline, params, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
#         grid_search.fit(X, y)
#         if grid_search.best_score_ > best_score:
#             best_score = grid_search.best_score_
#             best_k = k
#     return best_k


# def impute_missing_values_with_knn(df, n_neighbors):
#     df_imputed = df.copy()
#     imputer = KNNImputer(n_neighbors=n_neighbors)
#     numeric_columns = df_imputed.select_dtypes(
#         include=['int64', 'float64']).columns
#     df_imputed[numeric_columns] = imputer.fit_transform(
#         df_imputed[numeric_columns])
#     return df_imputed
