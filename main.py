import pandas as pd
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

url = 'https://drive.google.com/uc?id=1BZjTKO_zYZf7kdtWw3JAq3PsV8Gr7IFA&export=download'

saida = 'climate_change.csv'

gdown.download(url, saida, False)
df = pd.read_csv(saida)

# Usando para exibir as primeiras linhas e ter certeza que a base foi chamada da maneira certa e mostrar colunas
df.head()

print("Antes da exclusão:", df.shape)
print("\nLinhas que contém dados ausentes: ", df.isnull().any(axis=1).sum())
print("Colunas que contém dados ausentes:\n", df.isnull().sum())
print("Linhas contendo dados duplicados: ", df.duplicated().sum())

df = df.dropna()

print("Depois da exclusão:", df.shape)
print("\nValores ausentes após a exclusão:\n", df.isnull().sum())

# Gráficos e valores da AED
# """ 
# numericas = df.select_dtypes(include=["number"])

# print("Médias das colunas numéricas:")
# print(numericas.mean())

# print("\nMedianas das colunas numéricas:")
# print(numericas.median())

# df[['Avg_Temperature(°C)', 'CO2_Emissions(Mt)',
#     'Sea_Level_Rise(mm)', 'Climate_Risk_Index']].hist(
#     bins=20, figsize=(12,6), color='skyblue', edgecolor='black'
# )

# plt.suptitle("Distribuição das Variáveis Climáticas", fontsize=16)
# plt.show()

# plt.figure(figsize=(10,5))

# line_data = df.groupby("Year")["Avg_Temperature(°C)"].mean()

# plt.plot(line_data.index, line_data.values, marker='o', linewidth=2)

# plt.title("Temperatura Média Global por Ano", fontsize=14)
# plt.xlabel("Ano")
# plt.ylabel("Temperatura Média (°C)")
# plt.grid(True)

# plt.show()

# plt.figure(figsize=(10, 6))
# corr = df[['Avg_Temperature(°C)', 'CO2_Emissions(Mt)', 'Sea_Level_Rise(mm)', 'Climate_Risk_Index']].corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title("Mapa de Correlação das Variáveis Climáticas", fontsize=16)
# plt.show()

# plt.figure(figsize=(12, 6))
# sns.boxplot(
#     data=df,
#     x="Continent",
#     y="Climate_Risk_Index",
#     hue="Continent",
#     legend=False,
#     palette="Set2"
# )
# plt.title("Distribuição do Risco Climático por Continente", fontsize=16)
# plt.xlabel("Continente")
# plt.ylabel("Climate Risk Index")
# plt.show()
# """

# Modelos: Regressão Logistica, Random Forest, Gradient Boosting, XGBoost, Support Vector Machine (SVM).

# Discretização da variável
df = pd.read_csv("climate_change.csv")

#Definindo o climate risk como 0 e 1 - baixo e alto risco.
df["Risk_Class"] = df["Climate_Risk_Index"].apply(lambda x: 0 if x <= 50 else 1)

# """
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# LOGISTIC REGRESSION
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# """

X_lg = df[["Continent", "Avg_Temperature(°C)", "CO2_Emissions(Mt)", "Sea_Level_Rise(mm)"]]
y_lg = df["Risk_Class"]

categorical_features = ["Continent"]
numeric_features = ["Avg_Temperature(°C)", "CO2_Emissions(Mt)", "Sea_Level_Rise(mm)"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)

X_train_lg, X_test_lg, y_train_lg, y_test_lg = train_test_split(
    X_lg, y_lg, test_size=0.25, stratify=y_lg, random_state=42
)

lg_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=500))
])

lg_model.fit(X_train_lg, y_train_lg)
y_pred = lg_model.predict(X_test_lg)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test_lg, y_pred))
print("Precision:", precision_score(y_test_lg, y_pred))
print("Recall:", recall_score(y_test_lg, y_pred))
print("F1 Score:", f1_score(y_test_lg, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_lg, y_pred))

# """
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# RANDOM FOREST
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# """

X_rd = df[["Continent", "Avg_Temperature(°C)", "CO2_Emissions(Mt)", "Sea_Level_Rise(mm)"]]
y_rd = df["Risk_Class"]

categorical_features = ["Continent"]
numeric_features = ["Avg_Temperature(°C)", "CO2_Emissions(Mt)", "Sea_Level_Rise(mm)"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)

X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(
    X_rd, y_rd, test_size=0.25, stratify=y_rd, random_state=42
)

rd_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

rd_model.fit(X_train_rd, y_train_rd)
y_pred_rd = rd_model.predict(X_test_rd)

# Resultados do Random Forest
print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test_rd, y_pred_rd))
print("Precision:", precision_score(y_test_rd, y_pred_rd))
print("Recall:", recall_score(y_test_rd, y_pred_rd))
print("F1 Score:", f1_score(y_test_rd, y_pred_rd))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_rd, y_pred_rd))

# """
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# GRADIENT BOOSTING
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# """

X_gb = df[["Continent", "Avg_Temperature(°C)", "CO2_Emissions(Mt)", "Sea_Level_Rise(mm)"]]
y_gb = df["Risk_Class"]

categorical_features = ["Continent"]
numeric_features = ["Avg_Temperature(°C)", "CO2_Emissions(Mt)", "Sea_Level_Rise(mm)"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)

X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(
    X_gb, y_gb, test_size=0.25, stratify=y_gb, random_state=42
)

gb_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", GradientBoostingClassifier(random_state=42))
])

gb_model.fit(X_train_gb, y_train_gb)
y_pred_gb = gb_model.predict(X_test_gb)

# Resultados desse modelo Gradient Boosting
print("=== Gradient Boosting ===")
print("Accuracy:", accuracy_score(y_test_gb, y_pred_gb))
print("Precision:", precision_score(y_test_gb, y_pred_gb))
print("Recall:", recall_score(y_test_gb, y_pred_gb))
print("F1 Score:", f1_score(y_test_gb, y_pred_gb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_gb, y_pred_gb))

# """
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# XGBOOST
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# """

xg_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ))
])

# Utilizando mesmos valores de treinamento/teste do gradiente boosting
xg_model.fit(X_train_gb, y_train_gb)
y_pred_xg = xg_model.predict(X_test_gb)

# Resultados do modelos de XG Boost
print("=== XG Boost ===")
print("Accuracy:", accuracy_score(y_test_gb, y_pred_xg))
print("Precision:", precision_score(y_test_gb, y_pred_xg))
print("Recall:", recall_score(y_test_gb, y_pred_xg))
print("F1 Score:", f1_score(y_test_gb, y_pred_xg))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_gb, y_pred_xg))

# """
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# SVM - SUPPORT VECTOR MACHINE
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# """

svm_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", SVC(
        kernel="rbf",
        C=1.5,
        gamma="scale",
        probability=True
    ))
])

# Utilizando mesmos valores de treinamento/teste do gradiente boosting
svm_model.fit(X_train_gb, y_train_gb)
y_pred_svm = svm_model.predict(X_test_gb)

# Resultado do valor de support vector machine - svm
print("=== Support Vector Machine ===")
print("Accuracy:", accuracy_score(y_test_gb, y_pred_svm))
print("Precision:", precision_score(y_test_gb, y_pred_svm))
print("Recall:", recall_score(y_test_gb, y_pred_svm))
print("F1 Score:", f1_score(y_test_gb, y_pred_svm))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_gb, y_pred_svm))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# DEPLOY WITH STREAMLIT
# ----------------------------------------------------------------------------------------------------------------------------------------------------------


import streamlit as st

# ----------------------------------------------------------------------
# MODELOS TREINADOS (assume que já existem: lg_model, rd_model, clf, xg_model, svm_model)
# ----------------------------------------------------------------------

st.title("Previsão de Risco Climático 🌍")

st.write("Selecione o modelo e insira os valores abaixo para estimar o nível de risco climático.")


# ----------------------------------------------------------------------
# Seleção do modelo
# ----------------------------------------------------------------------
model_choice = st.selectbox(
    "Escolha o modelo de Machine Learning:",
    [
        "Regressão Logística",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "Support Vector Machine (SVM)"
    ]
)

model_map = {
    "Regressão Logística": lg_model,
    "Random Forest": rd_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xg_model,
    "Support Vector Machine (SVM)": svm_model
}

model = model_map[model_choice]


# ----------------------------------------------------------------------
# Entradas do usuário
# ----------------------------------------------------------------------
continents = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]

continent = st.selectbox("Continente:", continents)

temperature = st.number_input(
    "Temperatura média (°C):",
    min_value=-50.0,
    max_value=60.0,
    step=0.1
)

co2 = st.number_input(
    "Emissões de CO2 (Mt):",
    min_value=0.0,
    max_value=50000.0,
    step=0.1
)

sea_level = st.number_input(
    "Nível do mar (mm):",
    min_value=-200.0,
    max_value=2000.0,
    step=0.1
)


# ----------------------------------------------------------------------
# Rodar a previsão
# ----------------------------------------------------------------------
if st.button("Executar Previsão"):
    input_df = pd.DataFrame([{
        "Continent": continent,
        "Avg_Temperature(°C)": temperature,
        "CO2_Emissions(Mt)": co2,
        "Sea_Level_Rise(mm)": sea_level
    }])

    pred = model.predict(input_df)[0]

    risco = "ALTO RISCO 🌡️🔥" if pred == 1 else "BAIXO RISCO 🌿"

    st.subheader("Resultado da Previsão")
    st.write(f"O modelo estimou: **{risco}**.")
