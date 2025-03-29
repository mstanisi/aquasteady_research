import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import streamlit as st
from IPython.display import display


os.chdir('/Users/markos98/aquasteady_research')
path = "/Users/markos98/aquasteady_research/data/"

df23 = pd.read_csv(path + 'irrigated_df23.csv')
corr_matrix = df23.drop(['GEOGRAPHIC AREA'], axis=1).corr()
fig, ax = plt.subplots(figsize=(10, 7))
sb.heatmap(corr_matrix, annot=True, ax=ax)
plt.show()

st.title("Correlation matrix on my features")
st.write("This heatmap shows how the barriers are all multi-correlated.")

fig, ax = plt.subplots(figsize=(10, 7))
sb.heatmap(corr_matrix, annot=True, ax=ax)
st.pyplot(fig)

st.title("Ridge Regression")
st.write("Determining the most important features from an overdetermined dataset.")

dfc = pd.read_csv(path + 'irrigation data - compiled.csv')
cols = dfc.columns[1:51]
for col in cols:
    dfc[col] = dfc[col].str.replace(',', '').astype(float)
le = LabelEncoder()
y = le.fit_transform(dfc['AREA'])
X = dfc.drop(['AREA'], axis=1)
features = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

st.write("With machine learning, gives me a peek into feature importance.")

ridgeReg = Ridge(alpha=.01)
ridgeReg.fit(X_train,y_train)
train_score_ridge = ridgeReg.score(X_train, y_train)
test_score_ridge = ridgeReg.score(X_test, y_test)
print(train_score_ridge)
print(test_score_ridge)

coefficients = ridgeReg.coef_
importance_dfc = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

st.dataframe(importance_dfc)

st.title("Feature Importance")
st.write("The top features are related to finances and crop conditions.")

importance_dfc['AbsCoefficient'] = importance_dfc['Coefficient'].abs()
importance_dfc = importance_dfc.sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=importance_dfc, 
    x='Coefficient', 
    y='Feature', 
    palette='viridis'
)
plt.title('Feature Coefficient Rankings')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.axvline(0, color='red', linestyle='--', linewidth=1)  # To show the neutral line
plt.tight_layout()
st.pyplot(plt)

st.write("Time to see how these trends track over the last 20 years.")

dfy = pd.read_csv(path + 'irrigation data - by year.csv')
dfy[['2023']] = dfy[['2023']].div(29930162)
dfy[['2018']] = dfy[['2018']].div(29511478)
dfy[['2013']] = dfy[['2013']].div(29305842)
dfy[['2008']] = dfy[['2008']].div(24119889)
dfy[['2003']] = dfy[['2003']].div(26165456)
print(dfy)

st.title("Trends over the years")
st.write("Finances and climate-related barriers remain highest.")

years = ['2003', '2008', '2013', '2018', '2023']
for index, row in dfy.iterrows():
    plt.plot(years, row[1:], label=row['barrier'], marker='o')
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Barrier Trends Over the Years')
plt.legend(title='Barrier', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

st.title("States with greatest barriers to irrigation")
st.write("The Lower Mississippi ecoregion experiences the greatest stress.")

dfc_t_st = pd.read_csv(path + 'dfc_t_st.csv')
cols = dfc_t_st.columns[1:51]
for col in cols:
    dfc_t_st[col] = dfc_t_st[col].astype(float)

le = LabelEncoder()
y = le.fit_transform(dfc_t_st['barrier'])
columns_to_drop = ['barrier']
X = dfc_t_st.drop(columns=columns_to_drop, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = RandomForestClassifier(random_state=1, max_depth=12)
model.fit(X_train, y_train)
display(model.feature_importances_)
fl = pd.get_dummies(X)
features = fl.columns
importances = model.feature_importances_
indices = np.argsort(importances)[:]
plt.figure(figsize=(10, 20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
st.pyplot(plt)

st.title("Regions ranked by importance")

dfc_t_rg = pd.read_csv(path + 'dfc_t_rg.csv')
cols = dfc_t_rg.columns[1:51]
for col in cols:
    dfc_t_rg[col] = dfc_t_rg[col].astype(float)

le = LabelEncoder()
y = le.fit_transform(dfc_t_rg['barrier'])
columns_to_drop = ['barrier']
X = dfc_t_rg.drop(columns=columns_to_drop, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = RandomForestClassifier(random_state=1, max_depth=12)
model.fit(X_train, y_train)
display(model.feature_importances_)
fl = pd.get_dummies(X)
features = fl.columns
importances = model.feature_importances_
indices = np.argsort(importances)[:]
plt.figure(figsize=(10, 20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
st.pyplot(plt)

st.title("Linear Regression for Lower MS")
st.write("Main barriers to conservation are increasing for the region.")

lwr_mspi_yr = pd.read_csv(path + 'irrigation data - lwr mspi by year.csv')
cols = lwr_mspi_yr.columns[1:51]
for col in cols:
    lwr_mspi_yr[col] = lwr_mspi_yr[col].str.replace(',', '').astype(float)
print(lwr_mspi_yr)

lwr_mspi_lr = lwr_mspi_yr.melt(id_vars='barrier', var_name='year', value_name='stress')
lwr_mspi_lr['year'] = lwr_mspi_lr['year'].astype(int)

X = lwr_mspi_lr['year'].values.reshape(-1, 1)
y = lwr_mspi_lr['stress'].values

X = lwr_mspi_lr['year'].values.reshape(-1, 1)
y = lwr_mspi_lr['stress'].values
reg = LinearRegression()
reg.fit(X, y)
lwr_mspi_lr['predicted'] = reg.predict(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=lwr_mspi_lr,
    x='year',
    y='stress',
    hue='barrier', 
    palette='tab10', 
    s=50  
)
plt.plot(lwr_mspi_lr['year'], lwr_mspi_lr['predicted'], color='red', label='Regression Line', linewidth=2)
st.pyplot(plt)

st.title("Extrapolation")
st.write("These trends extend into the next 5 years.")

extrapolation = pd.read_csv(path + 'irrigation data - lwr mspi extrapolation.csv')
cols = extrapolation.columns[1:51]
for col in cols:
    extrapolation[col] = extrapolation[col].str.replace(',', '').astype(float)

extrapolation = extrapolation.melt(id_vars='barrier', var_name='year', value_name='stress')
extrapolation['year'] = extrapolation['year'].astype(int)
X = extrapolation['year'].values.reshape(-1, 1)
y = extrapolation['stress'].values
X = extrapolation['year'].values.reshape(-1, 1)
y = extrapolation['stress'].values
reg = LinearRegression()
reg.fit(X, y)
extrapolation['predicted'] = reg.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(extrapolation['year'], extrapolation['stress'], color='blue', label='Actual')
plt.plot(extrapolation['year'], extrapolation['predicted'], color='red', label='Regression Line')
plt.title('extrapolation')
plt.xlabel('year')
plt.ylabel('stress')
plt.legend()
st.pyplot(plt)

st.title("Cluster Analysis")
st.write("Most regions and states cluster together, but CA and MO regions warrant investigation.")

try:
    cluster = pd.read_csv(path + 'irrigation data - 2023.csv')
    
    # Verify the data loaded correctly
    if cluster.empty:
        st.error("Cluster data failed to load or is empty!")
        st.stop()
    
    # Process columns - only numeric columns
    numeric_cols = cluster.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col != 'GEOGRAPHIC AREA':
            # Convert to string, remove commas, then to float
            cluster[col] = pd.to_numeric(cluster[col].astype(str).str.replace(',', ''), errors='coerce')
    
    X = cluster.select_dtypes(include=['number'])
    if X.empty:
        st.error("No numeric columns found for clustering!")
        st.stop()
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    st.write(f'Silhouette Score: {silhouette_avg:.2f}')

    regions = cluster['GEOGRAPHIC AREA']  
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
    plt.colorbar(scatter, label="Cluster")

    for i, region in enumerate(regions):
        plt.text(X_pca[i, 0], X_pca[i, 1], region, fontsize=8, alpha=0.7)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Cluster Visualization with Region Labels')
    st.pyplot(plt)

    # CA Year analysis
    ca_year = pd.read_csv(path + 'irrigation data - CA by year.csv')
    if ca_year.empty:
        st.error("CA year data failed to load or is empty!")
    else:
        numeric_cols = ca_year.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != 'barrier':
                ca_year[col] = pd.to_numeric(ca_year[col].astype(str).str.replace(',', ''), errors='coerce')
        
        ca_year = ca_year.melt(id_vars='barrier', var_name='year', value_name='stress')
        ca_year['year'] = pd.to_numeric(ca_year['year'], errors='coerce')
        st.write(ca_year)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

st.title("Linear Regression for CA region")
st.write("California experiencing an inverse relationship between climate-related barriers and financial ones.")

X = ca_year['year'].values.reshape(-1, 1)
y = ca_year['stress'].values
reg = LinearRegression()
reg.fit(X, y)
ca_year['predicted'] = reg.predict(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=ca_year,
    x='year',
    y='stress',
    hue='barrier',  # Color points by the 'barrier' column
    palette='tab10',  # Choose a color palette (e.g., 'tab10', 'viridis', etc.)
    s=50  # Adjust point size
)
plt.plot(ca_year['year'], ca_year['predicted'], color='red', label='Regression Line', linewidth=2)
st.pyplot(plt)