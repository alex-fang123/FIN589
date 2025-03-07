import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm

# grs test for pca_3 and FF3
def grs_test(y, x, factors, alpha):
    T, N = y.shape
    K = factors.shape[1]
    F = factors
    F = sm.add_constant(F)
    F = pd.DataFrame(F)
    F = F.reset_index()[[0, 1, 2, 3]]
    e = y - x @ alpha
    e = e.reset_index()[0]
    model = sm.OLS(e.astype(float), F.astype(float)).fit()
    return T / N * (model.rsquared_adj / (1 - model.rsquared_adj)) * ((T - N - K) / (T - K - 1))


source = pd.read_csv("./data/25_Portfolios_5x5.CSV", nrows=1182, skiprows=1)
source['Unnamed: 0'] = source['Unnamed: 0'].astype(int)
target = source[(source['Unnamed: 0'] >= 201001)& (source['Unnamed: 0'] <= 202412)]

# run PCA and get the latent factors

pca = PCA(n_components=3)
X = target.iloc[:, 1:]
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca.singular_values_)
print(pca.mean_)

factors = pca.transform(X)
print(factors)


# read ./data/ff3.csv
ff3 = pd.read_csv('./data/ff3.csv', nrows=1182)
ff3['Unnamed: 0'] = ff3['Unnamed: 0'].astype(int)
ff3 = ff3[(ff3['Unnamed: 0'] >= 201001) & (ff3['Unnamed: 0'] <= 202412)]
ff3 = ff3[['Unnamed: 0' ,'Mkt-RF', 'SMB', 'HML']]

# merge ff3 and factors into one
result = ff3.copy()
result['f1'] = factors[:, 0]
result['f2'] = factors[:, 1]
result['f3'] = factors[:, 2]

industries = target.columns[1:]
vw_t_values = pd.DataFrame(columns=industries, index=['pca_3', 'FF3'])
vw_abs_alpha = pd.DataFrame(columns=industries, index=['pca_3', 'FF3'])
vw_adjusted_r2 = pd.DataFrame(columns=industries, index=['pca_3', 'FF3'])
vw_grs = pd.DataFrame(columns=industries, index=['pca_3', 'FF3'])

# ff3
for industry in industries:
    y = pd.DataFrame(target[industry]).reset_index()[industry]
    x = result[['Mkt-RF', 'SMB', 'HML']]
    x = x.reset_index()[['Mkt-RF', 'SMB', 'HML']]
    x = sm.add_constant(x)
    model = sm.OLS(y.astype(float), x.astype(float)).fit()
    # print(model.summary())
    vw_t_values.loc['FF3', industry] = model.tvalues['const']
    vw_abs_alpha.loc['FF3', industry] = abs(model.params['const'])
    vw_adjusted_r2.loc['FF3', industry] = model.rsquared_adj
    vw_grs.loc['FF3', industry] = grs_test(pd.DataFrame(target[industries]), result[['Mkt-RF', 'SMB', 'HML']], factors, model.params)

# pca_3
for industry in industries:
    y = pd.DataFrame(target[industry]).reset_index()[industry]
    x = result[['f1', 'f2', 'f3']]
    x = x.reset_index()[['f1', 'f2', 'f3']]
    x = sm.add_constant(x)
    model = sm.OLS(y.astype(float), x.astype(float)).fit()
    # print(model.summary())
    vw_t_values.loc['pca_3', industry] = model.tvalues['const']
    vw_abs_alpha.loc['pca_3', industry] = abs(model.params['const'])
    vw_adjusted_r2.loc['pca_3', industry] = model.rsquared_adj
    vw_grs.loc['pca_3', industry] = grs_test(pd.DataFrame(target[industries]), result[['f1', 'f2', 'f3']], factors, model.params)

vw_t_values.to_excel('./output/vw_t_values.xlsx')
vw_abs_alpha.to_excel('./output/vw_abs_alpha.xlsx')
vw_adjusted_r2.to_excel('./output/vw_adjusted_r2.xlsx')
