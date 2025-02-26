import pandas as pd
import statsmodels.api as sm

#%% Read the data from the file
source = pd.read_csv('./data/10_Industry_Portfolios.csv')

value_weighted = source.iloc[:1183,:]
equal_weighted = source.iloc[1186:2369,:]

value_weighted.iloc[0,0] = 'Date'
equal_weighted.iloc[0,0] = 'Date'

value_weighted.columns = value_weighted.iloc[0,:]
value_weighted = value_weighted.drop(value_weighted.index[0])
equal_weighted.columns = equal_weighted.iloc[0,:]
equal_weighted = equal_weighted.drop(equal_weighted.index[0])

value_weighted['Date'] = value_weighted['Date'].astype(int)
value_weighted = value_weighted[(value_weighted['Date'] >= 201001) & (value_weighted['Date'] <= 202012)]
equal_weighted['Date'] = equal_weighted['Date'].astype(int)
equal_weighted = equal_weighted[(equal_weighted['Date'] >= 201001) & (equal_weighted['Date'] <= 202012)]


ff3 = pd.read_csv('./data/ff3.csv')
monthly_ff3 = ff3.iloc[:1182, :]
monthly_ff3['Unnamed: 0'] = monthly_ff3['Unnamed: 0'].astype(int)
monthly_ff3 = monthly_ff3[(monthly_ff3['Unnamed: 0'] >= 201001) & (monthly_ff3['Unnamed: 0'] <= 202012)]

#%% value_weighted
industries = value_weighted.columns[1:-1]
vw_t_values = pd.DataFrame(columns=industries, index=['CAPM', 'FF3'])
vw_abs_alpha = pd.DataFrame(columns=industries, index=['CAPM', 'FF3'])

# CAPM
for industry in industries:
    y = pd.DataFrame(value_weighted[industry]).reset_index()[industry]
    x = monthly_ff3['Mkt-RF']
    x = sm.add_constant(x)
    x = x.reset_index()[['const', 'Mkt-RF']]
    model = sm.OLS(y.astype(float), x.astype(float)).fit()
    # print(model.summary())
    vw_t_values.loc['CAPM', industry] = model.tvalues['const']
    vw_abs_alpha.loc['CAPM', industry] = abs(model.params['const'])

# FF3
for industry in industries:
    y = pd.DataFrame(value_weighted[industry]).reset_index()[industry]
    x = monthly_ff3[['Mkt-RF', 'SMB', 'HML']]
    x = sm.add_constant(x)
    x = x.reset_index()[['const', 'Mkt-RF', 'SMB', 'HML']]
    model = sm.OLS(y.astype(float), x.astype(float)).fit()
    # print(model.summary())
    vw_t_values.loc['FF3', industry] = model.tvalues['const']
    vw_abs_alpha.loc['FF3', industry] = abs(model.params['const'])

