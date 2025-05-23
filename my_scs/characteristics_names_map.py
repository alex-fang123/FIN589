def characteristics_names_map():
    # anomalies
    anom_names = ['size', 'value', 'prof', 'valprof', 'fscore', 'debtiss', 'repurch', 'nissa', 'accruals', 'growth',
                  'aturnover', 'gmargins',
                  'divp', 'ep', 'cfp', 'noa', 'inv', 'invcap', 'igrowth', 'sgrowth', 'lev', 'roaa', 'roea', 'sp',
                  'gltnoa',
                  'mom', 'indmom', 'valmom', 'valmomprof', 'shortint', 'mom12', 'momrev', 'lrrev', 'valuem', 'nissm',
                  'sue', 'roe', 'rome', 'roa', 'strev', 'ivol', 'betaarb',
                  'season', 'indrrev', 'indrrevlv', 'indmomrev', 'ciss', 'price', 'age', 'shvol']

    anom_descriptions = ['Size', 'Value (A)', 'Gross profitability', 'Value-profitablity', 'F-score', 'Debt issuance',
                         'Share repurchases',
                         'Net issuance (A)', 'Accruals', 'Asset growth', 'Asset turnover', 'Gross margins',
                         'Dividend/Price', 'Earnings/Price', 'Cash Flows/Price', 'Net operating assets',
                         'Investment/Assets', 'Investment/Capital', 'Investment growth', 'Sales growth',
                         'Leverage', 'Return on assets (A)', 'Return on book equity (A)', 'Sales/Price',
                         'Growth in LTNOA',
                         'Momentum (6m)', 'Industry momentum', 'Value-momentum', 'Value-momentum-prof.',
                         'Short interest', 'Momentum (12m)',
                         'Momentum-reversals', 'Long-run reversals', 'Value (M)', 'Net issuance (M)',
                         'Earnings surprises',
                         'Return on book equity (Q)', 'Return on market equity', 'Return on assets (Q)',
                         'Short-term reversals',
                         'Idiosyncratic volatility', 'Beta arbitrage', 'Seasonality', 'Industry rel. reversals',
                         'Industry rel. rev. (L.V.)', 'Ind. mom-reversals', 'Composite issuance', 'Price', 'Age',
                         'Share volume']

    mapAnomalies = dict(zip(anom_names, anom_descriptions))

    # WRDS financial ratios
    finratios_names = ['capital_ratio', 'equity_invcap', 'debt_invcap', 'totdebt_invcap', 'at_turn', 'inv_turn',
                       'pay_turn', 'rect_turn', 'sale_equity', 'sale_invcap', 'sale_nwc', 'invt_act', 'rect_act',
                       'fcf_ocf', 'ocf_lct', 'cash_debt', 'cash_lt', 'cfm', 'short_debt', 'profit_lct', 'curr_debt',
                       'debt_ebitda', 'dltt_be', 'int_debt', 'int_totdebt', 'lt_debt', 'lt_ppent', 'cash_conversion',
                       'cash_ratio', 'curr_ratio', 'quick_ratio', 'Accrual', 'RD_SALE', 'adv_sale', 'staff_sale',
                       'efftax', 'GProf', 'aftret_eq', 'aftret_equity', 'aftret_invcapx', 'gpm', 'npm', 'opmad',
                       'opmbd', 'pretret_earnat', 'pretret_noa', 'ptpm', 'roa', 'roce', 'roe', 'de_ratio',
                       'debt_assets', 'debt_at', 'debt_capital', 'intcov', 'intcov_ratio', 'dpr', 'PEG_1yrforward',
                       'PEG_ltgforward', 'PEG_trailing', 'bm', 'capei', 'divyield', 'evm', 'pcf', 'pe_exi', 'pe_inc',
                       'pe_op_basic', 'pe_op_dil', 'ps', 'ptb', 'be']

    finratios_descriptions = ['Capitalization ratio','Common equity/Invested capital','Long-term debt/Invested capital'
        ,'Total debt/Invested capital','Asset turnover','Inventory turnover','Payables turnover','Receivables turnover',
        'Sales/Stockholders equity','Sales/Invested capital','Sales/Working capital','Inventory/Current assets',
        'Receivables/Current assets','Free cash flow/Operating cash flow','Operating CF/Current liabilities',
        'Cash flow/Total debt','Cash balance/Total liabilities','Cash flow margin','Short-term debt/Total debt',
        'Profit before depreciation/Current liabilities','Current liabilities/Total liabilities','Total debt/EBITDA',
        'Long-term debt/Book equity','Interest/Average long-term debt','Interest/Average total debt',
        'Long-term debt/Total liabilities','Total liabilities/Total tangible assets','Cash conversion cycle (days)',
        'Cash ratio','Current ratio','Quick ratio (acid test)','Accruals/Average assets','Research and development/Sales',
        'Avertising expenses/Sales','Labor expenses/Sales','Effective tax rate','Gross profit/Total assets',
        'After-tax return on average common equity','After-tax return on total stockholders equity',
        'After-tax return on invested capital','Gross profit margin','Net profit margin','Operating profit margin after depreciation',
        'Operating profit margin before depreciation','Pre-tax return on total earning assets',
        'Pre-tax return on net operating assets','Pre-tax profit margin','Return on assets',
        'Return on capital employed','Return on equity','Total debt/Equity','Total debt/Total assets',
        'Total debt/Total assets','Total debt/capital','After-tax interest coverage','Interest coverage ratio',
        'Dividend payout ratio','Forward P/E to 1-year growth (PEG) ratio',
        'Forward P/E to long-term growth (PEG) ratio','Trailing P/E to growth (PEG) ratio',
        'Book/Market','Shillers cyclically adjusted P/E Ratio','Dividend yield','Enterprise value multiple',
        'Price/Cash flow','P/E (diluted, excl. EI)','P/E (diluted, incl. EI)',
        'Price/Operating earnings (basic, excl. EI)','Price/Operating earnings (diluted, excl. EI)',
        'Price/Sales','Price/Book','Book equity']

    mapWRDSfinratios = dict(zip(transform_names(finratios_names), finratios_descriptions))

    # GHZrps
    GHZrps_names = ['beta', 'betasq', 'ep', 'mve', 'dy', 'sue', 'chfeps', 'bm', 'mom36m', 'fgr5yr', 'lev', 'currat',
                    'pchcurrat', 'quick', 'pchquick', 'salecash', 'salerec', 'saleinv', 'pchsaleinv', 'cashdebt',
                    'baspread', 'mom1m', 'mom6m', 'mom12m', 'depr', 'pchdepr', 'mve_ia', 'cfp_ia', 'bm_ia', 'sgr',
                    'chempia', 'IPO', 'divi', 'divo', 'sp', 'acc', 'turn', 'pchsale_pchinvt', 'pchsale_pchrect',
                    'pchcapx_ia', 'pchgm_pchsale', 'pchsale_pchxsga', 'nincr', 'indmom', 'ps', 'dolvol', 'std_dolvol',
                    'std_turn', 'sfe', 'nanalyst', 'disp', 'chinv', 'idiovol', 'grltnoa', 'rd', 'cinvest', 'tb', 'cfp',
                    'roavol', 'lgr', 'egr', 'ill', 'age', 'ms', 'pricedelay', 'rd_sale', 'rd_mve', 'retvol', 'herf',
                    'grcapex', 'zerotrade', 'chmom', 'roic', 'aeavol', 'chnanalyst', 'agr', 'chcsho', 'chpmia',
                    'chatoia', 'ear', 'rsup', 'stdcf', 'tang', 'sin', 'hire', 'cashpr', 'roaq', 'invest', 'realestate',
                    'absacc', 'stdacc', 'chtx', 'maxret', 'pctacc', 'cash', 'gma', 'orgcap', 'secured', 'securedind',
                    'convind']

    GHZrps_descriptions = ['Beta', 'Beta squared', 'Earnings-to-price', 'Firm size (market cap)', 'Dividends-to-price',
                           'Unexpected quarterly earnings', 'Change in forecasted annual EPS', 'Book-to-market',
                           '36-month momentum', 'Forecasted growth in 5-year EPS', 'Leverage', 'Current ratio',
                           '% change in current ratio', 'Quick ratio', '% change in quick ratio', 'Sales-to-cash',
                           'Sales-to-receivables', 'Sales-to-inventory', '% change in sales-to-inventory',
                           'Cash flow-to-debt', 'Illiquidity (bid-ask spread)', '1-month momentum', '6-month momentum',
                           '12-month momentum', 'Depreciation-to-gross PP&E', '% change in depreciation-to-gross PP&E',
                           'Industry-adjusted firm size', 'Industry-adjusted cash flow-to-price ratio',
                           'Industry-adjusted book-to-market', 'Annual sales growth',
                           'Industry-adjusted change in employees', 'New equity issue', 'Dividend initiation',
                           'Dividend omission', 'Sales-to-price', 'Working capital accruals', 'Share turnover',
                           '% change in sales - % change in inventory',
                           '% change in sales - % change in accounts receivable',
                           '% change in CAPEX - industry % change in CAPEX',
                           '% change in gross margin - % change in sales', '% change in sales - % change in SG&A',
                           '# of consecutive earnings increases', 'Industry momentum', 'Financial statements score',
                           'Dollar trading volume in month t-2', 'Volatility of dollar trading volume',
                           'Volatility of share turnover', 'Scaled analyst forecast of one year ahead earnings',
                           '# of analysts covering stock', 'Dispersion in forecasted eps', 'Changes in inventory',
                           'Idiosyncratic return volatility', 'Growth in long term net operating assets', 'RD-increase',
                           'Corporate investment', 'Taxable income to book income', 'Cash flow-to-price',
                           'Earnings volatility', 'Change in long-term debt', 'Change in common shareholder equity',
                           'Illiquidity', '# of years since first Compustat coverage', 'Financial statements score',
                           'Price delay', 'R&D-to-sales', 'R&D-to-market cap', 'Return volatility',
                           'Industry sales concentration', '% change over two years in CAPEX', 'Zero-trading days',
                           'Change in 6-month momentum', 'Return on invested capital',
                           'Abnormal volume in earnings announcement month', 'Change in # analysts', 'Asset growth',
                           'Change in shares outstanding', 'Industry-adjusted change in profit margin',
                           'Industry-adjusted change in asset turnover', '3-day return around earnings announcement',
                           'Revenue surprise', 'Cash flow volatility', 'Debt capacity-to-firm tangibility', 'Sin stock',
                           'Employee growth rate', 'Cash productivity', 'ROA', 'CAPEX and inventory',
                           'Real estate holdings', 'Absolute accruals', 'Accrual volatility', 'Change in tax expense',
                           'Maximum daily return in prior month', 'Percent accruals', 'Cash holdings',
                           'Gross profitability', 'Organizational capital', 'Secured debt-to-total debt',
                           'Secured debt indicator', 'Convertible debt indicator']

    mapGHZrps = dict(zip(transform_names(GHZrps_names), GHZrps_descriptions))

    # return lags
    nlags = 60
    retlags_names = [f'ret_lag{i}' for i in range(1, nlags + 1)]
    retlags_descriptions = [f'Month $t-{i}$' for i in range(1, nlags + 1)]
    mapRetLags = dict(zip(retlags_names, retlags_descriptions))

    # PCs
    nlags = 100
    pc_names = [f'PC{i}' for i in range(1, nlags + 1)]
    pc_descriptions = [f'PC {i}' for i in range(1, nlags + 1)]
    mapPCs = dict(zip(pc_names, pc_descriptions))

    # concatenate all maps
    map_combined = {}
    map_combined.update(mapGHZrps)
    map_combined.update(mapRetLags)
    map_combined.update(mapPCs)
    map_combined.update(mapAnomalies)
    map_combined.update(mapWRDSfinratios)

    return map_combined


def transform_names(names):
    """缩短名称为12个符号，如SAS中实现的那样"""
    return [name[:12] for name in names]