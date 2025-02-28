import pandas_datareader as pdr
import sqlite3
import pandas as pd
import os

if not os.path.exists("data"):
    os.makedirs("data")

start_date = "1960-01-01"
end_date = "2024-12-31"

tidy_finance = sqlite3.connect(database="data/tidy_finance_python.sqlite")

factors_ff3_monthly_raw = pdr.DataReader(
  name="F-F_Research_Data_Factors",
  data_source="famafrench",
  start=start_date,
  end=end_date)[0]

factors_ff3_monthly = (factors_ff3_monthly_raw
  .divide(100)
  .reset_index(names="date")
  .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
  .rename(str.lower, axis="columns")
  .rename(columns={"mkt-rf": "mkt_excess"})
)

(factors_ff3_monthly
  .to_sql(name="factors_ff3_monthly",
          con=tidy_finance,
          if_exists="replace",
          index=False)
)


# FF5
factors_ff5_monthly_raw = pdr.DataReader(
  name="F-F_Research_Data_Factors",
  data_source="famafrench",
  start=start_date,
  end=end_date)[0]

factors_ff5_monthly = (factors_ff5_monthly_raw
  .divide(100)
  .reset_index(names="date")
  .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
  .rename(str.lower, axis="columns")
  .rename(columns={"mkt-rf": "mkt_excess"})
)

(factors_ff5_monthly
  .to_sql(name="factors_ff5_monthly",
          con=tidy_finance,
          if_exists="replace",
          index=False)
)

# 10 industries
industries_10_ff_monthly_raw = pdr.DataReader(
  name="10_Industry_Portfolios",
  data_source="famafrench",
  start=start_date,
  end=end_date)[0]

industries_10_ff_monthly = (industries_10_ff_monthly_raw
  .divide(100)
  .reset_index(names="date")
  .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
  .rename(str.lower, axis="columns")
)

(industries_10_ff_monthly.to_sql(
    name="industries_10_ff_monthly",
    con=tidy_finance,
    if_exists="replace",
    index=False
    )
)

# 49 industries
industries_49_ff_monthly_raw = pdr.DataReader(
  name="49_Industry_Portfolios",
  data_source="famafrench",
  start=start_date,
  end=end_date)[0]

industries_49_ff_monthly = (industries_49_ff_monthly_raw
    .divide(100)
    .reset_index(names="date")
    .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
    .rename(str.lower, axis="columns")
    )

(industries_49_ff_monthly.to_sql(
    name="industries_49_ff_monthly",
    con=tidy_finance,
    if_exists="replace",
    index=False
    )
)
