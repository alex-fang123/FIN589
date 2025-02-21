import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

start_date = "01/01/2023"
end_date = "12/31/2023"

connection_string = (
  "postgresql+psycopg2://"
 f"{os.getenv('WRDS_USER')}:{os.getenv('WRDS_PASSWORD')}"
  "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
)

wrds = create_engine(connection_string, pool_pre_ping=True)

crsp_monthly_query = (
  "SELECT msf.permno, date_trunc('month', msf.mthcaldt)::date AS date, "
         "msf.mthret AS ret, msf.shrout, msf.mthprc AS altprc, "
         "ssih.primaryexch, ssih.siccd "
    "FROM crsp.msf_v2 AS msf "
    "INNER JOIN crsp.stksecurityinfohist AS ssih "
    "ON msf.permno = ssih.permno AND "
       "ssih.secinfostartdt <= msf.mthcaldt AND "
       "msf.mthcaldt <= ssih.secinfoenddt "
   f"WHERE msf.mthcaldt BETWEEN '{start_date}' AND '{end_date}' "
          "AND ssih.sharetype = 'NS' "
          "AND ssih.securitytype = 'EQTY' "  
          "AND ssih.securitysubtype = 'COM' " 
          "AND ssih.usincflg = 'Y' " 
          "AND ssih.issuertype in ('ACOR', 'CORP') " 
          "AND ssih.primaryexch in ('N', 'A', 'Q') "
          "AND ssih.conditionaltype in ('RW', 'NW') "
          "AND ssih.tradingstatusflg = 'A'"
)

crsp_monthly = (pd.read_sql_query(
    sql=crsp_monthly_query,
    con=wrds,
    dtype={"permno": int, "siccd": int},
    parse_dates={"date"})
  .assign(shrout=lambda x: x["shrout"]*1000)
)