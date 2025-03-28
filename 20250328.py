from my_scs import test
from my_scs import load_ff25
from my_scs import load_ff_anomalies
from my_scs import load_managed_portfolios

test.hello()

dates, ret, mkt, DATA, labels = load_ff25.load_ff25('./my_scs/Data/', True)

dates, ret, mkt, DATA = load_ff_anomalies.load_ff_anomalies('./my_scs/Data/', True)

dates, re, mkt, names, DATA = load_managed_portfolios.load_managed_portfolios("./my_scs/Data/Instruments/managed_portfolios_anom_d_50.csv", True)

