from pyspark import SparkContext
import pandas as pd
from pandas.io.data import DataReader, DataFrame
import datetime
import matplotlib.pylab as plt
import statsmodels.api as sm
from collections import defaultdict
import pdb
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import csv

def mult(n):
	return n*forward.value

def get_data(assets, start_date):
	prices = {}
	for asset, ticker in assets.items():
		prices[asset] = get_close_price(ticker, start_date)

	df = DataFrame(prices)
	return df

def get_close_price(ticker, start_date):
	data = DataReader(ticker, "yahoo", start = start_date) 
	adjClose = data['Adj Close']
	return adjClose

def sample_portfolio_weights(n):
	assets = assets_b.value
	n = len(assets)
	dirichlet = np.random.dirichlet(np.ones(n), size=1)
	w = list(dirichlet.reshape(-1)) 
	return dict(zip(assets, w))

def get_assets(df):
	return df.columns.values

def predictFutureProfit(df, forward):
	results = {}

	for asset in get_assets(df):
		ts = df[asset]
		ts_log = np.log(ts)

		model = ARIMA(ts_log, order=(1, 1, 0))  
		results_ARIMA = model.fit(disp=-1)  
		predictions_diff = results_ARIMA.predict(2, len(ts.index)-1, dynamic=True)
		predictions_diff_cumsum = predictions_diff.cumsum()
		predictions_log = pd.Series(ts_log.ix[0], index=ts_log.index)
		predictions_log = predictions_log.add(predictions_diff_cumsum,fill_value=0)
		predictions = np.exp(predictions_log)
		results[asset] = predictions[-1] - df[asset][-forward]

	return results

def var(weights):
	assets = assets_b.value
	C = corr.value.as_matrix()
	W = np.array(weights.values())
	temp = np.dot(W, C)
	return np.dot(temp, W)

def compute_profit(weights):
	assets = assets_b.value
	profits = profits_b.value
	results = {}

	for asset in assets:
		results[asset] = profits[asset] * weights[asset]

	return results

# Process data
useSavedData = True

# define the desired portfolio characteristics
MAX_ITERS = 10000	# max number of iterations
start_date = datetime.datetime(2015, 1, 1)
SAVED_FILE_NAME = 'prices.csv'

# Create DataFrame by either downloading new data from yahoo finance or reading previously downloaded data from a file
assets_dict = {'GOOGLE': 'GOOG', 'APPLE': 'AAPL', 'CAT': 'CAT', 'SPDR_GOLD': 'GLD', 
'OIL': 'OIL','NATURAL_GAS': 'GAZ', 'USD': 'UUP', 'GOLDMANSACHS': 'GS', 'DOMINION': 'D'}

if useSavedData == False:
	print('Downloading data from Yahoo! Finance')
	df = get_data(assets_dict, start_date)
	df.to_csv(SAVED_FILE_NAME, sep='\t')
else:
	print('Reading data from file')
	df = pd.read_csv(SAVED_FILE_NAME, sep='\t', parse_dates='Date', index_col='Date')


#Start monte carlo on spark
sc = SparkContext()

# Broadcast variables
forward = sc.broadcast(200)
std = sc.broadcast(df.std())
mean = sc.broadcast(df.mean())
profits_b = sc.broadcast((predictFutureProfit(df, forward=200)))
beta = sc.broadcast(0.6)
df_diff = df.pct_change(1)
df_diff.dropna(inplace=True)
corr = sc.broadcast(df_diff.corr())
assets_b = sc.broadcast(get_assets(df))
iterations = range(MAX_ITERS)

# Calculate profits and risk
rdd = sc.parallelize(iterations)
weights_rdd = rdd.map(sample_portfolio_weights)
risk_rdd = weights_rdd.map(var)
profits_rdd = weights_rdd.map(compute_profit)

# Collect resutls
risk = risk_rdd.collect()
profits = profits_rdd.collect()
i = rdd.collect()

print('='*80)
print i
print('='*80)
print('Profits: ')
print profits
print('='*80)
print('Risk: ')
print risk
print('='*80)

ret = []
for profit in profits:
	ret.append(sum(profit.values()))

plt.plot(risk, ret, 'o')
plt.show()