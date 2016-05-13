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

def sample_portfolio_weights(n, sum):
	dirichlet = np.random.dirichlet(np.ones(n), size=sum)
	return list(dirichlet.reshape(-1)) 

def compute_ewma(df, lam):
	mean = df.mean()
	ewma = {}

	for asset in get_assets(df):
		squares = 0

		for i in range(len(df.index)-1, 0, -1):
			squares += pow(lam, i) * pow(df[asset][i] - mean[asset], 2)

		ewma[asset] = np.sqrt((1-lam) * squares)

	return ewma

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
		results[asset] = predictions[-1]

	return results

def utl(profit, EWMA, beta): 
	return 1 - np.exp(-(profit / np.sqrt(EWMA)))

def var(df, weights):
	assets = get_assets(df)
	df = df.pct_change(1)
	df.dropna(inplace=True)
	corr = df.corr()
	C = corr.as_matrix()
	W = np.array(weights)
	temp = np.dot(W, C)
	return np.dot(temp, W)

def test(beta):
	print('Running monte carlo simulation')
	all_weights = []
	
	for i in range(1, MAX_ITERS):
		print('Percent Done: \t' + str(float(i)/float(MAX_ITERS)*100)+' %')
		expected_utility = {}

		weights = sample_portfolio_weights(len(assets), 1)
		weights = dict(zip(get_assets(df), weights))
		all_weights.append(weights)

		var_portfolio = var(df, weights.values())
		if var_portfolio > beta:
			print('Dropping the trial')
			best[:, i] = np.zeros((5,)) # drop the trial 
			continue

		std_w = {}
		ewma_w = {}
		profit = {}
		EU = {}
		profit_true = {}

		for asset in get_assets(df):
			std_w[asset] = std[asset] * weights[asset]
			ewma_w[asset] = ewma[asset] * weights[asset]
			# profit[asset] = (preds[asset] - df[asset][-forward]) * weights[asset]
			profit[asset] = (preds[asset] - df[asset][-forward]) * weights[asset]
			# profit_true[asset] = (df[asset][-1] - df[asset][-forward]) * weights[asset]
			profit_true[asset] = (df[asset][-1] - df[asset][-forward]) * weights[asset]
			EU[asset] = utl(profit[asset], ewma_w[asset], beta)
		
		best[0:5, i] = [sum(EU.values()), var_portfolio, sum(ewma_w.values()), sum(profit.values()), sum(profit_true.values())]
		print(sum(profit.values()))

	bestInx = np.argmax(best[3,:])
	returns.append(best[3, bestInx])
	returns_true.append(best[4, bestInx])
	optimal_weights.append(all_weights[bestInx])
	print( "The maximum expected profit is: %f") % best[3, bestInx]
	print( "True profit is: %f") % best[4, bestInx]





useSavedData = True

# define the desired portfolio characteristics
MAX_ITERS = 10000	# max number of iterations
lam = .94  			# exponential decay number 
exit_date = 12  	# when you sell stocks (in months)
window = 6    # how far back you want to window reg. (in months)
delta = .99			# discount factor 
start_date = datetime.datetime(2015, 1, 1)
SAVED_FILE_NAME = 'prices.csv'
forward = 200

# Create DataFrame by either downloading new data from yahoo finance or reading previously downloaded data from a file
assets = {'GOOGLE': 'GOOG', 'APPLE': 'AAPL', 'CAT': 'CAT', 'SPDR_GOLD': 'GLD', 
'OIL': 'OIL','NATURAL_GAS': 'GAZ', 'USD': 'UUP', 'GOLDMANSACHS': 'GS', 'DOMINION': 'D'}

if useSavedData == False:
	print('Downloading data from Yahoo! Finance')
	df = get_data(assets, start_date)
	df.to_csv(SAVED_FILE_NAME, sep='\t')
else:
	print('Reading data from file')
	df = pd.read_csv(SAVED_FILE_NAME, sep='\t', parse_dates='Date', index_col='Date')

std = df.std()
corr = df.corr()
mean = df.mean()
ewma = compute_ewma(df, lam)
preds = predictFutureProfit(df, forward=forward)
best = np.zeros((5, MAX_ITERS))
optimal_weights = []
returns = []
returns_true = []

betas = np.arange(0.1, 1.1, 0.1)
for beta in betas:
	test(beta)

with open("optimal_allocations.csv",'w') as f:
	wr = csv.writer(f, dialect='excel')
	for d in optimal_weights:
		wr.writerow(d.values())

plt.plot(betas, returns)
# plt.plot(betas, returns_true, color='green')
plt.show()