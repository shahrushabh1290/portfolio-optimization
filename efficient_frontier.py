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

def get_assets(df):
	return sorted(df.columns.values)

def var(corr, weights, assets):
	C = corr.as_matrix()
	W = np.array(weights)
	temp = np.dot(W, C)
	return np.dot(temp, W)

def predictFutureProfit(df, forward):
	results = []

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
		results.append(predictions[-1] - df[asset][-forward])

	return results

def test(useSavedData = False):
	'''
	useSavedData: True if you want to use previously donwloaded data saved in a csv file
	'''

	# define the desired portfolio characteristics
	std_max = 0.2		# maximum standard deviation 
	MAX_ITERS = 10000 	# max number of iterations
	lam = .94  			# exponential decay number 
	exit_date = 12  	# when you sell stocks (in months)
	window = 6    # how far back you want to window reg. (in months)
	beta = .6			# personal risk adversion level
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

	prices = df
	df = df.pct_change(1)
	df.dropna(inplace=True)
	std = df.std()
	corr = df.corr()
	mean = df.mean()
	risk = []
	ret = []
	preds = predictFutureProfit(prices, forward)
	print get_assets(df)

	# for i in range(1, MAX_ITERS):
	# 	weights = sample_portfolio_weights(len(assets), 1)	
	# 	risk.append(var(corr, weights, get_assets(df)))
	# 	ret.append(np.dot(preds, weights))


	plt.plot(risk, ret, 'o')
	plt.show()

	# print('Running monte carlo simulation')
	# for i in range(1, MAX_ITERS):
	# 	print('Percent Done: \t' + str(float(i)/float(MAX_ITERS)*100)+' %')
	# 	expected_utility = {}

	# 	weights = sample_portfolio_weights(len(assets), 1)
	# 	weights = dict(zip(get_assets(df), weights))

	# 	var_portfolio = var(corr, weights, get_assets(df))
	# 	if var_portfolio == std_max:
	# 		best[:, i] = zeros(( (4+len(assets)) )) # drop the trial 
	# 		continue

	# 	std_w = {}
	# 	ewma_w = {}
	# 	profit = {}
	# 	EU = {}
	# 	profit_true = {}

	# 	for asset in get_assets(df):
	# 		std_w[asset] = std[asset] * weights[asset]
	# 		ewma_w[asset] = ewma[asset] * weights[asset]
	# 		profit[asset] = (preds[asset] - df[asset][-forward]) * weights[asset]
	# 		profit_true[asset] = (df[asset][-1] - df[asset][-forward]) * weights[asset]
	# 		EU[asset] = utl(profit[asset], ewma_w[asset], beta)
		
	# 	# calculate the variance of the portfolio
	# 	var_portfolio = var(corr, weights, get_assets(df))

	# 	best[0:5, i] = [sum(EU.values()), var_portfolio, sum(ewma_w.values()), sum(profit.values()), sum(profit_true.values())]
	# 	sorted_assest = sorted(weights.keys())
	# 	print(sum(profit.values()))
	# 	all_weights.append(weights)


	# maxP = max(best[0,:])
	# bestInx = np.argmax(best[0,:])
	# print( "The maximum profit to be made is: %f") % maxP
	# print(best[:, bestInx])
	# # find the column where the sum is equal to the max
	# opt = np.where(best[0,:] == maxP)
	# # OptimalAllocation = dict(zip(assets, [float(xx) for xx in best[2:,opt]]))
	# print('\n The optimal asset allocation in the portfolio is:')
	# print(all_weights[bestInx])
	# # print(OptimalAllocation)
	# return df, best, EU, profit

test(useSavedData=True)