import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import os 
# os.getcwd()
# os.chdir()
# os.chdir('/Users/RaymonYue/Desktop/MC_Trading_Proj')
# ↑ these were used when I debugged in terminal

# In this project we did a simple pairs trading strat
# using Netflix and Google as two companies to pairs trade with
# Intuition is based on our preliminary data exploration
# which shows that in the year 2013, among the companies we considered,
# google and netflix prices have the strongest correlation
nflx = pd.read_csv('NFLX.csv', index_col=0, parse_dates=[0])
goog = pd.read_csv('GOOG.csv', index_col=0, parse_dates=[0])
iyw = pd.read_csv('IYW.csv', index_col=0, parse_dates=[0])

# Only data after 2013 were used in thes backtest
nflx_2013 = nflx[nflx.index.year >= 2013]
goog_2013 = goog[goog.index.year >= 2013]
iyw_2013 = iyw[iyw.index.year >= 2013]


goog_2013 = goog_2013.rename({"Close": "Google"},axis = 1)
nflx_2013 = nflx_2013.rename({"Close": "Netflix"}, axis = 1)
prices = pd.concat([goog_2013["Google"], nflx_2013["Netflix"]],axis = 1)
prices.head()

# the volatility of the IYW tech ETF was used to scale the width of our bollinger bands
# intuition is that when the tech industry is more volatile, 
# we'd expect mean reversion to take longer to occur
iyw_2013['std'] = iyw_2013['Close'].rolling(21).std()
# this calculation is rather arbitrary, but does scales the multiplier to between 0 and 1.5
iyw_2013['Multiplier'] = (20*iyw_2013['std']/iyw_2013['Close'])**0.5
iywmult = pd.DataFrame(iyw_2013['Multiplier'],iyw_2013.index)

# plot showing the multiplier's value over time
plt.plot(iywmult)
plt.title("Multiplier Values")
plt.xlabel("Date")
plt.ylabel("Multiplier")

# concat the multiplier together with Google and Netflix prices
prices_iyw_multiplier = pd.concat([prices, iyw_2013['Multiplier']], axis = 1)
prices_iyw_multiplier.describe()


def run_strategy(data, lookback, years, hr_lookback_months):
    # data is a data frame with Netflix and Google columns (probably could be generalized)
    # lookback is the number of lookback days used in rolling averages calculation
    # years is an array of year numbers to be used in the backtest
    # hr_look_back_nonths is the number of months used for hedge ratio look back
   monthly_trading_days = 21
   hr_lookback = monthly_trading_days * hr_lookback_months 
   data_year = data[data.index.year.isin(years)]
   df = data_year.copy()
   # finding hedge_ratio
   df['hedge_ratio'] = df['Google'].rolling(hr_lookback).corr(df['Netflix']) * df['Google'].rolling(hr_lookback).std() / df['Netflix'].rolling(hr_lookback).std()
   # and finding the spread
   df['spread'] = df['Google'] - df['hedge_ratio'] * df['Netflix']

   # enter value of our spread, used to calculate returns
   begin = df.loc[df.index[hr_lookback_months*monthly_trading_days-1], "spread"]
   print(begin)
   
   # Bollinger Bands calculations
   df['rolling_spread'] = df['spread'].rolling(lookback).mean() # lookback-day SMA of spread
   df['rolling_spread_std'] = df['spread'].rolling(lookback).std() # lookback-day rolling STD of spread
   df['upper_band'] = df['rolling_spread'] + (df["Multiplier"] * df['rolling_spread_std']) #upper = SMA + width * STD
   df['lower_band'] = df['rolling_spread'] - (df["Multiplier"] * df['rolling_spread_std']) #lower = SMA - width * STD

   # using the Bollinger bands to compute our positions for each company
   df['Position Google'] = np.nan
   for date in df.index:
      # over the band sell to anticipate downwards mean reversion
      if df.loc[date, 'spread'] > df.loc[date, 'upper_band']: 
         df.loc[date, 'Position Google'] = -1
      # below the band buy to anticipate upwards mean reversion
      elif df.loc[date, 'spread'] < df.loc[date, 'lower_band']:
         df.loc[date, 'Position Google'] = 1
      elif (df.loc[date, 'spread'] >= df.loc[date, 'lower_band']) & (df.loc[date, 'spread'] <= df.loc[date, 'upper_band']):
         df.loc[date, 'Position Google'] = 0

   # use hedge ratio to find the position for the paired company
   df['Position Netflix'] = -df['hedge_ratio'] * df['Position Google'] 

   # Using our position to compute the P&L for each
   df['P&L Google'] = df['Position Google'] * df['Google'].diff().shift(-1)
   df['P&L Netflix'] = df['Position Netflix'] * df['Netflix'].diff().shift(-1)

   # finding the total P&L with each company
   df['P&L'] = df['P&L Google'] + df['P&L Netflix']

   # Adding P&L together to find the cumulative earnings of the strategy
   df["Cumulative Earnings"] = np.nan
   df.loc[df.index[hr_lookback_months*monthly_trading_days+lookback-2],"Cumulative Earnings"]=0
   for i in range(hr_lookback_months*monthly_trading_days+lookback-1,len(df.index),1):
     df.loc[df.index[i], "Cumulative Earnings"] = df.loc[df.index[i-1], "Cumulative Earnings"]+df.loc[df.index[i],"P&L"]
   # finding daily return of the strategy using P&L and the value of the day before
   # computed using cumulative earnings and the beginning value
   df["return"] = df["P&L"]/(begin+df['Cumulative Earnings'].shift(-1))

   return df

# sharpe ratio calculation
# rf assumed to be 0 here
def get_sharpe(r):
  if r.std():
    return (r.mean()) / r.std() * np.sqrt(252)
  return 0

# to optimize sharpe ratio, try a range of parameters on the training data
# which is selected to be 2014-2018 (while tested on 2019-2020, a rough 70%-30% split)
sharpes_dictionary = {} 
for lookback in range(1, 21):
  for hr_lookback_months in range(1, 15):
    params = (lookback, hr_lookback_months)
    my_result = run_strategy(prices_iyw_multiplier, lookback, [2014,2015,2016,2017,2018], hr_lookback_months)
    sharpes = get_sharpe(my_result['return'])
    sharpes_dictionary.update({params:sharpes})


# To ensure that the optimal value is not an outlier,
# create function to find average Sharpe in a neighborhood around a given set of parameters
# only choose a set of parameters around which the value is consistently high
# Higher average Sharpe in neighborhood should ensure that paramter values are optimized and 
# Sharpe function is smooth in that neighborhood
def average_neighborhood_sharpe(lookback, hr_lookback):
  total_neighborhood_sharpe = 0
  # sum up neighborhood and
  for i in range(lookback - 2, lookback + 3):
    for j in range(hr_lookback - 2, hr_lookback + 3):
      total_neighborhood_sharpe += sharpes_dictionary[(i,j)]
  average_neighborhood_sharpe = float(total_neighborhood_sharpe) / 25.0
  # find the neighborhood average
  return average_neighborhood_sharpe

# loop through the dictionary to find the optimal neighborhood
optimal_sharpe = -10
nb_sharpe = []
for lookback in range(3,19):
  for hr_lookback in range(3,13):
    if average_neighborhood_sharpe(lookback,hr_lookback) > optimal_sharpe:
      optimal_sharpe = average_neighborhood_sharpe(lookback,hr_lookback)
      optimal_paramter_values = (lookback,hr_lookback)
      nb_sharpe.append((optimal_sharpe,optimal_paramter_values))
print(optimal_sharpe,optimal_paramter_values)
# in this test set up turns out to be (5,10) that has the best Sharpe

# out of sample testing with 2019-2020 data
my_result = run_strategy(prices_iyw_multiplier, 5, [2019,2020],10)

plt.plot(my_result["Cumulative Earnings"])
plt.title("Strategy Performance")
plt.xlabel("Date")
plt.ylabel("Cumulative Earnings")

my_result.describe()
print('Sharpe Ratio:', get_sharpe(my_result['return']))
# in this test set-up, it has 0.5383 Sharpe Ratio

# A Bollinger Band plot sample, using the year 2020
my_result_2020 = my_result[my_result.index.year == 2020]
plt.plot(my_result_2020["spread"])
plt.plot(my_result_2020["upper_band"])
plt.plot(my_result_2020["lower_band"])
plt.title("Bolinger Bands Trading Example")
plt.ylabel("Price")
plt.xlabel("Date")
