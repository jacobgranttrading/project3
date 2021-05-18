#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import all our modules and libraries 

import pandas as pd
import yfinance as yf
import time
from alpha_vantage.timeseries import TimeSeries
import requests
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from pathlib import Path
import hvplot.pandas


# In[3]:


#Create function to pull data from Alphavantage

def pull_alpha_data(function, ticker): 

    base_url = 'https://www.alphavantage.co/query?'
    params = {'function': function,
             'symbol': ticker,
             'apikey': '0GTFFL66G1VMDZJ5'}

    response = requests.get(base_url, params=params)
    data = response.json()
    return data


# In[5]:


#Pull data from yfinance 
'''
ticker_df = {}

td = timedelta(730)
end_date = datetime.now()
start_date = end_date - td

user_tickers = input("Enter the symbbols of the stock you wish to analyze seperated by a comma: i.e SPY, AAPL, TSLA")
user_tickers = user_tickers.split(', ')

short_ticker_list = user_tickers

short_ticker_list.append('SPY', 'AAPL', 'FB', 'AMZN', 'GOOG')

print(short_ticker_list)

#for loop for iterating through ticker list
for ticker in short_ticker_list:
    print(f'Checking Data for Ticker {ticker}')
    #statement for continuing loop in the case of an error from Yahoo Finance Package & getting ticker symbol data
    try: 
        data_df = yf.download(ticker, start=start_date, end=end_date)
        data_df['ticker'] = ticker
        data_df.reset_index(inplace=True)
        #Include the header on the 1st file write 
        if short_ticker_list.index(ticker) == 0:
            mode = None
            header = True
        #ignoring header on subsequent writes
        else:
            mode = 'a'
            header = False
        #writing dataframe to csv
        data_df.to_csv('project_ticker_data.csv',mode='a',header=header)
        time.sleep(2)
        
    except:
        print(f'Data for ticker: {ticker} is not available')
        pass


# In[6]:


#Pull second set of data from alphavantage
for ticker in short_ticker_list:
    print(f'Checking Data for Ticker {ticker}')
    #statement for continuing loop in the case of an error from Yahoo Finance Package & getting ticker symbol data
    try:
        data_df_2 = pull_alpha_data('OVERVIEW', ticker)
        data_df_df = pd.DataFrame([data_df_2])
        data_df_df['ticker'] = ticker     
        data_df_df.reset_index(inplace=True)
        #Include the header on the 1st file write 

        #print(short_ticker_list.index(ticker))    
        #input('Hold')

        if short_ticker_list.index(ticker) == 0:
            mode = 'w'
            header = True
        #ignoring header on subsequent writes
        else:
            mode = 'a'
            header = False
        #writing dataframe to csv
        
        data_df_df.to_csv('project_ticker_data_2.csv',mode=mode, header=header)
    
        time.sleep(15)
        
    except:
        print(f'Data for ticker: {ticker} is not available')
        pass

'''
# In[7]:


# Read in and clean data
'''
project_1 = pd.read_csv('project_ticker_data.csv')
project_2 = pd.read_csv('project_ticker_data_2.csv')

project_1.set_index('ticker',inplace=True)
project_1.drop(columns=['Unnamed: 0'],inplace=True)

project_2.set_index('ticker',inplace=True)
project_2.drop(columns=['Unnamed: 0'],inplace=True)


# In[8]:


clean_data = project_1[[ 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
clean_data.tail()


# In[9]:


project_2


# In[10]:


clean_data_2 = project_2[['QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY', 'ProfitMargin',
                          '52WeekHigh', '52WeekLow', 'Beta', 'EPS', '50DayMovingAverage', '200DayMovingAverage', 'AnalystTargetPrice']]



clean_data_2.head()


# In[11]:


#joining cleaned data into a single csv

new_data = clean_data.join(clean_data_2, how='outer', on=clean_data.index)
new_data.dropna(inplace=True)
new_data.drop(columns='key_0', inplace=True)
#new_data.drop(axis=0, index='ticker', inplace=True)
new_data.index.name = 'Ticker'
new_data['Close'] = new_data['Close'].astype("float")
new_data["MADelta"] = new_data["50DayMovingAverage"] - new_data["200DayMovingAverage"]
new_data["52wkhighpercent"] = new_data["Close"] / (new_data["52WeekHigh"])
new_data["52wklowpercent"] = new_data["Close"] / (new_data["52WeekLow"])

new_data.to_csv('join_data.csv')
print(new_data)


# In[12]:


new_data['Close'] = new_data['Close'].astype("float")
new_data.dtypes
#type(new_data["52WeekHigh"])
#new_data["52wkhighpercent"] = new_data[["Close"]].div(new_data["52WeekHigh"])

#new_data["52wkhighpercent"] = new_data["Close"]/new_data["52WeekHigh"]

#new_data["52wkhighpercent"]
new_data['Close']


# In[13]:


#create new data frames to make ploting information easier 

#read in dataframe
data_df = pd.read_csv(Path("join_data.csv"), index_col="Ticker")


#Creating dataframe with unique ticker row & information
summary_df = data_df.drop(columns=['Date'])
unique_sum_df = summary_df.reset_index().drop_duplicates(subset='Ticker',keep='first').set_index('Ticker')

#Plotting Top Earnings Growth YOY Companies
quart_earnings_df = unique_sum_df.sort_values('QuarterlyEarningsGrowthYOY',ascending=False)

#Creating Moving Average Delta DF
madelta_earnings_df = unique_sum_df.sort_values('MADelta', ascending=False)

#Creating 52 Wk High Percentage DF
yearhighpercentage_df = unique_sum_df.sort_values('52wkhighpercent', ascending=False)

#Creating 52 Wk Low Percentage DF
yearlowpercentage_df = unique_sum_df.sort_values('52wklowpercent', ascending=False)

#Creating beta DF
beta_df = unique_sum_df.sort_values('Beta', ascending=True)

top_earnings_df = quart_earnings_df.head(15)
#top_unique_df.head(10)
earnings_graph = top_earnings_df.hvplot.barh(x='Ticker', y='QuarterlyEarningsGrowthYOY',flip_yaxis=True)


#Plotting Top Earnings Per Share Companies
eps_df = unique_sum_df.sort_values('EPS',ascending=False)
top_eps_df = eps_df.head(15)
eps_graph = top_eps_df.hvplot.barh(x='Ticker', y='EPS',flip_yaxis=True)



#Plotting Top Profit Margin Sectors
pm_df = unique_sum_df.sort_values('ProfitMargin',ascending=False)
top_pm_df = pm_df.head(15)
#profit_margin_graph = top_pm_df.hvplot.barh(x='Sector', y='ProfitMargin',flip_yaxis=True)

#Plotting Top Quarterly Revenue Sectors
revenue_df = unique_sum_df.sort_values('QuarterlyRevenueGrowthYOY',ascending=False)
top_revenue_df = revenue_df.head(15)
revenue_graph = top_revenue_df.hvplot.barh(x='Ticker', y='QuarterlyRevenueGrowthYOY',flip_yaxis=True)


# In[14]:


quart_earnings_df


# In[15]:


top_revenue_df


# In[16]:


madelta_earnings_df


# In[17]:


pm_df


# In[18]:


eps_df


# In[19]:


yearhighpercentage_df


# In[20]:


yearlowpercentage_df


# In[21]:


beta_df


# In[22]:


unique_sum_df


# In[23]:


#Rank Stocks
fundamental_measures = unique_sum_df.loc[:,['QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY', 'ProfitMargin', 'EPS','MADelta', '52wkhighpercent', '52wklowpercent']]
ranked_stocks = fundamental_measures.rank(axis=0, ascending=True, method='dense').astype(int)
ranked_stocks['sum_total'] = (ranked_stocks['QuarterlyEarningsGrowthYOY'] * 0.15) + (ranked_stocks['QuarterlyRevenueGrowthYOY'] * 0.15) + (ranked_stocks['ProfitMargin'] * 0.15) + (ranked_stocks[ 'EPS'] * 0.1) +  (ranked_stocks[ 'MADelta'] * 0.25) +  (ranked_stocks['52wkhighpercent'] * 0.1) +  (ranked_stocks['52wklowpercent'] * 0.1)
 


# In[51]:


ranked_stocks = ranked_stocks.sort_values('sum_total',ascending=False)
ranked_stocks
recommendation


# In[29]:


ranked_stocks.index
    


# In[25]:





# In[60]:
'''

#Function to pull fundamental data and make recommendation

def analyze_stocks(tickers, debug=False):
    ticker_df = {}

    td = timedelta(730)
    end_date = datetime.now()
    start_date = end_date - td

    short_ticker_list = tickers

    if debug:
        print(short_ticker_list)

    #for loop for iterating through ticker list
    for ticker in short_ticker_list:
            if debug:
                print(f'Checking Data for Ticker {ticker}')
        #statement for continuing loop in the case of an error from Yahoo Finance Package & getting ticker symbol data
            try: 
                data_df = yf.download(ticker, start=start_date, end=end_date)
                data_df['ticker'] = ticker
                data_df.reset_index(inplace=True)
                
                #Include the header on the 1st file write 
                if short_ticker_list.index(ticker) == 0:
                    mode = None
                    header = True
                #ignoring header on subsequent writes
                else:
                    mode = 'a'
                    header = False
                    #writing dataframe to csv
                    data_df.to_csv('project_ticker_data.csv',mode='a',header=header)
                    time.sleep(2)

            except:
                if debug:
                    print(f'Data for ticker: {ticker} is not available')
            pass
        
            for ticker in short_ticker_list:
                if debug:
                    print(f'Checking Data for Ticker {ticker}')
    
    #statement for continuing loop in the case of an error from Yahoo Finance Package & getting ticker symbol data
    try:
        data_df_2 = pull_alpha_data('OVERVIEW', ticker)
        data_df_df = pd.DataFrame([data_df_2])
        data_df_df['ticker'] = ticker     
        data_df_df.reset_index(inplace=True)
        #Include the header on the 1st file write 

        #print(short_ticker_list.index(ticker))    
        #input('Hold')

        if short_ticker_list.index(ticker) == 0:
            mode = 'w'
            header = True
        #ignoring header on subsequent writes
        else:
            mode = 'a'
            header = False
        #writing dataframe to csv
        
        data_df_df.to_csv('project_ticker_data_2.csv',mode=mode, header=header)
    
        time.sleep(15)
        
    except:
        if debug:
            print(f'Data for ticker: {ticker} is not available')
        pass
    
    #read in data
    project_1 = pd.read_csv('project_ticker_data.csv')
    project_2 = pd.read_csv('project_ticker_data_2.csv')

    project_1.set_index('ticker',inplace=True)
    project_1.drop(columns=['Unnamed: 0'],inplace=True)

    project_2.set_index('ticker',inplace=True)
    project_2.drop(columns=['Unnamed: 0'],inplace=True)
    
    #clean data
    clean_data = project_1[[ 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    clean_data.tail()
    
    clean_data_2 = project_2[['QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY', 'ProfitMargin',
                          '52WeekHigh', '52WeekLow', 'Beta', 'EPS', '50DayMovingAverage', '200DayMovingAverage', 'AnalystTargetPrice']]
    
    #joining cleaned data into a single csv

    new_data = clean_data.join(clean_data_2, how='outer', on=clean_data.index)
    new_data.dropna(inplace=True)
    new_data.drop(columns='key_0', inplace=True)
    #new_data.drop(axis=0, index='ticker', inplace=True)
    new_data.index.name = 'Ticker'
    new_data['Close'] = new_data['Close'].astype("float")
    new_data["MADelta"] = new_data["50DayMovingAverage"] - new_data["200DayMovingAverage"]
    new_data["52wkhighpercent"] = new_data["Close"] / (new_data["52WeekHigh"])
    new_data["52wklowpercent"] = new_data["Close"] / (new_data["52WeekLow"])

    new_data.to_csv('join_data.csv')
    if debug:
        print(new_data)
    
    #create new data frames to make ploting information easier 

    #read in dataframe
    data_df = pd.read_csv(Path("join_data.csv"), index_col="Ticker")


    #Creating dataframe with unique ticker row & information
    summary_df = data_df.drop(columns=['Date'])
    unique_sum_df = summary_df.reset_index().drop_duplicates(subset='Ticker',keep='first').set_index('Ticker')

    #Plotting Top Earnings Growth YOY Companies
    quart_earnings_df = unique_sum_df.sort_values('QuarterlyEarningsGrowthYOY',ascending=False)

    #Creating Moving Average Delta DF
    madelta_earnings_df = unique_sum_df.sort_values('MADelta', ascending=False)

    #Creating 52 Wk High Percentage DF
    yearhighpercentage_df = unique_sum_df.sort_values('52wkhighpercent', ascending=False)

    #Creating 52 Wk Low Percentage DF
    yearlowpercentage_df = unique_sum_df.sort_values('52wklowpercent', ascending=False)

    #Creating beta DF
    beta_df = unique_sum_df.sort_values('Beta', ascending=True)

    eps_df = unique_sum_df.sort_values('EPS',ascending=False)
    pm_df = unique_sum_df.sort_values('ProfitMargin',ascending=False)
    revenue_df = unique_sum_df.sort_values('QuarterlyRevenueGrowthYOY',ascending=False)
    
    
    #Rank Stocks
    fundamental_measures = unique_sum_df.loc[:,['QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY', 'ProfitMargin', 'EPS','MADelta', '52wkhighpercent', '52wklowpercent']]
    ranked_stocks = fundamental_measures.rank(axis=0, ascending=True, method='dense').astype(int)
    ranked_stocks['sum_total'] = (ranked_stocks['QuarterlyEarningsGrowthYOY'] * 0.15) + (ranked_stocks['QuarterlyRevenueGrowthYOY'] * 0.15) + (ranked_stocks['ProfitMargin'] * 0.15) + (ranked_stocks[ 'EPS'] * 0.1) +  (ranked_stocks[ 'MADelta'] * 0.25) +  (ranked_stocks['52wkhighpercent'] * 0.1) +  (ranked_stocks['52wklowpercent'] * 0.1)
    ranked_stocks = ranked_stocks.sort_values('sum_total',ascending=False)
    recommendation = ranked_stocks.head(3)
    recommended_stocks = recommendation.index.to_list()
    
    return recommended_stocks


# In[59]:

'''
recommendation = ranked_stocks.head(3)
recommendation.index.to_list()


# In[49]:


analyze_stocks(short_ticker_list)


# In[43]:


import pandas as pd
from webull import paper_webull
from webull import webull

wb = webull()
pwb = paper_webull()
data = pwb.login('etlivefree@gmail.com','liveFree0815!','TBpython','422834')
pwb.refresh_login()

def execute_webull_option(optionId,limit_price,action='BUY',quantity=1,order_type='LMT',enforce='GTC'):
    # Places an options trade from an option Id via the Webull Platform; returns the result of the trade
    result = wb.place_order_option(optionId=optionId, lmtPrice=limit_price, action=action, orderType=order_type, enforce=enforce,quant=quantity)
    return result

def get_webull_options(ticker):
    options = wb.get_options(ticker)

    options_dict = {}
    official_options_list = []
    #loops for inputting options data in a dataframe
    for option_iter in range(0,len(options)):
        # Loop for determining Strike Price, Calls, Puts
        list_option = options[option_iter]
        #print(list_option)
        for key in list_option:
            #strike_price = list_option[key]
            if key != 'strikePrice':
                temp_dict = list_option[key]
                temp_dict.update({'ask_price': temp_dict['askList'][0]['price']})
                temp_dict.update({'ask_volume': temp_dict['askList'][0]['volume']})
                temp_dict.update({'bid_price': temp_dict['bidList'][0]['price']})
                temp_dict.update({'bid_volume': temp_dict['bidList'][0]['volume']})
                official_options_list.append(temp_dict)

    #Turning summarized list into a dataframe
    options_df = pd.DataFrame(official_options_list)

    #reordering columns
    official_options_df = options_df[
        ['tickerId', 'unSymbol', 'symbol', 'direction', 'strikePrice', 'ask_price','ask_volume', 'bid_price','bid_volume', 'expireDate', 'tradeTime',
         'tradeStamp', 'volume', 'close', 'preClose', 'open', 'high', 'low', 'delta', 'vega', 'gamma', 'theta', 'rho',
         'changeRatio', 'change', 'weekly', 'activeLevel', 'openIntChange']]


    options_df.set_index('tickerId', inplace=True)
    return official_options_df

def placeWebullOrder(stock, price, num_shares, orderType='LMT', action='BUY', enforce='GTC'):
    pwb.place_order(stock=stock, orderType=orderType, action=action, enforce=enforce, price=price, quant=num_shares)


# In[ ]:





# In[ ]:





# In[ ]:

'''


