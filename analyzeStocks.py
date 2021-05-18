import yfinance as yf
import pandas as pd

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