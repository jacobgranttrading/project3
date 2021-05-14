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