import pandas as pd
from flask import Flask, render_template, request
import plotly.graph_objects as go
import tickerData as td
import executeTrade as et
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table as dtable
import dash
import time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Establishing Web App Environment
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app2 = Flask(__name___)
#@app2.route('/form')
def form():
    return render_template('form.html')


#@app2.route('/data/', methods=['POST','GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        return render_template('data.html', form_data=form_data)


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


def index():
    return render_template('index.html')

# ---------- Import and clean data (importing csv into pandas)

# Getting the SPY (S&P 500) ticker list
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
spy_df = table[0]
spy_list = spy_df.Symbol.to_list()

# User input for ticker symbol, short & long windows, & time sample period
ticker = input ('Enter the ticker symbol that you want to use: ')
short_window = int(input ('Enter the short window period (days): '))
long_window = int(input ('Enter the long window period (days): '))
period = int(input ('Enter the ticker sample period (days): '))

tickers = ticker.split(' ')
result = set(tickers)
ticker_list = list(result)

opts = [{'label': i, 'value': i} for i in ticker_list]

# Web App function to update the data
def update_graph(data_df, ticker=None, future_price=0):
    dff = data_df.copy()
    # Trade Signals
    # Plotly Graph Objects (GO)

    candlestick = go.Figure(
        data=[go.Candlestick(x=dff.index,
                             open=dff['Open'],
                             high=dff['High'],
                             low=dff['Low'],
                             close=dff['Close'],
                             )]
    )

    candlestick.update_layout(
        title_text="%s Stock Prices" % ticker,
        title_xanchor="center",
        title_font=dict(size=24),
        title_x=0.5,
    )
    candlestick.update_yaxes(title_text='Close Prices', tickprefix='$', color='blue')

    xover = go.Figure()
    xover.add_trace(go.Scatter(x=dff.index, y=dff['SMA%s' % long_window], name='Long', line=dict(color='blue')))
    xover.add_trace(go.Scatter(x=dff.index, y=dff['SMA%s' % short_window], name='Short', line=dict(color='brown')))
    # xover.add_trace(go.Scatter(x=dff.index, y=dff[dff['Entry/Exit']== 1.0]['Close'], name='Entry', line=dict(color='green',dash='dot',width=8)))
    # xover.add_trace(go.Scatter(x=dff.index, y=dff[dff['Entry/Exit'] == -1.0]['Close'], name='Exit', line=dict(color='red',dash='dot',width=8)))
    xover.add_trace(go.Scatter(x=dff.index, y=dff['Close'], name='Price', line=dict(color='orange')))
    xover.update_yaxes(title_text='Close Prices', tickprefix='$', color='blue')

    return candlestick, xover

plot_objects = {}

for tkr in range(0,len(ticker_list)):
    # Read in initial ticker price data
    print(ticker_list)
    symbol = ticker_list[tkr]
    print(symbol)
    df = td.getTickerPriceData(symbol,period='%id'%period,interval='1d')

    # Adding 1st level of trade signals to the initial dataframe
    signals_df = td.makeTickerDfSignals(df,interval='1d',short_window=short_window,long_window=long_window)
    signals_df.drop(columns='Close',inplace=True)


    # Joining the Price & 1st level of signals dataframe
    comb_df = pd.concat([df,signals_df],join='inner',axis=1)

    # Loading the models and back-testing the data w/ signals; returns dataframe
    all_df,recommendation, predicted_price, strike_price_call, strike_price_put = td.execute_backtest(comb_df,initial_capital=10000.00,shares=500)

    # Load Ticker Options Chain from the Webull Trading Application
    #if ticker != 'SPY':
        #options_df = et.get_webull_options(symbol)

    print(all_df[:5])

    candlestick,xover = update_graph(all_df,symbol,predicted_price)
    #Storing objects in a dictionary for deployment
    plot_objects.update([('ticker%s'%tkr,symbol),('candlestick%s'%tkr,candlestick),('xover%s'%tkr,xover)])

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background']},children = [

    html.Div([html.H1("ITS BOT-TIME with Dash", style={'text-align': 'center','color':'white'}),

    html.Div(id='output',children=[]),

    dcc.Graph(id='my_ticker_chart', figure=candlestick)

    ]),
   html.Div([
        html.H1(children='%s Long %i Short %i Crossover'%(plot_objects['ticker0'],long_window,short_window),style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='XOVER1',figure=plot_objects['xover0']),
    ]),
    html.Div([
        html.H1(children='%s Candlestick Chart'%(plot_objects['ticker1']),style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='Candlestick2',figure=plot_objects['candlestick1']),
    ]),
    html.Div([
        html.H1(children='%s Long %i Short %i Crossover' % (plot_objects['ticker1'], long_window, short_window),
                style={'text-align': 'center', 'color': 'white'}),

        html.Div(children=[]),

        dcc.Graph(id='XOVER2', figure=plot_objects['xover1']),
    ]),
    html.Div([
        html.H1(children='%s Candlestick Chart'%(plot_objects['ticker2']),style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='Candlestick3',figure=plot_objects['candlestick2']),
    ]),
    html.Div([
        html.H1(children='%s Long %i Short %i Crossover' % (plot_objects['ticker2'], long_window, short_window),
                style={'text-align': 'center', 'color': 'white'}),

        html.Div(children=[]),

        dcc.Graph(id='XOVER3', figure=plot_objects['xover2']),
    ]),
])
'''
html.P([
    html.Label('Choose your Ticker Symbol'),
    dcc.Dropdown(id='opt',options = opts, value = opts[0])],
    style = {'width':'400px','fontSize':'20px', 'padding-left':'100px','display':'inline-block'})
    #]),
])
@app.callback([Output('my_ticker_chart','figure'),Output('XOVER','figure')],Input('opt','value'))

])
'''


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
    #app2.run(host='localhost',port=5000)
