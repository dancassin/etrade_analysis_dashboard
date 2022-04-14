# risk model: https://marderreport.com/creating-a-risk-model/
# https://www.vantharp.com/trading/wp-content/uploads/2018/06/A_Short_Lesson_on_R_and_R-multiple.pdf

import pandas as pd
import numpy as np
import datetime

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from pandas.io.formats import style

import plotly.express as px
from plotly.figure_factory import create_distplot


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])


# -----------------------------------------------------------
# Constants
today = pd.to_datetime(datetime.date.today())

year = today.year

ytd = pd.to_datetime(f'01-01-{year}')

t_minus_30 = today - datetime.timedelta(30)

t_minus_90 = today - datetime.timedelta(90)

ttm = today - datetime.timedelta(365)

time_periods = {1:ttm, 2:t_minus_90, 3:t_minus_30, 4:ytd}

# -----------------------------------------------------------
# Data Processing
DATE = '033122'
FILEPATH = f'./data/TTM_tax_lots_{DATE}.csv'

tax_lot_df = pd.read_csv(FILEPATH,
            usecols=[
                'Symbol', 'Quantity', 'Opening Date', 'Cost/Share $', 'Total Cost $',
                'Closing Date',	'Price/Share $', 'Proceeds $', 'Gain $'],
            skiprows=13,
            delimiter=',',
            skipfooter=2,
            engine='python'
)

# column formatting and selection
tax_lot_df.rename(columns = lambda x: x.lower().replace('$', '').strip().replace(' ','_'), inplace=True)

# replacing double dashes in symbol column with NaN and forward filling the stock ticker
tax_lot_df = tax_lot_df.replace('--', np.nan)
tax_lot_df['symbol']= tax_lot_df['symbol'].str.strip()
tax_lot_df['symbol'].replace('Sell', np.nan, inplace=True)
tax_lot_df['symbol'] = tax_lot_df['symbol'].ffill()

tax_lot_df.dropna(subset=['opening_date'], inplace=True)

tax_lot_df['opening_date'] = tax_lot_df['opening_date'].astype('Datetime64')
tax_lot_df['closing_date'] = tax_lot_df['closing_date'].astype('Datetime64')

tax_lot_df['pct_gain/loss'] = round((tax_lot_df['gain'] / tax_lot_df['total_cost']) * 100, 2)

tax_lot_df['category'] = pd.cut(tax_lot_df.gain, bins=[-5000, -0.000001, 5000],
    labels=['loss','gain'])

# # -----------------------------------------------------------
# # Layout
app.layout = dbc.Container([
    html.Div([
        html.Br(),
        html.H1('Trading Performance', style={'text-align':'left'}),
        html.Hr()
        ]
    ),
    html.Div([
        html.Br(),
        dcc.Slider(
            id='time_slider',
            min=1,
            max=4,
            step=None,
            marks={
                1 : 'TTM',
                2 : '90D',
                3 : '30D',
                4 : 'YTD',
            },
            value=1,
        ),
        
    ]),
    # html.Div(id='slider-drag-output'),
    html.Div(
        [
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Row(
                            
                                dcc.Graph(id='gain_hist', style={'height': '40vh'}),
                                #width={'size':9},
                            ),
                        dbc.Row(
                            
                                dcc.Graph(id='equity', style={'height': '40vh'}),
                                #width={'size':9},
                            ),
                    ]),
                    dbc.Col(    
                        [   
                            html.Br(),
                            # html.Br(),
                            dbc.Card(
                                dbc.CardBody(
                                    [   
                                        html.H6("Reward / Risk"), #className="card-subtitle"),
                                        html.H4("Title", #className="card-title", 
                                            id='risk_reward_num'),
                                    ]
                                ),
                                style={'text-align':'center'}
                                
                            ),
                            #width = {'size':2, 'offset':1},
                            html.Br(),
                            dbc.Card(
                                dbc.CardBody(
                                    [   
                                        html.H6("Winning / Losing"), #className="card-subtitle"),
                                        html.H4("Title", id='pct_winning_losing'),
                                    ]
                                ),
                                style={'text-align':'center'}
                                
                            ),
                            html.Br(),
                            dbc.Card(
                                dbc.CardBody(
                                    [   
                                        html.H6("Expectancy"), #className="card-subtitle"),
                                        html.H4("Title", id='expectancy'),
                                    ]
                                ),
                                style={'text-align':'center'}
                                
                            ),
                            html.Br(),
                            dbc.Card(
                                dbc.CardBody(
                                    [   
                                        html.H6("R"), #className="card-subtitle"),
                                        html.H4("Title", id='r_value'),
                                    ]
                                ),
                                style={'text-align':'center'}
                                
                            ),
                            html.Br(),
                            dbc.Card(
                                dbc.CardBody(
                                    [   
                                        html.H6("Last 10 Trades"), #className="card-subtitle"),
                                        html.H4("Title", id='ten_trades'),
                                    ]
                                ),
                                style={'text-align':'center'}
                                
                            ),
                            #
                        ],
                        width = {'size':2, #'offset':1
                        }
                    )    
                ]
            )
        ]
    ),
dcc.Store(id='sub_df')
])

# -----------------------------------------------------------
# Callbacks
@app.callback(Output('sub_df','data'),[Input('time_slider','value')])
def setting_date_filter(selected_time):
    sub_df = tax_lot_df[tax_lot_df['closing_date'] >= time_periods[selected_time]]

    return sub_df.to_json()

@app.callback(Output('gain_hist', 'figure'),
                #Output('slider-drag-output', 'children'), 
                [Input('sub_df', 'data')])
def avg_gain(jsonified_cleaned_data):

    df = pd.read_json(jsonified_cleaned_data)
    
    gain_df = df[df['category'] == 'gain']
    loss_df = df[df['category'] == 'loss']

    gain_mean_pct = round(gain_df['pct_gain/loss'].median(), 2)
    loss_mean_pct = round(loss_df['pct_gain/loss'].median(), 2)

    fig =px.histogram(
        df.sort_values(by='category'), x="pct_gain/loss",
        marginal='box', color='category',
        #histnorm = 'density',
        nbins=int(df.shape[0] / 2),
        color_discrete_map = {'gain':'#63C9C4', 'loss':'gray'},

        )
    
    fig.update_layout(
        xaxis_title_text= f'% Gain/Loss', # xaxis label
        yaxis_title_text='', # yaxis label
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend_x=.8,
        legend_y=.5,
    )
    
    fig.data[0].name = f'Gain<br>(Median: {gain_mean_pct}%)'
    fig.data[2].name = f'Loss<br>(Median: {loss_mean_pct}%)'

    fig.add_annotation(text=f"Max Gain: <br>{round(max(gain_df['pct_gain/loss']))}%",
                  xref="paper", yref="paper",
                  x=1, y=-0.2, showarrow=False)
    fig.add_annotation(text=f"Max Loss: <br>{round(min(loss_df['pct_gain/loss']))}%",
                  xref="paper", yref="paper",
                  x=0, y=-0.2, showarrow=False)

    # fig.add_trace(create_distplot(
    #     [df.sort_values(by='category')["pct_gain/loss"].values], 
    #     group_labels=['values'],
    #     show_hist=False, 
    #     ))
    # print(df.sort_values(by='category')["pct_gain/loss"].values)

    return fig

@app.callback(Output('equity', 'figure'),
                [Input('sub_df', 'data')])
def equity_graph(jsonified_cleaned_data):

    df = pd.read_json(jsonified_cleaned_data)
    #df['closing_date'] = df['closing_date'].astype('datetime64[ns]')
    df = df[['closing_date', 'gain']].groupby(by='closing_date').sum()
    df = df.sort_values(by='closing_date').reset_index()
    df['cumulative'] = df.gain.cumsum()
    
    fig =px.line(
        df, 
        x='closing_date', 
        y='cumulative', 
    )

    #fig.update_xaxes(visible=False)

    fig.update_layout(
        xaxis_title_text= 'Cumulative Gains', # xaxis label
        xaxis_showticklabels = False,
        yaxis_title_text='', # yaxis label
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        # legend_x=.8,
        # legend_y=.5,
    )
    

    fig.data[0].line.color = '#63C9C4'
    return fig



@app.callback([Output('risk_reward_num', 'children'),
            Output('pct_winning_losing', 'children'),
            Output('expectancy', 'children'),
            Output('r_value', 'children'),
            Output('ten_trades', 'children')], 
            [Input('sub_df', 'data')])
def update_reward_risk(jsonified_cleaned_data):

    df = pd.read_json(jsonified_cleaned_data)

    winning_trades = df[df['gain']>= 0]
    losing_trades = df[df['gain'] < 0]

    total_winning_trades = winning_trades.shape[0]
    total_losing_trades = losing_trades.shape[0]

    pct_winning_trades = round(total_winning_trades/df.shape[0]*100)
    pct_losing_trades = round(total_losing_trades/df.shape[0]*100)

    avg_amount_gain = round(winning_trades['gain'].median(), 2)
    avg_amount_loss = abs(round(losing_trades['gain'].median(), 2))

    avg_pct_gain = round(winning_trades['pct_gain/loss'].median(), 2)
    avg_pct_loss = abs(round(losing_trades['pct_gain/loss'].median(), 2))

    reward_risk = round((total_winning_trades * avg_amount_gain) / (total_losing_trades * avg_amount_loss), 2)
    
    # Expectancy = the average return of all your trades, 
    # i.e. what you can expect to make every time you roll the dice, on average.
    #(% of winning trades * avg winner) – (% of losing trades * avg loser)
    expectancy = round(((pct_winning_trades * avg_pct_gain) - (pct_losing_trades * avg_pct_loss)), 4) / 100

    # Frequency = the number of trades produced by the system each month or year.
    frequency = round((df.shape[0] * .8), 2)

    # objective = % return per month
    objective = .1

    # Risk model formula: Objective = Expectancy * R * Frequency
    # what percent risk, i.e. “R,” you should be using in order to achieve your 
    # monthly or annual return objective
    R = round((objective / expectancy / frequency), 4) * 100

    closing_date_sorted = df[['closing_date', 'gain']].sort_values(by='closing_date')
    last_ten = closing_date_sorted.tail(10)
    last_ten_pos = last_ten[last_ten['gain'] >= 0].shape[0]
    last_ten_neg = last_ten[last_ten['gain'] < 0].shape[0]

    return  f'{reward_risk} : 1', \
            f'{pct_winning_trades}% / {pct_losing_trades}%', \
            f'{expectancy:0.2f}%', \
            f'{R:0.2f}%', \
            f'{last_ten_pos} : {last_ten_neg}'

# -----------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)