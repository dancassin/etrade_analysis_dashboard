from numpy.lib.function_base import median
import pandas as pd
import numpy as np
import datetime

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from pandas.io.formats import style

import plotly.express as px


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])


# -----------------------------------------------------------
# Constants
today = pd.to_datetime(datetime.date.today())

year = today.year

ytd = pd.to_datetime(f'01-01-{year}')

t_minus_30 = today - datetime.timedelta(30)

t_minus_90 = today - datetime.timedelta(90)

ttm = today - datetime.timedelta(365)

time_periods = {1:ytd, 2:ttm, 3:t_minus_90, 4:t_minus_30}

# -----------------------------------------------------------
# Data Processing

FILEPATH = './data/TTM_tax_lots_090321.csv'

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

tax_lot_df['category'] = pd.cut(tax_lot_df.gain, bins=[-5000, .0001, 5000],
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
                1 : 'YTD',
                2 : 'TTM',
                3 : '90D',
                4 : '30D',
            },
            value=1,
        ),
        
    ]),
    # html.Div(id='slider-drag-output'),
    html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id='gain_hist'),
                        width={'size':9},
                    ),
                    dbc.Col(    
                        [   
                            html.Br(),
                            html.Br(),
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
                                        html.H6("Winning Trades"), #className="card-subtitle"),
                                        html.H4("Title", id='total_winning'),
                                    ]
                                ),
                                style={'text-align':'center'}
                                
                            ),
                            html.Br(),
                            dbc.Card(
                                dbc.CardBody(
                                    [   
                                        html.H6("Losing Trades"), #className="card-subtitle"),
                                        html.H4("Title", id='total_losing'),
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
                  

    return fig

@app.callback([Output('risk_reward_num', 'children'),
            Output('total_winning', 'children'),
            Output('total_losing', 'children')], 
            [Input('sub_df', 'data')])
def update_reward_risk(jsonified_cleaned_data):

    df = pd.read_json(jsonified_cleaned_data)

    winning_trades = df[df['gain']> 0]
    losing_trades = df[df['gain'] < 0]

    total_winning_trades = winning_trades.shape[0]
    total_losing_trades = losing_trades.shape[0]

    avg_gain = round(winning_trades['gain'].median(), 2)
    avg_loss = abs(round(losing_trades['gain'].median(), 2))

    reward_risk = round((total_winning_trades * avg_gain) / (total_losing_trades * avg_loss), 2)
    return f'{reward_risk} : 1', \
        f'{round(total_winning_trades/df.shape[0]*100)}%', \
        f'{round(total_losing_trades/df.shape[0]*100)}%'

# -----------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)