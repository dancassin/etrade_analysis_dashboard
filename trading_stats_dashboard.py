import pandas as pd
import numpy as np
import datetime
import json

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from pandas.io.formats import style

import plotly.graph_objs as go
import plotly.express as px




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])


# -----------------------------------------------------------
# Constants
today = pd.to_datetime(datetime.date.today())

year = today.year

ytd = pd.to_datetime(f'01-01-{year}')

t_minus_30 = today - datetime.timedelta(30)

t_minus_90 = today - datetime.timedelta(90)

ttm = today - datetime.timedelta(365)

time_periods = {1:ttm, 2:ytd, 3:t_minus_90, 4:t_minus_30}

# -----------------------------------------------------------
# Data Processing

FILEPATH = './data/TTM_tax_lots_083021.csv'

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

# # -----------------------------------------------------------
# # Layout
app.layout = dbc.Container([
    html.Div([
        html.Br(),
        html.H1('Trading Performance', style={'align':'left'}),
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
                2 : 'YTD',
                3 : '90 Days',
                4 : '30 Days',
            },
            value=2,
        ),
        
    ]),
    #html.Div(id='slider-drag-output'),
    html.Br(),
    html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id='gain_loss_hist'),
                        width={'size':4},
                    ),
                    dbc.Col(
                        dcc.Graph(id='gain_hist'),
                        width={'size':4},
                    ),
                    dbc.Col(
                        dcc.Graph(id='loss_hist'),
                        width={'size':4},
                    ),
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

@app.callback(Output('gain_loss_hist', 'figure'), [Input('sub_df', 'data')])
def gain_loss_hist(jsonified_cleaned_data):

    df = pd.read_json(jsonified_cleaned_data)

    mean_pct = round(df['pct_gain/loss'].mean(), 2)
    median_pct = round(df['pct_gain/loss'].median(), 2)

    # fig = px.histogram(
    #     df, 
    #     x='pct_gain/loss',
    #     nbins=50,
    #     marker_color='black')

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x = df['pct_gain/loss'],
        marker={
            'color':'#D6D6D6', #'#8E9AAF'
        },

    ))
    fig.update_layout(
        xaxis_title_text= f'% Gain & Loss\nMean: {mean_pct}% \n Median: {median_pct}%', # xaxis label
        yaxis_title_text='Total Trades', # yaxis label),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig



# -----------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)