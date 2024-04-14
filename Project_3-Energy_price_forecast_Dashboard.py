
import base64
from dash import Dash, html, dcc, dash, Input, Output
import plotly.express as px
import pandas as pd
import dash
from dash import html, dcc, ctx, callback
import pandas as pd
import plotly.express as px
import pickle
from sklearn import metrics
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
from dash import Output, Input


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Loading the data
raw_df = pd.read_csv('data_to_forecast.csv')

df_data = pd.read_csv('Final_Dataframe.csv')


#DATA PREPARATION

df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data.set_index('Date', inplace=True)
#df['Power-1'] = df['Power_kW'].shift(1)
df_data['Hour'] = df_data.index.hour



#Splitting the dataframe
#df_data = final_df.loc[final_df.index < '2019-01-01']
#df_2019 = final_df.loc[final_df.index >= '2019-01-01']
df_data = df_data.dropna()

dff_price=df_data.iloc[:, [7,2,3,4,5,6,1,8,9]]
dff_price.index = pd.to_datetime(dff_price.index)
dff_price['Year'] = dff_price.index.year
dff_price['Month'] = dff_price.index.month



#Splitting the data.
variables_19 = raw_df.columns[1:]

Date_19 = raw_df.iloc[:, 0]

real_data = raw_df.iloc[:, :2]

data_X_19 = raw_df.iloc[:, 1:]

X2 = raw_df.iloc[:, 0:]
X2['Date'] = pd.to_datetime(X2['Date'])
X2.set_index('Date', inplace=True)
df_2019 = X2.copy()

#X2['Power-1_(kWh)'] = X2['South Tower (kWh)'].shift(1)
#X2 = X2.dropna()

#X2['Hour'] = X2.index.hour
#X2=X2.drop(columns=['windSpeed_m/s','windGust_m/s','pres_mbar', 'rain_mm/h', 'rain_day', 'Hour'])
X2 = X2.iloc[:, [1,2,3,4,5,6,7,8,9]]
df_2019 = X2.copy()

#Y2 real data
y2 = X2.iloc[:, 0]
real_data = y2.reset_index()


X2 = X2.values


# Importing the regression models to be shown.


#1. Linear regression.

with open('LR_model.pkl', 'rb') as file:
    LR_model2 = pickle.load(file)

#Create matrix from data frame
Z=df_data.values
#Identify output Y
Y=Z[:,0]
#Identify input Y
X=Z[:,[1,2,3,4,5,6,7,8,9]]  

y2_pred_LR = LR_model2.predict(X2)
print(y2.size)
print(y2_pred_LR.size)

# Evaluate errors
MAE_LR = metrics.mean_absolute_error(y2, y2_pred_LR)
MBE_LR = np.mean(y2 - y2_pred_LR)
MSE_LR = metrics.mean_squared_error(y2, y2_pred_LR)
RMSE_LR = np.sqrt(metrics.mean_squared_error(y2, y2_pred_LR))
cvRMSE_LR = RMSE_LR / np.mean(y2)
NMBE_LR = MBE_LR / np.mean(y2)




#2. Random forrest Model
with open('RF_model.pkl', 'rb') as file:
    RF_model2 = pickle.load(file)

y2_pred_RF = RF_model2.predict(X2)

# Evaluate errors
MAE_RF = metrics.mean_absolute_error(y2, y2_pred_RF)
MBE_RF = np.mean(y2 - y2_pred_RF)
MSE_RF = metrics.mean_squared_error(y2, y2_pred_RF)
RMSE_RF = np.sqrt(metrics.mean_squared_error(y2, y2_pred_RF))
cvRMSE_RF = RMSE_RF / np.mean(y2)
NMBE_RF = MBE_RF / np.mean(y2)





#3. Decision Tree Model
with open('DT_model.pkl', 'rb') as file:
    DT_regr_model = pickle.load(file)

y2_pred_DT = DT_regr_model.predict(X2)

# Evaluate errors
MAE_DT = metrics.mean_absolute_error(y2, y2_pred_DT)
MBE_DT = np.mean(y2 - y2_pred_DT)
MSE_DT = metrics.mean_squared_error(y2, y2_pred_DT)
RMSE_DT = np.sqrt(metrics.mean_squared_error(y2, y2_pred_DT))
cvRMSE_DT = RMSE_DT / np.mean(y2)
NMBE_DT = MBE_DT / np.mean(y2)



#Putting all together with the error results.

d = {'Methods': ['Linear Regression', 'Random Forest', 'Decision Tree'],
     'MAE': [MAE_LR, MAE_RF, MAE_DT],
     'MBE': [MBE_LR, MBE_RF, MBE_DT],
     'MSE': [MSE_LR, MSE_RF, MSE_DT],
     'RMSE': [RMSE_LR, RMSE_RF, RMSE_DT],
     'cvMSE': [cvRMSE_LR, cvRMSE_RF, cvRMSE_DT],
     'NMBE': [NMBE_LR, NMBE_RF, NMBE_DT]}

df_metrics = pd.DataFrame(data=d)

d = {'Date': real_data['Date'].values,
     'LinearRegression': y2_pred_LR,
     'RandomForest': y2_pred_RF,
     'Decision Tree': y2_pred_DT}

df_forecast = pd.DataFrame(data=d)


df_results = pd.merge(real_data, df_forecast, on='Date')
df = pd.merge(real_data, df_data, on='Date')


fig = px.line(df, x='Date', y=df.columns[1])

fig2 = px.line(df_results, x=df_results.columns[0], y=df_results.columns[1:7])

df_data.reset_index(drop=True, inplace=True)


#Dividing the dashboard into tabs.
def perform_forecast(X_train, Y_train, forecast_method, X_test):

    y_pred = None

    if forecast_method == 'Linear Regression':

        regr = linear_model.LinearRegression()

        regr.fit(X_train, Y_train)

        y_pred = regr.predict(X_test)

    elif forecast_method == 'Decision Tree':

        DT_regr_model = DecisionTreeRegressor()

        DT_regr_model.fit(X_train, Y_train)

        y_pred = DT_regr_model.predict(X_test)

    elif forecast_method == 'Random Forrest':

        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 200,
                      'min_samples_split': 15,
                      'max_features': 'sqrt',
                      'max_depth': 20,
                      'max_leaf_nodes': None}

        RF_model = RandomForestRegressor(**parameters)

        RF_model.fit(X_train, Y_train)
        y_pred = RF_model.predict(X_test)

    return y_pred

def generate_table(dataframe, max_rows=11):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])



#DASHBOARD


app = Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div(children=[
    html.H1(children='ENERGY SERVICES COURSE '),
    html.H2(children='FORECASTING ELECTRICITY PRICE FOR GREECE'),
    html.Div(children='''
              Elaborated by: Julián Gómez, IST - Vasilis Pantelakis, IST - Santiago Valencia, IST1109166
            '''),
    dcc.Tabs([

        
        dcc.Tab(label='Electricity Price Graphs', value='tab-1', children=[
                 html.Div([
                     html.H2(' '),
                     html.H5('Electricity price by Year and by Month (kW)', style={'textAlign': 'center'}), 
                     html.H6('Year'),
                     dcc.Dropdown(
                         id='dropdown',
                         options=[
                             {'label': '2015', 'value': 2015},
                             {'label': '2016', 'value': 2016},
                             {'label': '2017', 'value': 2017},
                             {'label': '2018', 'value': 2018},
                             {'label': '2019', 'value': 2019},
                         ],
                         value= 2015
                     ),
                     dcc.Graph(id='yearly-data')
                 ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),
                 html.Div([
                     html.H6('Month'),
                     dcc.Dropdown(
                         id='dropdown2',
                         options=[
                             {'label': 'January', 'value': 1},
                             {'label': 'February', 'value': 2},
                             {'label': 'March', 'value': 3},
                             {'label': 'April', 'value': 4},
                             {'label': 'May', 'value': 5},
                             {'label': 'June', 'value': 6},
                             {'label': 'July', 'value': 7},
                             {'label': 'August', 'value': 8},
                             {'label': 'September', 'value': 9},
                             {'label': 'October', 'value': 10},
                             {'label': 'November', 'value': 11},
                             {'label': 'December', 'value': 12},
                         ],
                         value= 1
                     ),
                     dcc.Graph(id='monthly-data')
                 ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),
             ]),
        
        

        


        

        
        dcc.Tab(label='Raw variables', children=[
            html.H2(children='Data'),
            html.Label('Raw Variables'),
            dcc.Dropdown(
                id='Raw Variables',
                options=[{'label': i, 'value': i} for i in variables_19],
                value='South Tower (kWh)',
                multi=False
            ),

            html.Div([
                dcc.Graph(id='2019-data')
            ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),
        ], style={'columnCount': 1}),

        dcc.Tab(label='Forecast', children=[
            html.Div([
                html.H4('You may click on the different variables to see their plot'),
                dcc.Graph(
                    id='yearly-data-forecast',
                    figure=fig2,
                ),

            ])

        ]),

        dcc.Tab(label='Metrics', children=[
            html.Div([
                html.H2(children='Metrics'),
                html.H4('IST South Tower Electricity Forecast Error Metrics'),
                generate_table(df_metrics)
            ])
        ]),
    ])
])

             
             
             
'Callbacks "Energy Consumption Graphs Tab'
'Yearly Graph'
@app.callback(Output('yearly-data', 'figure'),
              Input('dropdown', 'value'))
                
def update_graph(value):
    dff = dff_price[dff_price['Year'] == value]
    a = dff.groupby(dff.index)['Electricity price [EUR/MWhe]'].mean()
    return create_graph(a)    

def create_graph(a):
    figure = px.bar(x=a.index, y=a.values, labels={'y':'Power (kW)'},color_discrete_sequence=['blue'])
    return figure



'Monthly Graph'
@app.callback(Output('monthly-data', 'figure'),
              Input('dropdown', 'value'),
              Input('dropdown2', 'value'))

def update_graph2(value, value2):
    dff = dff_price[(dff_price['Year'] == value) & (dff_price['Month'] == value2)]
    a = dff.reset_index().groupby('Date')['Electricity price [EUR/MWhe]'].mean()  # Reset index before grouping
    return create_graph2(a)    

def create_graph2(a):
    figure = px.bar(x=a.index, y=a.values, labels={'y': 'Electricity price [EUR/MWhe]'})
    figure.update_layout(xaxis=dict(tickmode='linear'))
    return figure
                 
             
             
             
             
             
             
             
             
             

#Show raw-data
@app.callback(
    dash.dependencies.Output('2019-data', 'figure'),
    [dash.dependencies.Input('Raw Variables', 'value'),
     ]
)
def Raw_data_graphs (variables_19):
    dff2 = data_X_19[variables_19]
    dff4 = pd.DataFrame(data={'Year': Date_19, 'Demand': dff2})
    fig = px.line(dff4, x='Year', y=[dff2])

    fig.update_layout(yaxis_title='')

    return fig


@app.callback(
    [Output('table-container', 'children'),
     Output('table-info', 'children'),
     Output('figure-shower', 'figure'),
     Output('Metrics','children')],
    [Input('feature-dropdown', 'value'),
     Input('table-shape-dropdown', 'value'),
     Input('Forecast-dropdown', 'value')]
)
def update_table(selected_features, table_shape, forecast_method):
    predictions_df = None

    if not selected_features:
        return [], [], {}, {}

    X_test_index = df_2019[selected_features].index
    X_test = df_2019[selected_features].values

    Y_train = df_data[df_data.columns[0]].dropna().values

    selected_df = df_data[selected_features].copy().dropna()

    if table_shape == 'IQR':
        first_column = df_data.columns[0]
        Q1 = df_data[first_column].quantile(0.25)
        Q3 = df_data[first_column].quantile(0.75)
        IQR = Q3 - Q1

        df_clean_IQR = df_data[((df_data[first_column] > (Q1 - 1.5 * IQR)) & (df_data[first_column] < (Q3 + 1.5 * IQR)))]
        selected_df = df_clean_IQR[selected_features]
        Y_train = df_data[(df_data[first_column] > (Q1 - 1.5 * IQR)) & (df_data[first_column] < (Q3 + 1.5 * IQR))][first_column]

    elif table_shape == 'Z_Score':
        first_column = df_data.columns[0]
        z = np.abs(stats.zscore(df_data[first_column]))
        selected_df['Z_Score'] = z
        selected_df = selected_df[(z < 3)].drop(columns=['Z_Score'])
        Y_train = df_data[(z < 3)][first_column].values

    elif table_shape == 'Interpole':
        first_column = df_data.columns[1]
        df_Interpolate = selected_df.dropna()
        df_Interpolate.index = pd.to_datetime(df_Interpolate.index)
        df_Interpolate[first_column] = df_Interpolate[first_column].mask(df_Interpolate[first_column] <= 100).interpolate(method='time')
        Y_train = df_Interpolate[first_column].values
        selected_df = df_Interpolate

    if forecast_method != 'None':
        predictions_df = perform_forecast(selected_df, Y_train, forecast_method, X_test)

        if predictions_df is not None:
            if not isinstance(predictions_df, pd.DataFrame):
                predictions_df = pd.DataFrame(predictions_df, columns=['Prediction'])

            if not predictions_df.empty:
                predictions_df.index = X_test_index

    table_header = html.Div([
        html.H4('Head of DataFrame'),
        html.Table([
            html.Tr([html.Th(col) for col in selected_df.columns]),
            *[html.Tr([html.Td(selected_df.iloc[i][col]) for col in selected_df.columns]) for i in range(min(5, len(selected_df)))]
        ])
    ])



    table_info = html.Div([
        html.H4('DataFrame Information'),
        html.P(f'Number of Columns: {selected_df.shape[1]}'),
        html.P(f'Number of Rows: {selected_df.shape[0]}'),
        html.Pre(selected_df.info()),
    ])

    if forecast_method != 'None':
        if predictions_df is not None:
            d = {'Date': real_data['Date'].values.ravel(),
                 'Ytest': predictions_df.values.ravel(),
                 'Real_Values_19': y2.values.ravel()}

            fig = px.line(d, x='Date', y=['Ytest', 'Real_Values_19'])
            fig.update_layout(yaxis_title='')

            MAE = metrics.mean_absolute_error(y2, predictions_df)
            MBE = np.mean(y2 - predictions_df)
            MSE = metrics.mean_squared_error(y2, predictions_df)
            RMSE = np.sqrt(metrics.mean_squared_error(y2, predictions_df))
            cvRMSE = RMSE_RF / np.mean(y2)
            NMBE = MBE_RF / np.mean(y2)

            d = {'Methods': forecast_method,
                 'MAE': MAE,
                 'MBE': MBE,
                 'MSE': MSE,
                 'RMSE': RMSE,
                 'cvMSE': cvRMSE,
                 'NMBE': NMBE}

            Metrics = html.Div([
                html.H4('Metrics'),
                html.Table([
                    html.Thead(html.Tr([html.Th(key) for key in d.keys()])),
                    html.Tbody([html.Tr([html.Td(d[key]) for key in d.keys()])])
                ])
            ])
            return table_header, table_info, fig, Metrics

    # Return default table and graph when forecast_method is 'None'
    d = {'Date': real_data['Date'].values.ravel(), 'Real_Values_19': y2.values.ravel()}
    default_fig = px.line(d, x='Date', y='Real_Values_19')
    default_fig.update_layout(yaxis_title='Y2')

    return table_header, table_info, default_fig, {}

if __name__ == '__main__':
    app.run(port=1996)
