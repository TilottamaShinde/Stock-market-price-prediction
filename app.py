import dash
from dash import html, dcc
import pandas as pd
import plotly.graph_objs as go
from matplotlib.pyplot import figure

from data_loader import load_data
from prophet_model import train_prophet

df = load_data()
forecast = train_prophet(df)
fig = go.Figure()
fig.add_trace(go.Scatter(x = df["Date"], y = df["Close"], name = "Actual"))
fig.add_trace(go.Scatter(x= forecast["ds"], y = forecast["yhat"], name = "Forecast"))
app = dash.Dash(__name__)
app.title = "Stock Forecast Dashboard"

app.layout = html.Div([
    html.H2("Stock Price Forecast(Prophet)"),
    dcc.Graph(figure = fig)
])

if __name__ == "__main__":
    app.run(debug = True)