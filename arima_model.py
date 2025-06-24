from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def train_arima(df, order = (5, 1, 0), steps = 30):
    df = df.copy()
    df.set_index("Date", inplace = True)
    model = ARIMA(df['Close'], order = order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps = steps)
    return forecast