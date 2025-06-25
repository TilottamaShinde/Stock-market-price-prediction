from prophet import Prophet
from statsmodels.tsa.vector_ar.var_model import forecast


def train_prophet(df, periods = 30):
    df = df[["Date", "Close"]].rename(columns={"Date":"ds", "Close":"y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(periods)
