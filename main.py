
from data_loader import load_data
from prophet_model import train_prophet
from arima_model import train_arima

if __name__ == "__main__":
    df = load_data()

    print("\nARIMA forecast : ")
    arima_forecast = train_arima(df)
    print(arima_forecast)

    print("\nProphet forecast : ")
    prophet_forecast = train_prophet(df)
    print(prophet_forecast)
