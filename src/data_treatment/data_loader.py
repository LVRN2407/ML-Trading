import os

from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)




def get_dataset(token, StartDate=[2020,1,1], EndDate=[2023,1,1]) :
    
  key = os.getenv("ALPACA_API_KEY")
  secret = os.getenv("ALPACA_API_SECRET")

  #if not key or not secret:
      #raise ValueError("Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment or .env.")
    
  #  Initialize the historical data client
  client = StockHistoricalDataClient(key, secret)

  #  Build the request parameters
  y0, m0, d0 = StartDate
  yf, mf, df = EndDate
  request_params = StockBarsRequest(
      symbol_or_symbols=[token],                 # Choose the stocks data to import
      timeframe=TimeFrame.Day,                    # Options: TimeFrame.Minute, TimeFrame.Hour, TimeFrame.Day
      start=datetime(y0, m0, d0),                 # Start date (UTC)
      end=datetime(yf, mf, df)                    # Optional end date (UTC)
  )

  #  Fetch the bars
  bars = client.get_stock_bars(request_params) #

  #  Convert directly to a Pandas DataFrame for ML processing
  DF = bars.df
  return DF


def optimize_memory(df):
  for col in df.select_dtypes(include=['float']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
  for col in df.select_dtypes(include=['integer']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')
  return df