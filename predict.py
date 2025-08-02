import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st

# 現在の時刻で取得
now = datetime.now()
gold = yf.download("GC=F", period="1d", interval="1h")  # 1時間足データ（過去24時間分）

# 最新の1本のみ取得
latest = gold.tail(1).copy()
latest['Datetime'] = latest.index

# CSVに追記
latest[['Datetime', 'Close']].to_csv("hourly_gold.csv", mode='a', header=not pd.io.common.file_exists("hourly_gold.csv"), index=False)

st.pyplot(gold)

from streamlit_autorefresh import st_autorefresh

# 毎5分でリロード（ミリ秒）
st_autorefresh(interval=5 * 60 * 1000, key="refresh")