import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as web
from datetime import date, timedelta
import plotly.express as px

yesterday = date.today() - timedelta(days=1)

# 3. 平日インデックスを作成
daily_index = pd.date_range(start='2000-08-30', end=yesterday, freq='B')

# 金価格（基礎トレンド + ノイズ）
gold = yf.download('GC=F', start='2000-08-30', end=yesterday)['Low']

gold_diff = gold.diff().dropna()

# 政策金利（多少の変動を加える）
fr = web.DataReader('FEDFUNDS', 'fred', start='2000-08-30', end=yesterday)

# 2. 月初データにリサンプル（1日だけを使う）
monthly_rate_fr = fr.resample('MS').first()

# 3. 平日インデックスを作成
daily_index = pd.date_range(start='2000-08-30', end=yesterday, freq='B')  # Business days only

# 4. 月次データを平日に再インデックスし、前方補完（ffill）
fr_daily_rate = monthly_rate_fr.reindex(daily_index, method='ffill').dropna()

fr_diff = fr_daily_rate.diff().dropna()

#元データ
# DataFrameにまとめる（差分）
data_original = pd.concat({
    'gold_price': gold,
    'policy_rate': fr_daily_rate
},axis=1)

data1_original = data_original.dropna()

# DataFrameにまとめる（差分）
data_diff = pd.concat({
    'gold_price': gold_diff,
    'policy_rate': fr_diff
},axis=1)

data1_diff = data_diff.dropna(axis=0)

# VARモデル
model = sm.tsa.VAR(data1_diff)
results = model.fit(maxlags=1)

#適切なラグ数
lag_order = results.k_ar

forecast_input = data1_diff.values[-lag_order:]
forecast_steps = 50
forecast = results.forecast(y=forecast_input, steps=forecast_steps)


# 予測結果を元のスケールに戻す
last_values = data1_original.iloc[-1].values
forecast_cumsum = np.cumsum(forecast, axis=0) + last_values

forecast_df = pd.DataFrame(forecast_cumsum, columns=data1_diff.columns,
                           index=pd.date_range(data1_diff.index[-1] + pd.Timedelta(days=1), periods=forecast_steps))

# --- 4. プロット ---
fig = plt.figure(figsize=(12,5))
#plt.plot(data1_original.index, data1_original['gold_price'], label='Gold Price (Observed)')
plt.plot(forecast_df.index, forecast_df['gold_price'], label='Gold Price (Forecast)', linestyle='--')

plt.legend()
plt.title("VAR Forecast of Gold Price(Low)")
plt.tight_layout()
plt.show()

st.pyplot(fig)


import plotly.express as px

fig1 = px.line(
    data1_original,
    x=data1_original.index,
    y=data1_original['gold_price'].squeeze(),  # ← 1次元に変換
    title='Gold Price Prediction(Low)',
    labels={'gold_price': 'Gold Price (Observed)'}
)
fig1.show()

st.plotly_chart(fig1)