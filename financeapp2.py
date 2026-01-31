import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import jarque_bera, norm
from datetime import datetime,timedelta


#garch mspe's
#factor models, plot coefficients over time


st.set_page_config(page_title="Finance Dashboard", layout="wide")

def timedeltafunction(time, days,operator_string):
    duration = timedelta(days=days)
    if operator_string == "+":
        return time + duration
    else:
        return time - duration
    
@st.cache_data
def gatheringdata(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.ffill()
    
    monthly_prices = prices.resample("ME").last()
    monthly_prices.index = monthly_prices.index.to_period("M")
    monthly_simple_returns = monthly_prices.pct_change(fill_method=None).dropna()
    
    return prices, prices.pct_change(fill_method=None).dropna(), monthly_prices, monthly_simple_returns

@st.cache_data
def gatheringff3_m():
    ff3_m = web.DataReader("F-F_Research_Data_Factors", "famafrench")[0]
    ff3_m = ff3_m / 100
    return ff3_m

def runningfamafrenchregression(monthly_returns, fama_df):
    df = monthly_returns.join(fama_df, how="inner")
    X = df[["Mkt-RF", "SMB", "HML"]]
    X = sm.add_constant(X)
    
    results = {}
    for t in monthly_returns.columns:
        y = df[t] - df["RF"]
        model = sm.OLS(y, X).fit()
        results[t] = model.params
    
    return pd.DataFrame(results).T

def printingstatistics(df):
    mean = df.mean()
    std = df.std()
    skew = df.skew()
    kurt = df.kurtosis() + 3
    sharpe = mean/std
    n = df.shape[0]
    jb = n/6 * skew**2 + n/24 *(kurt-3)**2

    stats = pd.DataFrame({
        "Mean (%)": mean*100, "Std (%)": std*100, "Sharpe": sharpe, 
        "Skew": skew, "Kurt": kurt, "JB": jb
    })
    st.dataframe(stats)

def plotcorrelations(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

def extended_GARCH(p, o, q, returns, string):
    garch_model = arch_model(returns, vol='Garch', p=p, o=o, q=q, dist='normal', mean='Constant')
    fit = garch_model.fit(disp='off')


    garch_var = fit.conditional_volatility**2
    

    omega = fit.params['omega']
    alpha_sum = fit.params[fit.params.index.str.contains('alpha')].sum()
    beta_sum = fit.params[fit.params.index.str.contains('beta')].sum()
    
    denom = (1 - alpha_sum - beta_sum)
    long_run_var = omega / denom if denom > 0 else np.nan


    std_resid = fit.std_resid
    
    data_dict = {
        "mu": fit.params['mu'],
        "unconditional variance": long_run_var,
        "mean (std resid)": std_resid.mean(),
        "std (std resid)": std_resid.std(),
        "skew (std resid)": std_resid.skew(),
        "kurt (std resid)": std_resid.kurt()+3,
        "JB (resid)": jarque_bera(std_resid)[0]
    }
    
    for key, value in fit.params.items():
        if key != 'mu': data_dict[key] = value

    st.write(f"**GARCH Parameters ({string})**")
    st.dataframe(pd.DataFrame(data_dict, index=[string]))
    
    return garch_var

def extended_GARCHforecasting(dailygain,start,end,p,o,q,delta,string):
    end1= timedeltafunction(end,delta,"-")
   

    test_dates = pd.date_range(start=end1, end=end, freq='B')
    valid_dates = dailygain.index.intersection(test_dates)
    forecasts = []
    forecasts_indices = []

    for current_date, next_date in zip(valid_dates, valid_dates[1:]):
        subset = dailygain[start:current_date]
        garch_model = arch_model(subset, vol='Garch', p=p, o=o, q=q, dist='normal', mean='Constant')

        res = garch_model.fit(disp='off')
        prediction = res.forecast(horizon=1)
        forecasts.append(prediction.variance.iloc[-1,0])
        forecasts_indices.append(next_date)

    forecasts_df = pd.DataFrame({"Predictions": forecasts}, index= forecasts_indices)
    targets1 = dailygain.loc[forecasts_df.index]**2

    fig, ax = plt.subplots(figsize=(10, 5))
    
 
    ax.plot(targets1.index, targets1, color='grey', alpha=0.3, label='Actual Squared Returns')
    
    
    ax.plot(forecasts_df.index, forecasts_df['Predictions'], color='red', linewidth=1.5, label='Variance Forecast')

    ax.set_title(f"Rolling Forecast vs Realized Variance: {string}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Variance")

    MSPE = ((forecasts_df["Predictions"] - targets1[string])**2).mean()
    evaluation_df = pd.DataFrame({
        "MSPE": MSPE
    }, index = [string])
    return evaluation_df, fig
    




def extendedAnalysis(string, start,end, p, o, q):
    prices= yf.download(string, start=start,end=end, auto_adjust=True)["Close"]
    col = prices.pct_change().dropna()
    mu, std = norm.fit(col)

    statistics_df =pd.DataFrame({
        "Mean (%)": mu*100,
        "Std dev (%)": std*100,
        "Sharpe ratio": mu/std,
        "Skewness": (col*100).skew(),
        "Kurtosis": (col*100).kurt()+3,
        "Jarque-Bera": jarque_bera(col)[0]}, index=[string])
    
    st.write("### Asset summary")
    st.dataframe(statistics_df.style.format("{:.4f}"))  

    c1,c2 = st.columns(2)
    with c1:
        st.write("**Returns distribution**")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.hist(col, bins=int(np.floor(np.sqrt(len(col)))), density=True, 
                color='skyblue', edgecolor='black', alpha=0.7)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p_pdf = norm.pdf(x, mu, std)
        ax.plot(x, p_pdf, 'k--', linewidth=2, label=fr'$\mu$={mu:.4f}, $\sigma$={std:.4f}')
        ax.legend()
        ax.set_xlabel("Return")
        st.pyplot(fig)
    
    with c2:
        st.write("**Price History**")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(prices.index,prices,color='orange')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    st.divider()
    c3,c4 = st.columns(2)
    with c3:
        st.write("**ACF: Returns**")
        fig,ax = plt.subplots(figsize= (6,4))
        plot_acf(col,lags =50, ax=ax, title=None)
        st.pyplot(fig)
    
    with c4:
        st.write("**ACF: Squared Returns**")
        fig,ax = plt.subplots(figsize=(6,4))
        plot_acf(col**2, lags = 50, ax=ax, title=None)
        st.pyplot(fig)

    st.divider()
    st.write("### Realized variance 5 minutes (Only last 60 days :/ )")

    try:
        intra_data = yf.download(
            string, period="60d", interval ="5m", auto_adjust=True
        )
        if isinstance(intra_data, pd.DataFrame):
            if 'Close' in intra_data.columns: intra = intra_data['Close']
            else: intra = intra_data.iloc[:,0]
        else: intra = intra_data
        intra = intra.squeeze().ffill()
     
        intra_ret = intra.pct_change().dropna()

    
        intra_sq = intra_ret ** 2

       
        rv = intra_sq.groupby(intra_sq.index.date).sum()
        
        rv.index = pd.to_datetime(rv.index)

        fig,ax = plt.subplots(figsize=(10,4))
        ax.plot(rv.index, rv, 'purple', linewidth=0.5, markersize=4, linestyle='-')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Intraday data unavailable: {e}")

    st.divider()

    tab_roll, tab_garch = st.tabs(["Rolling Volatility", "GARCH Model"])

    with tab_roll:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, color in zip((63, 126, 252), ['green', 'blue', 'red']):
            hv = col.rolling(window=i).var()
            ax.plot(hv.index, hv, color=color, label=f'{i}d Var')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab_garch:
        try:
            garch_var = extended_GARCH(p, o, q, col*100, string)
        
            fig, ax = plt.subplots(figsize=(10, 4))
            sq_ret = (col * 100) ** 2
            ax.plot(sq_ret.index, sq_ret, color='grey', alpha=0.3, lw=0.5, label='Squared Returns')
            ax.plot(garch_var.index, garch_var, color='red', lw=1.5, label='GARCH Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"GARCH Model Failed: {e}")
        st.divider()
        try:
            st.subheader("Rolling forecasts (last 30 days)")
            eval_df, figure = extended_GARCHforecasting(col*100,start,end,p,o,q,30,string)
            st.pyplot(figure)

            with st.expander("Evaluation matrix"):
                st.dataframe(eval_df)
        except Exception as e:
            st.error(f"GARCH forecasts Failed: {e}")
            
def main():
    st.sidebar.title("Configuration")


    if 'tickers' not in st.session_state:
        st.session_state['tickers'] = ["ABN.AS", "IGLN.L", "ISLN.L", "VUSA.AS", "VWCE.DE","EUNK.DE"]


    st.sidebar.subheader("Manage Tickers")

    with st.sidebar.form(key='add_ticker_form', clear_on_submit=True):
        new_ticker_input = st.text_input("Add Ticker", placeholder="e.g. MSFT")
        
        submit_button = st.form_submit_button(label='Add Ticker')
        
        if submit_button:
            new_ticker = new_ticker_input.upper().strip()
            
            if new_ticker and new_ticker not in st.session_state['tickers']:
                st.session_state['tickers'].append(new_ticker)
                st.success(f"Added {new_ticker}!") # Shows inside the form
            elif new_ticker in st.session_state['tickers']:
                st.warning("Ticker already exists!")
            else:
                st.warning("Please enter a ticker name.")
    
    ticker_to_delete = st.sidebar.selectbox("Select to Delete", st.session_state['tickers'])
    if st.sidebar.button("Delete Ticker"):
        if len(st.session_state['tickers']) > 1:
            st.session_state['tickers'].remove(ticker_to_delete)
            st.sidebar.error(f"Deleted {ticker_to_delete}")
            st.rerun() 
        else:
            st.sidebar.error("You must keep at least one ticker!")


    st.sidebar.markdown("---")
    st.sidebar.write("**Current Portfolio:**")
    
    st.sidebar.multiselect("Current Portfolio:", st.session_state['tickers'], default=st.session_state['tickers'], disabled=True)


    start_date = st.sidebar.date_input("Start", value=pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End", value=pd.to_datetime("today"))


    if st.sidebar.button("Fetch Data"):
        current_tickers = st.session_state['tickers']
        prices, daily, monthly, monthly_g = gatheringdata(current_tickers, start_date, end_date)
        
        st.session_state['data'] = (prices, daily, monthly, monthly_g)
        st.session_state['fetched_tickers'] = current_tickers 
        st.sidebar.success("Data Updated!")

    if 'data' in st.session_state:
        prices, dailygain, monthly, monthlygain = st.session_state['data']
        valid_tickers = st.session_state.get('fetched_tickers', st.session_state['tickers'])
        
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Correlations", "Fama-French", "Deep Dive"])

        with tab1:
            st.write("### Basic Statistics")
            printingstatistics(dailygain)
           

        with tab2:
            st.write("### Correlations")
            col1, col2 = st.columns([1, 1]) 
            
            with col1:
                st.subheader("Correlation Matrix (daily returns)")
                plotcorrelations(dailygain)
            with col2: 
                st.subheader("Correlation Matrix (monthly returns)")
                plotcorrelations(monthlygain)

        with tab3:
            st.write("### Fama French 3-Factor Model")
            st.write("This model is based on the monthly returns :/")
            try:
                ff3 = gatheringff3_m()
                res = runningfamafrenchregression(monthlygain, ff3)
                st.dataframe(res.style.background_gradient(cmap="RdBu", axis=0))
            except Exception as e:
                st.error(f"FF3 Error (Check dates): {e}")

        with tab4:
            st.write("### Extended Analysis")
            target = st.selectbox("Select Asset", valid_tickers)
            
            c1, c2, c3 = st.columns(3)
            p = c1.number_input("P (Lag)", 1, 5, 1)
            q = c2.number_input("Q (Shock)", 1, 5, 1)
            type_g = c3.selectbox("Type", ["GARCH", "TGARCH"])
            o = 1 if type_g == "TGARCH" else 0
            
            if st.button("Run Deep Dive Analysis"):
                extendedAnalysis(target,start_date,end_date, p, o, q)

if __name__ == "__main__":
    main()

