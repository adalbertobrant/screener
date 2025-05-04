# -*- coding: utf-8 -*-
"""
Multi-Market Screener for technical signals (MACD, RSI) with on-demand
fundamental data and box plot visualization.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import concurrent.futures
import plotly.graph_objects as go
import math
import requests
from bs4 import BeautifulSoup

# --- Configuration & Constants ---

# Valid periods for yfinance
VALID_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
# Default interval (daily is usually best for these analyses)
DEFAULT_INTERVAL = "1d"

# Default list of assets (Mix of US, BR, Crypto, Forex, Indices) - Approx 35 assets
# User requested ~50, can be expanded later if needed.
DEFAULT_ASSETS = [
    # Crypto (5)
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD",
    # Bovespa (15)
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
    "WEGE3.SA", "MGLU3.SA", "LREN3.SA", "SUZB3.SA", "RENT3.SA",
    "AZUL4.SA", "GMAT3.SA", "EMBR3.SA", "TOTS3.SA", "ITSA4.SA",
    # US Stocks (10)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "BAC", "WFC",
    # Forex (3)
    "BRL=X", # USD/BRL
    "EURUSD=X", # EUR/USD (More common than EUR=X)
    "JPY=X", # USD/JPY
    # Indices (2)
    "^GSPC", "^BVSP"
]
# Notes on user request:
# - vale, vale3 -> Using VALE3.SA (Bovespa)
# - dolar-real -> BRL=X (USD/BRL)
# - dolar-iene -> JPY=X (USD/JPY)
# - dolar-euro -> EURUSD=X (EUR/USD)
# - s&p500 -> ^GSPC
# - mini ibovespa -> Using ^BVSP (main index)
# - azul -> AZUL4.SA
# - grupo matheus -> GMAT3.SA
# - ifcm3 -> Not found as valid ticker
# - bradesco -> BBDC4.SA
# - itau -> ITUB4.SA
# - banco do brasil -> BBAS3.SA
# - embraer -> EMBR3.SA
# - totvs -> TOTS3.SA
# - universo online -> Using PAGS (PagSeguro, listed in US) as a proxy, UOL is not public.

# RSI parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- Data Fetching and Processing ---

# Cache data fetching to avoid re-downloading during the same session
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_asset_data(ticker, period, interval):
    """Fetches historical data for a given asset ticker."""
    try:
        asset = yf.Ticker(ticker)
        # Use auto_adjust=True for adjusted prices (handles splits/dividends)
        data = asset.history(period=period, interval=interval, auto_adjust=True)
        if data.empty:
            return ticker, None, f"No data returned for {ticker}."
        data.index = pd.to_datetime(data.index)
        data.columns = [col.capitalize() for col in data.columns]
        if 'Close' not in data.columns:
             return ticker, None, f"'Close' column missing for {ticker}."
        return ticker, data, None # Return ticker, data, error_message
    except Exception as e:
        return ticker, None, f"Error fetching data for {ticker}: {e}"

def calculate_rsi(series, period=RSI_PERIOD):
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use Exponential Moving Average (EMA) for RSI calculation (common practice)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Cache indicator calculation
@st.cache_data
def calculate_indicators(df, macd_short=12, macd_long=26, macd_signal=9):
    """Calculates MACD and RSI indicators."""
    # Input df is already cached or freshly fetched
    if df is None or df.empty or 'Close' not in df.columns:
        return None

    df_copy = df.copy()

    # MACD
    ema_short = df_copy['Close'].ewm(span=macd_short, adjust=False).mean()
    ema_long = df_copy['Close'].ewm(span=macd_long, adjust=False).mean()
    df_copy['MACD'] = ema_short - ema_long
    df_copy['Signal'] = df_copy['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df_copy['Hist'] = df_copy['MACD'] - df_copy['Signal']
    # MACD Cross Signals (using last two points)
    if len(df_copy) >= 2:
        macd_now = df_copy['MACD'].iloc[-1]
        macd_prev = df_copy['MACD'].iloc[-2]
        signal_now = df_copy['Signal'].iloc[-1]
        signal_prev = df_copy['Signal'].iloc[-2]
        df_copy['MACD_Cross_Bull'] = (macd_prev < signal_prev) & (macd_now > signal_now)
        df_copy['MACD_Cross_Bear'] = (macd_prev > signal_prev) & (macd_now < signal_now)
    else:
        df_copy['MACD_Cross_Bull'] = False
        df_copy['MACD_Cross_Bear'] = False

    # RSI
    df_copy['RSI'] = calculate_rsi(df_copy['Close'], period=RSI_PERIOD)
    if not df_copy['RSI'].empty:
        rsi_last = df_copy['RSI'].iloc[-1]
        df_copy['RSI_Overbought'] = rsi_last > RSI_OVERBOUGHT
        df_copy['RSI_Oversold'] = rsi_last < RSI_OVERSOLD
    else:
        df_copy['RSI_Overbought'] = False
        df_copy['RSI_Oversold'] = False

    return df_copy

# --- Screener Logic ---

def run_screener(data_dict, criteria):
    """Filters assets based on selected technical criteria from the last data point."""
    results = {}
    for ticker, df in data_dict.items():
        if df is None or df.empty or len(df) < 2: # Need at least 2 points for signals
            continue

        # Get the last row which contains the boolean flags
        last_vals = df.iloc[-1]
        match = False

        # Check criteria based on the boolean flags calculated in calculate_indicators
        if criteria.get('macd_bull') and last_vals.get('MACD_Cross_Bull', False):
            match = True
        if criteria.get('macd_bear') and last_vals.get('MACD_Cross_Bear', False):
            match = True
        if criteria.get('rsi_ob') and last_vals.get('RSI_Overbought', False):
            match = True
        if criteria.get('rsi_os') and last_vals.get('RSI_Oversold', False):
            match = True

        if match:
            results[ticker] = {
                'Close': last_vals.get('Close'),
                'RSI': last_vals.get('RSI'),
                'MACD': last_vals.get('MACD'),
                'Signal': last_vals.get('Signal'),
                # Include the flags in the results for clarity
                'MACD_Cross_Bull': last_vals.get('MACD_Cross_Bull'),
                'MACD_Cross_Bear': last_vals.get('MACD_Cross_Bear'),
                'RSI_Overbought': last_vals.get('RSI_Overbought'),
                'RSI_Oversold': last_vals.get('RSI_Oversold')
            }
    return results

# --- Plotting ---
def plot_box_plot(df, ticker):
    """Generates a Box Plot for the asset's Close price."""
    if df is None or df.empty or 'Close' not in df.columns:
        st.warning(f"Dados insuficientes para gerar Box Plot para {ticker}.")
        return None

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df['Close'],
        name=ticker,
        boxpoints='outliers', # Show outliers
        marker_color='rgb(9,56,125)' # Example color
    ))
    fig.update_layout(
        title=f"Box Plot - Preço de Fechamento: {ticker}",
        yaxis_title="Preço",
        margin=dict(l=40, r=40, t=40, b=40) # Adjust margins
    )
    return fig

# --- Fundamental Data ---

# Helper function to parse Fundamentus values
def parse_fundamentus_value(value_str):
    if not isinstance(value_str, str):
        return None
    try:
        # Remove '.', replace ',' with '.', handle '%'
        cleaned_str = value_str.replace(".", "").replace(",", ".")
        if cleaned_str.endswith("%"):
            return float(cleaned_str.strip("%")) / 100.0
        return float(cleaned_str)
    except ValueError:
        return None

# Helper function to format numbers (similar to previous)
def format_value(value, value_type):
    if value is None or value == "N/A" or math.isnan(value):
        return "N/A"
    try:
        if value_type == "percent":
            return f"{value:.2%}"
        elif value_type == "currency": # For large numbers like Market Cap
            if abs(value) >= 1e12:
                return f"{value / 1e12:.2f} T"
            elif abs(value) >= 1e9:
                return f"{value / 1e9:.2f} B"
            elif abs(value) >= 1e6:
                return f"{value / 1e6:.2f} M"
            elif abs(value) >= 1e3:
                return f"{value / 1e3:.2f} K"
            else:
                return f"{value:.2f}"
        else: # Default (like EPS)
            return f"{value:.2f}"
    except (ValueError, TypeError):
        return "N/A"

# Cache fundamental data fetching
@st.cache_data(ttl=86400) # Cache for 1 day
def get_fundamental_data(ticker):
    """Fetches fundamental data (EPS, ROE, Market Cap, EBITDA).
       Uses Fundamentus for .SA tickers, otherwise yfinance.
    """
    fundamentals = {
        "Ticker": ticker,
        "LPA (EPS)": None,
        "ROE": None,
        "Valor de Mercado": None,
        "EBITDA": None, # EBITDA not directly available in Fundamentus main table
        "Fonte": "N/A"
    }

    if ticker.endswith(".SA"):
        ticker_fundamentus = ticker[:-3]
        url = f"https://www.fundamentus.com.br/detalhes.php?papel={ticker_fundamentus}"
        headers = {"User-Agent": "Mozilla/5.0"} # Mimic browser
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Raise error for bad status codes
            soup = BeautifulSoup(response.text, "html.parser")

            # Find table rows containing the data
            # This relies heavily on Fundamentus page structure
            data = {}
            tables = soup.find_all("table", class_="w728") # Find relevant tables
            if len(tables) >= 2:
                # First table often has Market Cap
                rows_t1 = tables[0].find_all("tr")
                for row in rows_t1:
                    cols = row.find_all("td")
                    if len(cols) == 4 and "Valor de mercado" in cols[0].text:
                        data["Valor de Mercado"] = parse_fundamentus_value(cols[1].text.strip())

                # Second table has LPA, ROE
                rows_t2 = tables[1].find_all("tr")
                for row in rows_t2:
                    cols = row.find_all("td")
                    if len(cols) == 4:
                        label1 = cols[0].text.strip()
                        value1 = cols[1].text.strip()
                        label2 = cols[2].text.strip()
                        value2 = cols[3].text.strip()
                        if "LPA" in label2:
                            data["LPA (EPS)"] = parse_fundamentus_value(value2)
                        if "ROE" in label2:
                            data["ROE"] = parse_fundamentus_value(value2)

            # Assign parsed values
            fundamentals["LPA (EPS)"] = data.get("LPA (EPS)")
            fundamentals["ROE"] = data.get("ROE")
            fundamentals["Valor de Mercado"] = data.get("Valor de Mercado")
            # EBITDA is not directly available in these tables
            fundamentals["EBITDA"] = None
            fundamentals["Fonte"] = "Fundamentus.com.br"

            # If any key data point is missing, maybe try yfinance as fallback?
            # For now, just report what was found.
            if fundamentals["LPA (EPS)"] is None and fundamentals["ROE"] is None and fundamentals["Valor de Mercado"] is None:
                 raise ValueError("Failed to parse key indicators from Fundamentus")

        except Exception as e:
            # Fallback to yfinance if Fundamentus fails
            fundamentals["Fonte"] = f"Fundamentus falhou ({e}), tentando yfinance.info"
            try:
                stock_info = yf.Ticker(ticker).info
                if stock_info and "symbol" in stock_info:
                    fundamentals["LPA (EPS)"] = stock_info.get("trailingEps", stock_info.get("forwardEps"))
                    fundamentals["ROE"] = stock_info.get("returnOnEquity")
                    fundamentals["Valor de Mercado"] = stock_info.get("marketCap")
                    fundamentals["EBITDA"] = stock_info.get("ebitda")
                else:
                    fundamentals["Fonte"] = "Fundamentus falhou, yfinance.info sem dados"
            except Exception as e_yf:
                 fundamentals["Fonte"] = f"Fundamentus e yfinance.info falharam ({e_yf})"

    else: # Not a .SA ticker, use yfinance
        try:
            stock_info = yf.Ticker(ticker).info
            if stock_info and "symbol" in stock_info:
                fundamentals["LPA (EPS)"] = stock_info.get("trailingEps", stock_info.get("forwardEps"))
                fundamentals["ROE"] = stock_info.get("returnOnEquity")
                fundamentals["Valor de Mercado"] = stock_info.get("marketCap")
                fundamentals["EBITDA"] = stock_info.get("ebitda")
                fundamentals["Fonte"] = "yfinance.info"
            else:
                 fundamentals["Fonte"] = "yfinance.info (sem dados)"
        except Exception as e:
            fundamentals["Fonte"] = f"yfinance.info falhou ({e})"

    # Format the final numbers
    formatted_fundamentals = fundamentals.copy()
    formatted_fundamentals["LPA (EPS)"] = format_value(fundamentals["LPA (EPS)"], "default")
    formatted_fundamentals["ROE"] = format_value(fundamentals["ROE"], "percent")
    formatted_fundamentals["Valor de Mercado"] = format_value(fundamentals["Valor de Mercado"], "currency")
    formatted_fundamentals["EBITDA"] = format_value(fundamentals["EBITDA"], "currency")

    return formatted_fundamentals

# --- Streamlit App Main Function ---
def main():
    st.set_page_config(layout="wide")
    st.title("Screener Multi-Mercado (Ações, Cripto, Forex)")

    # Initialize session state variables if they don't exist
    if 'screener_results' not in st.session_state:
        st.session_state.screener_results = {}
    if 'fetched_data_with_indicators' not in st.session_state:
        st.session_state.fetched_data_with_indicators = {}
    if 'fetch_errors' not in st.session_state:
        st.session_state.fetch_errors = {}

    st.warning("""
    **ATENÇÃO:** Ferramenta educacional. Informações de yfinance/Fundamentus, sem garantia de precisão.
    Não é recomendação de investimento. Faça sua própria análise.
    """)

    # --- User Inputs ---
    st.sidebar.header("Configurações Gerais")
    period = st.sidebar.selectbox("Período Histórico:", VALID_PERIODS, index=VALID_PERIODS.index("1y"))
    interval = DEFAULT_INTERVAL # Fixed to daily
    st.sidebar.info(f"Intervalo fixado em '{interval}' (diário).")

    st.sidebar.header("Critérios do Screener (Último Dia)")
    use_macd_bull = st.sidebar.checkbox("MACD: Cruzamento Bullish", value=True)
    use_macd_bear = st.sidebar.checkbox("MACD: Cruzamento Bearish", value=False)
    use_rsi_os = st.sidebar.checkbox(f"RSI: Sobrevendido (< {RSI_OVERSOLD})", value=True)
    use_rsi_ob = st.sidebar.checkbox(f"RSI: Sobrecomprado (> {RSI_OVERBOUGHT})", value=False)

    screener_criteria = {
        'macd_bull': use_macd_bull,
        'macd_bear': use_macd_bear,
        'rsi_os': use_rsi_os,
        'rsi_ob': use_rsi_ob
    }

    st.sidebar.header("Lista de Ativos")
    asset_list_option = st.sidebar.radio("Usar Lista:", ("Padrão", "Personalizada"))

    tickers_to_scan = []
    if asset_list_option == "Padrão":
        tickers_to_scan = DEFAULT_ASSETS
        st.sidebar.info(f"Analisando {len(tickers_to_scan)} ativos da lista padrão.")
    else:
        user_tickers = st.sidebar.text_area("Insira os tickers separados por vírgula ou espaço (ex: AAPL, BTC-USD, PETR4.SA)", ", ".join(DEFAULT_ASSETS[:5]))
        if user_tickers:
            tickers_to_scan = [ticker.strip().upper() for ticker in user_tickers.replace(',', ' ').split() if ticker.strip()]
        else:
            tickers_to_scan = [] # Handled by button logic

    # --- Screener Execution ---
    if st.sidebar.button("Executar Screener"):
        if not tickers_to_scan:
            st.warning("Por favor, insira ou selecione uma lista de ativos para analisar.")
            st.stop()
        if not any(screener_criteria.values()):
             st.warning("Por favor, selecione pelo menos um critério para o screener.")
             st.stop()

        st.header("Executando Análise...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_tickers = len(tickers_to_scan)
        processed_count = 0
        # Use session state to store fetched data across runs if needed, but caching is preferred for this structure
        # if 'fetched_data' not in st.session_state:
        #     st.session_state.fetched_data = {}
        fetched_data_with_indicators = {}
        fetch_errors = {}

        # Fetch data and calculate indicators in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit data fetching tasks
            future_to_ticker_fetch = {executor.submit(get_asset_data, ticker, period, interval): ticker for ticker in tickers_to_scan}
            intermediate_data = {}

            # Process fetching results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker_fetch):
                ticker = future_to_ticker_fetch[future]
                try:
                    _, data, error_msg = future.result()
                    if error_msg:
                        fetch_errors[ticker] = error_msg
                    elif data is not None:
                        intermediate_data[ticker] = data # Store raw data for indicator calculation
                    # else: data is None without error_msg (e.g., empty dataframe)

                except Exception as exc:
                    fetch_errors[ticker] = f"Exceção na busca: {exc}"
                finally:
                    # Update progress based on fetching completion for responsiveness
                    processed_count += 1
                    progress = int(50 * processed_count / total_tickers) # Fetching is ~50% of the work
                    progress_bar.progress(progress)
                    status_text.text(f"Buscando dados {processed_count}/{total_tickers}...")

            # Now calculate indicators for successfully fetched data
            processed_count = 0 # Reset for indicator calculation progress
            future_to_ticker_calc = {executor.submit(calculate_indicators, df): ticker for ticker, df in intermediate_data.items()}

            for future in concurrent.futures.as_completed(future_to_ticker_calc):
                ticker = future_to_ticker_calc[future]
                try:
                    df_processed = future.result()
                    if df_processed is not None:
                        fetched_data_with_indicators[ticker] = df_processed
                    else:
                        # This case might happen if data was minimal or had issues
                        if ticker not in fetch_errors:
                             fetch_errors[ticker] = "Erro ao calcular indicadores (dados insuficientes?)."
                except Exception as exc:
                     fetch_errors[ticker] = f"Exceção no cálculo: {exc}"
                finally:
                    processed_count += 1
                    progress = 50 + int(50 * processed_count / len(intermediate_data)) # Calc is other 50%
                    progress_bar.progress(min(progress, 100))
                    status_text.text(f"Calculando indicadores {processed_count}/{len(intermediate_data)}...")

        # Store results in session state to persist across interactions
        st.session_state.fetched_data_with_indicators = fetched_data_with_indicators
        st.session_state.fetch_errors = fetch_errors
        st.session_state.screener_results = run_screener(fetched_data_with_indicators, screener_criteria)

    # --- Display Results (if available in session state) ---
    if st.session_state.fetch_errors:
        with st.expander(f"Erros/Avisos durante a busca/processamento ({len(st.session_state.fetch_errors)} tickers)"):
            for ticker in sorted(st.session_state.fetch_errors.keys()):
                st.error(f"- {ticker}: {st.session_state.fetch_errors[ticker]}")

    if st.session_state.screener_results:
        st.header("Resultados do Screener")
        st.info(f"Encontrados {len(st.session_state.screener_results)} ativos que correspondem aos critérios selecionados no último dia.")

        # Display results in a DataFrame
        results_list = []
        for ticker in sorted(st.session_state.screener_results.keys()):
            data = st.session_state.screener_results[ticker]
            row = {
                "Ticker": ticker,
                "Preço Fech.": data["Close"],
                "RSI": data["RSI"],
                "MACD": data["MACD"],
                "Sinal MACD": data["Signal"],
                "Sinal": []
            }
            if data["MACD_Cross_Bull"]: row["Sinal"].append("MACD Bull Cross")
            if data["MACD_Cross_Bear"]: row["Sinal"].append("MACD Bear Cross")
            if data["RSI_Oversold"]: row["Sinal"].append(f"RSI Oversold (<{RSI_OVERSOLD})")
            if data["RSI_Overbought"]: row["Sinal"].append(f"RSI Overbought (>{RSI_OVERBOUGHT})")
            row["Sinal"] = ", ".join(row["Sinal"])
            results_list.append(row)

        results_df = pd.DataFrame(results_list)
        st.dataframe(results_df.style.format({
            "Preço Fech.": "{:.2f}",
            "RSI": "{:.2f}",
            "MACD": "{:.4f}",
            "Sinal MACD": "{:.4f}"
        }), use_container_width=True)

        # --- On-Demand Details Section ---
        st.markdown("--- ")
        st.subheader("Análise Detalhada")
        ticker_options = ["Selecione um ticker..."] + sorted(st.session_state.screener_results.keys())
        selected_ticker = st.selectbox("Escolha um ticker dos resultados acima para ver detalhes:", options=ticker_options)

        if selected_ticker != "Selecione um ticker...":
            st.markdown(f"#### Detalhes para: {selected_ticker}")
            # Retrieve the full dataframe with history from session state
            selected_df = st.session_state.fetched_data_with_indicators.get(selected_ticker)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Gráfico Box Plot:**")
                if selected_df is not None:
                    box_fig = plot_box_plot(selected_df, selected_ticker)
                    if box_fig:
                        st.plotly_chart(box_fig, use_container_width=True)
                    else:
                        st.warning("Não foi possível gerar o gráfico Box Plot.")
                else:
                    st.warning(f"Dados históricos não encontrados para {selected_ticker}.")

            with col2:
                st.markdown("**Dados Fundamentalistas:**")
                fundamental_res = get_fundamental_data(selected_ticker)
                # Display fundamental data more clearly
                if fundamental_res:
                    st.text(f"Fonte: {fundamental_res.get("Fonte", "N/A")}")
                    # Create a small table/list for fundamentals
                    fund_df = pd.DataFrame([
                        {"Indicador": "LPA (EPS)", "Valor": fundamental_res.get("LPA (EPS)", "N/A")},
                        {"Indicador": "ROE", "Valor": fundamental_res.get("ROE", "N/A")},
                        {"Indicador": "Valor de Mercado", "Valor": fundamental_res.get("Valor de Mercado", "N/A")},
                        {"Indicador": "EBITDA", "Valor": fundamental_res.get("EBITDA", "N/A")},
                    ])
                    st.dataframe(fund_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("Não foi possível obter dados fundamentalistas.")

    # Display message if screener hasn't run or found no results
    elif not st.session_state.screener_results and not st.session_state.fetch_errors:
        # Check if the button was ever clicked in this session
        # This requires tracking the button click state, which adds complexity.
        # A simpler approach is just to show nothing until the button is clicked.
        pass # Or display a message like "Clique em Executar Screener para começar"


if __name__ == "__main__":
    main()

