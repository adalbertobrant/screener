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
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# --- Configuration & Constants ---

VALID_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
DEFAULT_INTERVAL = "1d"
MAX_FETCH_WORKERS = 15

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- Asset Lists ---

# Top 30 S&P 500 by market cap (Feb 2025, source: slickcharts.com)
SP500_TOP30 = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO",
    "TSLA", "BRK.B", "WMT", "LLY", "JPM", "V", "XOM", "JNJ", "MA",
    "MU", "COST", "ORCL", "ABBV", "HD", "BAC", "PG", "CAT", "CVX",
    "GE", "KO", "AMD", "NFLX",
]

# Top 30 Nasdaq 100 by weight (Feb 2025, source: slickcharts.com)
NASDAQ100_TOP30 = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO",
    "TSLA", "WMT", "ASML", "MU", "COST", "NFLX", "AMD", "PLTR",
    "CSCO", "LRCX", "AMAT", "TMUS", "LIN", "INTC", "PEP", "AMGN",
    "TXN", "KLAC", "GILD", "ISRG", "ADI", "SHOP",
]

# Crypto & Forex & Indices sempre inclusos
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD"]
FOREX  = ["BRL=X", "EURUSD=X", "JPY=X"]
INDICES = ["^GSPC", "^BVSP", "^NDX"]

# B3 — lista abrangente das principais ações negociadas (ações ordinárias e preferenciais mais líquidas)
# Cobre os índices IBOVESPA, IBRX-100, IDIV, SMLL e outros, totalizando ~250 tickers .SA
B3_TICKERS = [
    # Financeiro / Bancos
    "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ITSA4.SA", "SANB11.SA",
    "BRSR6.SA", "BPAC11.SA", "BPAN4.SA", "BMGB4.SA", "ABCB4.SA",
    "PINE4.SA", "RBRB3.SA", "INTER3.SA", "INBR32.SA",
    # Petróleo e Gás
    "PETR4.SA", "PETR3.SA", "RRRP3.SA", "RECV3.SA", "PRIO3.SA",
    "UGPA3.SA", "VBBR3.SA", "CSAN3.SA", "ENAT3.SA",
    # Mineração / Siderurgia / Metais
    "VALE3.SA", "CSNA3.SA", "USIM5.SA", "GGBR4.SA", "GOAU4.SA",
    "FESA4.SA", "CMIN3.SA", "CBAV3.SA",
    # Energia Elétrica
    "ELET3.SA", "ELET6.SA", "CMIG4.SA", "CPFE3.SA", "ENGI11.SA",
    "ENBR3.SA", "EQTL3.SA", "TIET11.SA", "AURE3.SA", "EGIE3.SA",
    "ENEV3.SA", "CESP6.SA", "CPLE6.SA", "LIGT3.SA", "NEOE3.SA",
    "AESB3.SA", "TRPL4.SA", "ISAE4.SA", "ARNA3.SA",
    # Telecomunicações
    "VIVT3.SA", "TIMS3.SA", "OIBR3.SA",
    # Varejo / Consumo
    "MGLU3.SA", "LREN3.SA", "AMER3.SA", "VIIA3.SA", "PETZ3.SA",
    "SOMA3.SA", "ALPA4.SA", "TFCO4.SA", "AMAR3.SA", "CEAB3.SA",
    "MOVI3.SA", "GRND3.SA", "HGTX3.SA", "GUPY3.SA",
    # Bebidas / Alimentos / Agro
    "ABEV3.SA", "JBSS3.SA", "MRFG3.SA", "BEEF3.SA", "BRFS3.SA",
    "SMTO3.SA", "SLCE3.SA", "AGRO3.SA", "CAML3.SA", "TTEN3.SA",
    "SOJA3.SA", "PCAR3.SA", "CRFB3.SA",
    # Saúde
    "RDOR3.SA", "HAPV3.SA", "GNDI3.SA", "DASA3.SA", "RADL3.SA",
    "FLRY3.SA", "QUAL3.SA", "AALR3.SA", "PNVL3.SA", "BLAU3.SA",
    "HYPE3.SA", "PGMN3.SA", "ONCO3.SA", "SIMH3.SA",
    # Construção / Imobiliário
    "CYRELA.SA", "MRVE3.SA", "EVEN3.SA", "EZTC3.SA", "TRIS3.SA",
    "CYRE3.SA", "TEND3.SA", "DIRR3.SA", "PLPL3.SA", "JHSF3.SA",
    "LAVV3.SA", "HBOR3.SA",
    # Logística / Transporte
    "RAIL3.SA", "CCRO3.SA", "ECOR3.SA", "AZUL4.SA", "GOLL4.SA",
    "TGMA3.SA", "PSSA3.SA", "VSPT3.SA",
    # Tecnologia / Software
    "TOTS3.SA", "LWSA3.SA", "DXCO3.SA", "CASH3.SA", "BRTT3.SA",
    "MOSI3.SA", "POSI3.SA", "TOTVS3.SA",
    # Papel e Celulose
    "SUZB3.SA", "KLBN11.SA", "RANI3.SA",
    # Petroquímica / Química
    "BRKM5.SA", "UNIP6.SA",
    # Locação / Serviços
    "RENT3.SA", "MOVI3.SA", "POSI3.SA", "SIMH3.SA", "VAMO3.SA",
    "MLAS3.SA", "LOGG3.SA",
    # Indústria / Bens de Capital
    "WEGE3.SA", "ROMI3.SA", "TUPY3.SA", "FRAS3.SA", "KEPL3.SA",
    "EMBR3.SA", "EMBJ3.SA",
    # Shopping / Real Estate
    "MULT3.SA", "IGTI11.SA", "BRML3.SA", "ALSO3.SA", "ALLOS3.SA",
    # Seguros
    "BBSE3.SA", "PSSA3.SA", "IRBR3.SA",
    # Educação
    "YDUQ3.SA", "COGN3.SA", "ANIM3.SA", "SEER3.SA",
    # Saneamento
    "SBSP3.SA", "CSMG3.SA", "SAPR11.SA", "SULA11.SA",
    # Mineração Diversificada
    "BRAP4.SA", "LEVE3.SA",
    # FIIs mais líquidos (fundos imobiliários)
    "MXRF11.SA", "HGLG11.SA", "XPML11.SA", "KNRI11.SA", "BRCO11.SA",
    "BTLG11.SA", "GGRC11.SA", "CXCE11B.SA", "VISC11.SA", "XPCA11.SA",
    "PVBI11.SA", "HSML11.SA", "HGRE11.SA", "RBRF11.SA", "HFOF11.SA",
    # Grupo Matheus / Outros
    "GMAT3.SA", "VVEO3.SA", "INTB3.SA", "LUPA3.SA", "BLKB3.SA",
]

# Remove duplicatas mantendo ordem
def _dedup(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

# Lista combinada padrão (US + B3 + Crypto + Forex + Índices)
US_COMBINED = _dedup(SP500_TOP30 + NASDAQ100_TOP30)
DEFAULT_ASSETS = _dedup(CRYPTO + US_COMBINED + B3_TICKERS + FOREX + INDICES)


# --- Data Fetching and Processing ---

@st.cache_data(ttl=3600)
def get_asset_data(ticker, period, interval):
    """Fetches historical OHLCV data for a single ticker via yfinance."""
    try:
        data = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=True
        )
        if data.empty:
            return ticker, None, f"Nenhum dado retornado para {ticker}."
        data.index = pd.to_datetime(data.index)
        data.columns = [col.capitalize() for col in data.columns]
        if 'Close' not in data.columns:
            return ticker, None, f"Coluna 'Close' ausente para {ticker}."
        return ticker, data, None
    except Exception as e:
        return ticker, None, f"Erro ao buscar dados para {ticker}: {e}"


def calculate_rsi(series, period=RSI_PERIOD):
    """Calculates RSI using Wilder's EMA method."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_indicators(df, macd_short=12, macd_long=26, macd_signal=9):
    """Calculates MACD and RSI indicators. Not cached — faster than hashing DataFrames."""
    if df is None or df.empty or 'Close' not in df.columns:
        return None

    df_out = df.copy()

    ema_short = df_out['Close'].ewm(span=macd_short, adjust=False).mean()
    ema_long  = df_out['Close'].ewm(span=macd_long,  adjust=False).mean()
    df_out['MACD']   = ema_short - ema_long
    df_out['Signal'] = df_out['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df_out['Hist']   = df_out['MACD'] - df_out['Signal']

    df_out['MACD_Cross_Bull'] = False
    df_out['MACD_Cross_Bear'] = False
    if len(df_out) >= 2:
        macd_now,  signal_now  = df_out['MACD'].iloc[-1], df_out['Signal'].iloc[-1]
        macd_prev, signal_prev = df_out['MACD'].iloc[-2], df_out['Signal'].iloc[-2]
        df_out.iloc[-1, df_out.columns.get_loc('MACD_Cross_Bull')] = bool(
            (macd_prev < signal_prev) and (macd_now > signal_now)
        )
        df_out.iloc[-1, df_out.columns.get_loc('MACD_Cross_Bear')] = bool(
            (macd_prev > signal_prev) and (macd_now < signal_now)
        )

    df_out['RSI'] = calculate_rsi(df_out['Close'], period=RSI_PERIOD)
    df_out['RSI_Overbought'] = False
    df_out['RSI_Oversold']   = False
    if not df_out['RSI'].empty:
        rsi_last = df_out['RSI'].iloc[-1]
        df_out.iloc[-1, df_out.columns.get_loc('RSI_Overbought')] = bool(rsi_last > RSI_OVERBOUGHT)
        df_out.iloc[-1, df_out.columns.get_loc('RSI_Oversold')]   = bool(rsi_last < RSI_OVERSOLD)

    return df_out


# --- Screener Logic ---

def run_screener(data_dict, criteria):
    results = {}
    for ticker, df in data_dict.items():
        if df is None or df.empty or len(df) < 2:
            continue
        last  = df.iloc[-1]
        match = (
            (criteria.get('macd_bull') and bool(last.get('MACD_Cross_Bull', False))) or
            (criteria.get('macd_bear') and bool(last.get('MACD_Cross_Bear', False))) or
            (criteria.get('rsi_ob')   and bool(last.get('RSI_Overbought',   False))) or
            (criteria.get('rsi_os')   and bool(last.get('RSI_Oversold',     False)))
        )
        if match:
            results[ticker] = {
                'Close':           last.get('Close'),
                'RSI':             last.get('RSI'),
                'MACD':            last.get('MACD'),
                'Signal':          last.get('Signal'),
                'MACD_Cross_Bull': bool(last.get('MACD_Cross_Bull', False)),
                'MACD_Cross_Bear': bool(last.get('MACD_Cross_Bear', False)),
                'RSI_Overbought':  bool(last.get('RSI_Overbought',  False)),
                'RSI_Oversold':    bool(last.get('RSI_Oversold',    False)),
            }
    return results


# --- Plotting ---

def plot_box_plot(df, ticker):
    if df is None or df.empty or 'Close' not in df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df['Close'], name=ticker,
        boxpoints='outliers', marker_color='rgb(9,56,125)'
    ))
    fig.update_layout(
        title=f"Box Plot - Preço de Fechamento: {ticker}",
        yaxis_title="Preço",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


# --- Fundamental Data ---

def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_fundamentus_value(value_str):
    if not isinstance(value_str, str):
        return None
    try:
        cleaned = value_str.replace(".", "").replace(",", ".")
        if cleaned.endswith("%"):
            return float(cleaned.strip("%")) / 100.0
        return float(cleaned)
    except ValueError:
        return None


def format_value(value, value_type):
    fval = _safe_float(value)
    if fval is None or math.isnan(fval):
        return "N/A"
    try:
        if value_type == "percent":
            return f"{fval:.2%}"
        elif value_type == "currency":
            if abs(fval) >= 1e12: return f"{fval / 1e12:.2f} T"
            if abs(fval) >= 1e9:  return f"{fval / 1e9:.2f} B"
            if abs(fval) >= 1e6:  return f"{fval / 1e6:.2f} M"
            if abs(fval) >= 1e3:  return f"{fval / 1e3:.2f} K"
            return f"{fval:.2f}"
        else:
            return f"{fval:.2f}"
    except (ValueError, TypeError):
        return "N/A"


@st.cache_data(ttl=86400)
def get_fundamental_data(ticker):
    fundamentals = {
        "Ticker": ticker, "LPA (EPS)": None, "ROE": None,
        "Valor de Mercado": None, "EBITDA": None, "Fonte": "N/A"
    }
    if ticker.endswith(".SA"):
        ticker_fundamentus = ticker[:-3]
        url = f"https://www.fundamentus.com.br/detalhes.php?papel={ticker_fundamentus}"
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            data = {}
            tables = soup.find_all("table", class_="w728")
            if len(tables) >= 2:
                for row in tables[0].find_all("tr"):
                    cols = row.find_all("td")
                    if len(cols) == 4 and "Valor de mercado" in cols[0].text:
                        data["Valor de Mercado"] = parse_fundamentus_value(cols[1].text.strip())
                for row in tables[1].find_all("tr"):
                    cols = row.find_all("td")
                    if len(cols) == 4:
                        if "LPA" in cols[2].text:
                            data["LPA (EPS)"] = parse_fundamentus_value(cols[3].text.strip())
                        if "ROE" in cols[2].text:
                            data["ROE"] = parse_fundamentus_value(cols[3].text.strip())
            fundamentals.update({**data, "Fonte": "Fundamentus.com.br"})
            if all(fundamentals[k] is None for k in ["LPA (EPS)", "ROE", "Valor de Mercado"]):
                raise ValueError("Nenhum indicador encontrado no Fundamentus")
        except Exception as e:
            fundamentals["Fonte"] = f"Fundamentus falhou ({e}), tentando yfinance"
            try:
                info = yf.Ticker(ticker).info
                if info and "symbol" in info:
                    fundamentals.update({
                        "LPA (EPS)": info.get("trailingEps", info.get("forwardEps")),
                        "ROE": info.get("returnOnEquity"),
                        "Valor de Mercado": info.get("marketCap"),
                        "EBITDA": info.get("ebitda"),
                        "Fonte": "yfinance (fallback)"
                    })
            except Exception as e2:
                fundamentals["Fonte"] = f"Fundamentus e yfinance falharam ({e2})"
    else:
        try:
            info = yf.Ticker(ticker).info
            if info and "symbol" in info:
                fundamentals.update({
                    "LPA (EPS)": info.get("trailingEps", info.get("forwardEps")),
                    "ROE": info.get("returnOnEquity"),
                    "Valor de Mercado": info.get("marketCap"),
                    "EBITDA": info.get("ebitda"),
                    "Fonte": "yfinance"
                })
            else:
                fundamentals["Fonte"] = "yfinance (sem dados)"
        except Exception as e:
            fundamentals["Fonte"] = f"yfinance falhou ({e})"

    out = fundamentals.copy()
    out["LPA (EPS)"]        = format_value(fundamentals["LPA (EPS)"],        "default")
    out["ROE"]              = format_value(fundamentals["ROE"],               "percent")
    out["Valor de Mercado"] = format_value(fundamentals["Valor de Mercado"],  "currency")
    out["EBITDA"]           = format_value(fundamentals["EBITDA"],            "currency")
    return out


# --- Streamlit App ---

def main():
    st.set_page_config(layout="wide", page_title="Screener Multi-Mercado")
    st.title("📊 Screener Multi-Mercado (B3 · S&P 500 · Nasdaq · Cripto · Forex)")

    for key in ['screener_results', 'fetched_data_with_indicators', 'fetch_errors']:
        if key not in st.session_state:
            st.session_state[key] = {}
    if 'screener_ran' not in st.session_state:
        st.session_state.screener_ran = False

    st.warning(
        "**ATENÇÃO:** Ferramenta educacional. Dados via yfinance/Fundamentus sem garantia de precisão. "
        "Não constitui recomendação de investimento."
    )

    # --- Sidebar: Configurações ---
    st.sidebar.header("⚙️ Configurações Gerais")
    period   = st.sidebar.selectbox("Período Histórico:", VALID_PERIODS, index=VALID_PERIODS.index("1y"))
    interval = DEFAULT_INTERVAL
    st.sidebar.info(f"Intervalo fixado em '{interval}' (diário).")

    st.sidebar.header("🔍 Critérios do Screener")
    use_macd_bull = st.sidebar.checkbox("MACD: Cruzamento Bullish",                 value=True)
    use_macd_bear = st.sidebar.checkbox("MACD: Cruzamento Bearish",                 value=False)
    use_rsi_os    = st.sidebar.checkbox(f"RSI: Sobrevendido (< {RSI_OVERSOLD})",    value=True)
    use_rsi_ob    = st.sidebar.checkbox(f"RSI: Sobrecomprado (> {RSI_OVERBOUGHT})", value=False)

    screener_criteria = {
        'macd_bull': use_macd_bull, 'macd_bear': use_macd_bear,
        'rsi_os': use_rsi_os, 'rsi_ob': use_rsi_ob,
    }

    # --- Sidebar: Lista de Ativos ---
    st.sidebar.header("📋 Lista de Ativos")
    asset_list_option = st.sidebar.radio(
        "Universo de ativos:",
        ("🌎 Completo (B3 + S&P500 + Nasdaq + Cripto)",
         "🇧🇷 Apenas B3",
         "🇺🇸 Apenas S&P 500 Top 30",
         "💹 Apenas Nasdaq 100 Top 30",
         "₿ Apenas Cripto",
         "✏️ Personalizada")
    )

    tickers_to_scan = []
    if asset_list_option.startswith("🌎"):
        tickers_to_scan = DEFAULT_ASSETS
        st.sidebar.info(f"{len(tickers_to_scan)} ativos no universo completo.")
    elif asset_list_option.startswith("🇧🇷"):
        tickers_to_scan = _dedup(B3_TICKERS + INDICES[-1:])
        st.sidebar.info(f"{len(tickers_to_scan)} ativos da B3.")
    elif asset_list_option.startswith("🇺🇸"):
        tickers_to_scan = _dedup(SP500_TOP30 + [INDICES[0]])
        st.sidebar.info(f"{len(tickers_to_scan)} ativos S&P 500 Top 30.")
    elif asset_list_option.startswith("💹"):
        tickers_to_scan = _dedup(NASDAQ100_TOP30 + [INDICES[2]])
        st.sidebar.info(f"{len(tickers_to_scan)} ativos Nasdaq 100 Top 30.")
    elif asset_list_option.startswith("₿"):
        tickers_to_scan = CRYPTO
        st.sidebar.info(f"{len(tickers_to_scan)} criptomoedas.")
    else:
        user_tickers = st.sidebar.text_area(
            "Tickers separados por vírgula ou espaço:",
            "PETR4.SA, VALE3.SA, AAPL, BTC-USD"
        )
        if user_tickers:
            tickers_to_scan = [
                t.strip().upper()
                for t in user_tickers.replace(',', ' ').split()
                if t.strip()
            ]

    # --- Execução do Screener ---
    if st.sidebar.button("🚀 Executar Screener", use_container_width=True):
        if not tickers_to_scan:
            st.warning("Por favor, selecione ou insira ativos para analisar.")
            st.stop()
        if not any(screener_criteria.values()):
            st.warning("Por favor, selecione pelo menos um critério.")
            st.stop()

        st.header("⏳ Executando Análise...")
        progress_bar = st.progress(0)
        status_text  = st.empty()
        total        = len(tickers_to_scan)
        fetched_data = {}
        fetch_errors = {}

        ctx = get_script_run_ctx()

        def fetch_with_ctx(ticker):
            add_script_run_ctx(ctx=ctx)
            return get_asset_data(ticker, period, interval)

        # Fase 1: busca paralela
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
            future_map = {executor.submit(fetch_with_ctx, t): t for t in tickers_to_scan}
            done_count = 0
            for future in concurrent.futures.as_completed(future_map):
                ticker = future_map[future]
                done_count += 1
                try:
                    _, data, error_msg = future.result()
                    if error_msg:
                        fetch_errors[ticker] = error_msg
                    elif data is not None:
                        fetched_data[ticker] = data
                except Exception as exc:
                    fetch_errors[ticker] = f"Exceção: {exc}"
                finally:
                    progress_bar.progress(int(50 * done_count / total))
                    status_text.text(f"🔄 Buscando dados: {done_count}/{total} ({ticker})")

        # Fase 2: cálculo de indicadores (main thread, rápido)
        status_text.text("📐 Calculando indicadores...")
        fetched_data_with_indicators = {}
        calc_total = len(fetched_data)
        for i, (ticker, df) in enumerate(fetched_data.items(), 1):
            df_processed = calculate_indicators(df)
            if df_processed is not None:
                fetched_data_with_indicators[ticker] = df_processed
            elif ticker not in fetch_errors:
                fetch_errors[ticker] = "Dados insuficientes para calcular indicadores."
            progress_bar.progress(50 + int(50 * i / calc_total) if calc_total else 100)

        progress_bar.progress(100)
        ok_count  = len(fetched_data_with_indicators)
        err_count = len(fetch_errors)
        status_text.text(f"✅ Concluído! {ok_count} ativos processados, {err_count} erros/avisos.")

        st.session_state.fetched_data_with_indicators = fetched_data_with_indicators
        st.session_state.fetch_errors                 = fetch_errors
        st.session_state.screener_results             = run_screener(
            fetched_data_with_indicators, screener_criteria
        )
        st.session_state.screener_ran = True

    # --- Exibição de Resultados ---
    if st.session_state.fetch_errors:
        with st.expander(f"⚠️ Erros/Avisos ({len(st.session_state.fetch_errors)} tickers — clique para ver)"):
            for ticker in sorted(st.session_state.fetch_errors):
                st.error(f"- **{ticker}**: {st.session_state.fetch_errors[ticker]}")

    if st.session_state.screener_results:
        results = st.session_state.screener_results
        st.header(f"🎯 Resultados do Screener — {len(results)} ativo(s) encontrado(s)")

        results_list = []
        for ticker in sorted(results):
            d = results[ticker]
            sinais = []
            if d["MACD_Cross_Bull"]: sinais.append("📈 MACD Bull Cross")
            if d["MACD_Cross_Bear"]: sinais.append("📉 MACD Bear Cross")
            if d["RSI_Oversold"]:    sinais.append(f"🟢 RSI Sobrevendido (<{RSI_OVERSOLD})")
            if d["RSI_Overbought"]:  sinais.append(f"🔴 RSI Sobrecomprado (>{RSI_OVERBOUGHT})")
            results_list.append({
                "Ticker":      ticker,
                "Preço Fech.": d["Close"],
                "RSI":         d["RSI"],
                "MACD":        d["MACD"],
                "Sinal MACD":  d["Signal"],
                "Sinal(is)":   " | ".join(sinais),
            })

        results_df = pd.DataFrame(results_list)
        st.dataframe(
            results_df.style.format({
                "Preço Fech.": "{:.2f}",
                "RSI":         "{:.2f}",
                "MACD":        "{:.4f}",
                "Sinal MACD":  "{:.4f}",
            }),
            use_container_width=True,
            height=min(400, 36 + 35 * len(results_list))
        )

        # Download CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Baixar resultados em CSV",
            data=csv,
            file_name="screener_resultados.csv",
            mime="text/csv"
        )

        # --- Análise Detalhada ---
        st.markdown("---")
        st.subheader("🔎 Análise Detalhada")
        ticker_options  = ["Selecione um ticker..."] + sorted(results)
        selected_ticker = st.selectbox("Escolha um ticker dos resultados:", options=ticker_options)

        if selected_ticker != "Selecione um ticker...":
            st.markdown(f"#### Detalhes: **{selected_ticker}**")
            selected_df = st.session_state.fetched_data_with_indicators.get(selected_ticker)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📦 Box Plot (Preço de Fechamento)**")
                box_fig = plot_box_plot(selected_df, selected_ticker)
                if box_fig:
                    st.plotly_chart(box_fig, use_container_width=True)
                else:
                    st.warning("Não foi possível gerar o Box Plot.")

            with col2:
                st.markdown("**📑 Dados Fundamentalistas**")
                fund = get_fundamental_data(selected_ticker)
                if fund:
                    st.caption(f"Fonte: {fund.get('Fonte', 'N/A')}")
                    fund_df = pd.DataFrame([
                        {"Indicador": "LPA (EPS)",        "Valor": fund.get("LPA (EPS)", "N/A")},
                        {"Indicador": "ROE",              "Valor": fund.get("ROE", "N/A")},
                        {"Indicador": "Valor de Mercado", "Valor": fund.get("Valor de Mercado", "N/A")},
                        {"Indicador": "EBITDA",           "Valor": fund.get("EBITDA", "N/A")},
                    ])
                    st.dataframe(fund_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("Não foi possível obter dados fundamentalistas.")

    elif st.session_state.screener_ran:
        st.info("🔍 Nenhum ativo correspondeu aos critérios selecionados no último pregão.")
    else:
        st.info("👈 Configure os critérios na barra lateral e clique em **🚀 Executar Screener**.")


if __name__ == "__main__":
    main()
