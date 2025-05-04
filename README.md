<!-- README.md atualizado -->

# 🚀 Screener Multi-Mercado 2.0

Um app em **Streamlit** para screener de Ações, Cripto, Forex e Índices com:
- 📈 Sinais Técnicos: **MACD** & **RSI**
- 📊 Box Plot de preços
- 🧐 Dados Fundamentalistas direto do **Fundamentus**
- 📁 Upload de CSV próprio
- ⚡️ Processamento paralelo e cache

## Funcionalidades

1. **Multi-Mercado**: EUA, Brasil, Crypto, Forex, Índices  
2. **Técnicos**:
   - Bullish & Bearish MACD  
   - Sobrecompra/Sobrevenda RSI  
3. **Detalhamento**:
   - Box Plot  
   - P/L, LPA, ROE, Valor de Mercado (Fundamentus)  
4. **Custom**:
   - Lista de tickers à mão  
   - Upload de seu CSV (intraday ou diário)  
5. **Performance**:
   - Cache 🔄  
   - Paralelismo 🚀  

## Como Executar 🏃

```bash
pip install streamlit pandas numpy plotly yfinance requests
streamlit run screener.py

