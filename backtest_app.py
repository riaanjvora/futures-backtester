import streamlit as st
import streamlit.components.v1 as components

from backtest_engine import (
    ContractSpec,
    Backtester,
    MovingAverageCross_SMA_EntryOnly,
    EMACross_EntryOnly,
    RSIReversion_EntryOnly,
    BollingerReversion_EntryOnly,
    DonchianBreakout_EntryOnly,
    VWAPReversion_EntryOnly,
    ATRBreakout_EntryOnly,
    TrendFilter_RSI50_EntryOnly,
    load_tradingview_csv_from_bytes,
    compute_metrics,
)

st.set_page_config(page_title="Futures App (TradingView + Backtester)", layout="wide")

st.title("Futures App (TradingView + Backtester)")

st.info(
    "Directions:\n"
    "1) Go to **TradingView** tab for the live chart.\n"
    "2) Export data from TradingView: Chart → (⋯) or Export → CSV.\n"
    "3) Go to **Backtest** tab → upload CSV → choose strategy + risk (ticks) → Run.\n"
    "4) Every trade uses **2:1 Risk:Reward** (TP = 2×SL)."
)

tab_tv, tab_bt = st.tabs(["TradingView", "Backtest"])


# =========================
# TAB 1: TradingView
# =========================
with tab_tv:
    st.subheader("Live Chart (TradingView)")

    colA, colB, colC = st.columns([1.2, 0.8, 0.8])

    tv_symbol = colA.selectbox(
        "Chart Symbol",
        ["CME_MINI:NQ1!", "CME_MINI:ES1!"],
        index=0
    )

    interval = colB.selectbox("Interval", ["1", "5", "15", "60", "240", "D"], index=0)
    tv_height = colC.slider("Chart Height", min_value=550, max_value=1300, value=900, step=50)

    # Button to open TradingView itself (full drawing tools live on TradingView)
    # Use symbol without exchange prefix to build a reliable URL, but keep both.
    raw_symbol = tv_symbol.replace("CME_MINI:", "")
    st.link_button("Open on TradingView (full tools)", f"https://www.tradingview.com/symbols/{raw_symbol}/")

    tradingview_widget = f"""
    <div class="tradingview-widget-container" style="height:{tv_height}px; width:100%;">
      <div id="tv_chart" style="height:{tv_height}px; width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{tv_symbol}",
          "interval": "{interval}",
          "timezone": "America/New_York",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#131722",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "hide_side_toolbar": false,
          "withdateranges": true,
          "details": true,
          "hotlist": false,
          "calendar": false,
          "studies": [
            "MASimple@tv-basicstudies",
            "VWAP@tv-basicstudies"
          ],
          "container_id": "tv_chart"
        }});
      </script>
    </div>
    """

    # Give extra room so the header/symbol isn't clipped
    components.html(tradingview_widget, height=tv_height + 140)


# =========================
# TAB 2: Backtest
# =========================
with tab_bt:
    st.subheader("Backtest")

    st.write(
        "Backtest notes:\n"
        "- Upload a TradingView CSV (time/open/high/low/close; volume optional).\n"
        "- The engine forces **2:1 Risk:Reward** on every trade using a bracket:\n"
        "  - Stop = `risk_ticks` ticks\n"
        "  - Take Profit = `2 × risk_ticks` ticks\n"
        "- Trades are **entry-only** signals; exits are ONLY stop/TP to keep the 2:1 rule consistent."
    )

    st.sidebar.header("Backtest Inputs")

    uploaded = st.sidebar.file_uploader("Upload TradingView CSV", type=["csv"])

    symbol = st.sidebar.selectbox("Backtest Symbol", ["NQ", "ES"], index=0)

    initial_cash = st.sidebar.number_input(
        "Starting Cash ($)", min_value=1000, max_value=1_000_000, value=50_000, step=1000
    )
    qty = st.sidebar.number_input("Contracts (qty)", min_value=1, max_value=50, value=1, step=1)

    strategy_name = st.sidebar.selectbox(
        "Strategy",
        [
            "SMA Cross (Entry Only)",
            "EMA Cross (Entry Only)",
            "RSI Reversion (Entry Only)",
            "Bollinger Reversion (Entry Only)",
            "Donchian Breakout (Entry Only)",
            "VWAP Reversion (Entry Only)",
            "ATR Breakout (Entry Only)",
            "RSI Trend Filter (Entry Only)",
        ],
        index=0
    )

    st.sidebar.subheader("Risk Management (forced 2:1)")
    risk_ticks = st.sidebar.number_input("Stop size (ticks)", min_value=1.0, max_value=400.0, value=20.0, step=1.0)
    rr = 2.0
    st.sidebar.caption(f"Take-profit is fixed at {rr}:1 (TP = {rr} × stop).")

    st.sidebar.subheader("Costs")
    commission = st.sidebar.number_input("Commission per side ($)", min_value=0.0, max_value=50.0, value=2.0, step=0.25)
    slippage_ticks = st.sidebar.number_input("Slippage (ticks)", min_value=0.0, max_value=10.0, value=1.0, step=0.25)

    st.sidebar.subheader("Strategy Settings")

    # Strategy params
    fast = slow = None
    ema_fast = ema_slow = None
    rsi_period = oversold = overbought = None
    bb_period = bb_k = None
    donchian_lookback = None
    vwap_band = None
    atr_period = atr_mult = None
    trend_rsi_period = trend_upper = trend_lower = None

    if strategy_name == "SMA Cross (Entry Only)":
        fast = st.sidebar.number_input("Fast SMA", min_value=2, max_value=500, value=20, step=1)
        slow = st.sidebar.number_input("Slow SMA", min_value=3, max_value=1000, value=50, step=1)
        if slow <= fast:
            st.sidebar.warning("Slow SMA should be greater than Fast SMA.")

    elif strategy_name == "EMA Cross (Entry Only)":
        ema_fast = st.sidebar.number_input("Fast EMA", min_value=2, max_value=200, value=12, step=1)
        ema_slow = st.sidebar.number_input("Slow EMA", min_value=3, max_value=500, value=26, step=1)
        if ema_slow <= ema_fast:
            st.sidebar.warning("Slow EMA should be greater than Fast EMA.")

    elif strategy_name == "RSI Reversion (Entry Only)":
        rsi_period = st.sidebar.number_input("RSI Period", min_value=2, max_value=200, value=14, step=1)
        oversold = st.sidebar.number_input("Oversold", min_value=1.0, max_value=49.0, value=30.0, step=1.0)
        overbought = st.sidebar.number_input("Overbought", min_value=51.0, max_value=99.0, value=70.0, step=1.0)

    elif strategy_name == "Bollinger Reversion (Entry Only)":
        bb_period = st.sidebar.number_input("BB Period", min_value=5, max_value=300, value=20, step=1)
        bb_k = st.sidebar.number_input("BB K (std dev)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    elif strategy_name == "Donchian Breakout (Entry Only)":
        donchian_lookback = st.sidebar.number_input("Lookback", min_value=5, max_value=500, value=20, step=1)

    elif strategy_name == "VWAP Reversion (Entry Only)":
        vwap_band = st.sidebar.number_input("VWAP Band (points)", min_value=1.0, max_value=200.0, value=10.0, step=1.0)

    elif strategy_name == "ATR Breakout (Entry Only)":
        atr_period = st.sidebar.number_input("ATR Period", min_value=2, max_value=200, value=14, step=1)
        atr_mult = st.sidebar.number_input("ATR Multiplier", min_value=0.25, max_value=5.0, value=1.0, step=0.25)

    elif strategy_name == "RSI Trend Filter (Entry Only)":
        trend_rsi_period = st.sidebar.number_input("RSI Period", min_value=2, max_value=200, value=14, step=1)
        trend_upper = st.sidebar.number_input("Long if RSI >", min_value=50.0, max_value=90.0, value=55.0, step=1.0)
        trend_lower = st.sidebar.number_input("Short if RSI <", min_value=10.0, max_value=50.0, value=45.0, step=1.0)

    run = st.sidebar.button("Run Backtest")

    if uploaded is None:
        st.info("Upload a CSV to begin.")
        st.stop()

    try:
        df = load_tradingview_csv_from_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        st.stop()

    st.subheader("Data Preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df)}")
    c2.metric("Start", str(df.index.min()))
    c3.metric("End", str(df.index.max()))
    c4.metric("RR", "2:1 (fixed)")
    st.dataframe(df.head(50), use_container_width=True)

    if not run:
        st.warning("Set inputs on the left, then click 'Run Backtest'.")
        st.stop()

    # Contract spec
    tick_size = 0.25
    tick_value = 5.0 if symbol == "NQ" else 12.5

    spec = ContractSpec(
        symbol=symbol,
        tick_size=tick_size,
        tick_value=tick_value,
        commission_per_side=float(commission),
        slippage_ticks=float(slippage_ticks),
    )

    bt = Backtester(spec, initial_cash=float(initial_cash))

    # Build strategy
    if strategy_name == "SMA Cross (Entry Only)":
        strat = MovingAverageCross_SMA_EntryOnly(fast=int(fast), slow=int(slow), qty=int(qty))
    elif strategy_name == "EMA Cross (Entry Only)":
        strat = EMACross_EntryOnly(fast=int(ema_fast), slow=int(ema_slow), qty=int(qty))
    elif strategy_name == "RSI Reversion (Entry Only)":
        strat = RSIReversion_EntryOnly(period=int(rsi_period), oversold=float(oversold), overbought=float(overbought), qty=int(qty))
    elif strategy_name == "Bollinger Reversion (Entry Only)":
        strat = BollingerReversion_EntryOnly(period=int(bb_period), k=float(bb_k), qty=int(qty))
    elif strategy_name == "Donchian Breakout (Entry Only)":
        strat = DonchianBreakout_EntryOnly(lookback=int(donchian_lookback), qty=int(qty))
    elif strategy_name == "VWAP Reversion (Entry Only)":
        strat = VWAPReversion_EntryOnly(band_points=float(vwap_band), qty=int(qty))
    elif strategy_name == "ATR Breakout (Entry Only)":
        strat = ATRBreakout_EntryOnly(period=int(atr_period), mult=float(atr_mult), qty=int(qty))
    else:
        strat = TrendFilter_RSI50_EntryOnly(period=int(trend_rsi_period), upper=float(trend_upper), lower=float(trend_lower), qty=int(qty))

    results = bt.run(df, strat, risk_ticks=float(risk_ticks), rr=2.0)
    metrics = compute_metrics(float(initial_cash), results)

    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("End Equity", f"${metrics['end_equity']:.2f}")
    m2.metric("Net Profit", f"${metrics['net_profit']:.2f}")
    m3.metric("Trades", f"{int(metrics['trades'])}")
    m4.metric("Win Rate", f"{metrics['win_rate']:.2f}%")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Max Drawdown", f"${metrics['max_drawdown']:.2f}")
    m6.metric("Profit Factor", f"{metrics['profit_factor']:.3f}")
    m7.metric("Avg Trade", f"${metrics['avg_trade']:.2f}")
    m8.metric("Expectancy", f"${metrics['expectancy']:.2f}")

    st.subheader("Equity Curve")
    st.line_chart(results["equity"][["equity"]], height=320)

    st.subheader("Trades")
    trades = results["trades"]
    if len(trades) == 0:
        st.write("No trades for this strategy on this dataset.")
    else:
        st.dataframe(trades, use_container_width=True)

    st.subheader("Downloads")
    trades_csv = trades.to_csv(index=False).encode("utf-8")
    equity_csv = results["equity"].to_csv().encode("utf-8")

    # Risk summary
    stop_points = float(risk_ticks) * tick_size
    tp_points = float(risk_ticks) * tick_size * 2.0

    summary_text = (
        "--- Summary ---\n"
        f"Strategy: {strategy_name}\n"
        f"Symbol: {symbol}\n"
        f"Qty: {int(qty)}\n"
        f"Commission/side: {float(commission)}\n"
        f"Slippage ticks: {float(slippage_ticks)}\n"
        f"Risk ticks (stop): {float(risk_ticks)}\n"
        f"Stop (points): {stop_points:.2f}\n"
        f"TP (points): {tp_points:.2f}\n"
        f"RR: 2:1 (fixed)\n"
        "\n"
        f"Start Equity: {metrics['start_equity']:.2f}\n"
        f"End Equity: {metrics['end_equity']:.2f}\n"
        f"Net Profit: {metrics['net_profit']:.2f}\n"
        f"Trades: {int(metrics['trades'])}\n"
        f"Win Rate: {metrics['win_rate']:.2f}%\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2f}\n"
        f"Profit Factor: {metrics['profit_factor']:.3f}\n"
        f"Avg Trade: {metrics['avg_trade']:.2f}\n"
    ).encode("utf-8")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download trades.csv", trades_csv, file_name="trades.csv", mime="text/csv")
    d2.download_button("Download equity.csv", equity_csv, file_name="equity.csv", mime="text/csv")
    d3.download_button("Download summary.txt", summary_text, file_name="summary.txt", mime="text/plain")
