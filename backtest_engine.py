from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Protocol
from io import BytesIO

import numpy as np
import pandas as pd


# =========================
# Domain models
# =========================

@dataclass
class ContractSpec:
    symbol: str
    tick_size: float
    tick_value: float
    commission_per_side: float
    slippage_ticks: float


@dataclass
class Order:
    side: str        # "BUY" or "SELL"
    qty: int
    type: str = "MKT"


@dataclass
class Fill:
    side: str
    qty: int
    price: float
    commission: float
    timestamp: Any


@dataclass
class Position:
    qty: int = 0      # +long, -short
    avg_price: float = 0.0


@dataclass
class Trade:
    entry_time: Any
    entry_price: float
    exit_time: Any
    exit_price: float
    qty: int
    pnl: float
    exit_reason: str


class Strategy(Protocol):
    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        ...


# =========================
# Indicator helpers
# =========================

def ema_series(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi_last(close: pd.Series, period: int) -> float:
    if period <= 1 or len(close) < period + 1:
        return float("nan")

    diffs = close.diff().iloc[-(period + 1):]
    gains = diffs.clip(lower=0.0).iloc[1:]
    losses = (-diffs.clip(upper=0.0)).iloc[1:]

    avg_gain = gains.mean()
    avg_loss = losses.mean()

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def donchian_high(high: pd.Series, lookback: int) -> float:
    if lookback <= 0 or len(high) < lookback + 1:
        return float("nan")
    return float(high.iloc[-(lookback + 1):-1].max())


def donchian_low(low: pd.Series, lookback: int) -> float:
    if lookback <= 0 or len(low) < lookback + 1:
        return float("nan")
    return float(low.iloc[-(lookback + 1):-1].min())


def vwap_series(df: pd.DataFrame) -> pd.Series:
    price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0).replace(0, 1)
    pv = price * vol
    return pv.cumsum() / vol.cumsum()


def atr_last(df: pd.DataFrame, period: int) -> float:
    if period <= 1 or len(df) < period + 1:
        return float("nan")
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-period:].mean())


# =========================
# Strategies (ENTRY ONLY)
# NOTE: exits are handled by 2:1 bracket (stop + take-profit)
# =========================

class MovingAverageCross_SMA_EntryOnly:
    def __init__(self, fast: int = 20, slow: int = 50, qty: int = 1):
        self.fast = int(fast)
        self.slow = int(slow)
        self.qty = int(qty)

    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        if i < self.slow:
            return None
        if int(state.get("pos_qty", 0)) != 0:
            return None  # already in trade

        close = data["close"]
        fast_ma = close.iloc[i - self.fast + 1:i + 1].mean()
        slow_ma = close.iloc[i - self.slow + 1:i + 1].mean()
        prev_fast = close.iloc[i - self.fast:i].mean()
        prev_slow = close.iloc[i - self.slow:i].mean()

        if prev_fast <= prev_slow and fast_ma > slow_ma:
            return Order(side="BUY", qty=self.qty)
        if prev_fast >= prev_slow and fast_ma < slow_ma:
            return Order(side="SELL", qty=self.qty)
        return None


class EMACross_EntryOnly:
    def __init__(self, fast: int = 12, slow: int = 26, qty: int = 1):
        self.fast = int(fast)
        self.slow = int(slow)
        self.qty = int(qty)

    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        if i < self.slow + 2:
            return None
        if int(state.get("pos_qty", 0)) != 0:
            return None

        close = data["close"].iloc[:i + 1]
        ef = ema_series(close, self.fast)
        es = ema_series(close, self.slow)

        if ef.iloc[-2] <= es.iloc[-2] and ef.iloc[-1] > es.iloc[-1]:
            return Order(side="BUY", qty=self.qty)
        if ef.iloc[-2] >= es.iloc[-2] and ef.iloc[-1] < es.iloc[-1]:
            return Order(side="SELL", qty=self.qty)

        return None


class RSIReversion_EntryOnly:
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70, qty: int = 1):
        self.period = int(period)
        self.oversold = float(oversold)
        self.overbought = float(overbought)
        self.qty = int(qty)

    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        if int(state.get("pos_qty", 0)) != 0:
            return None

        close = data["close"].iloc[:i + 1]
        r = rsi_last(close, self.period)
        if np.isnan(r):
            return None

        if r < self.oversold:
            return Order(side="BUY", qty=self.qty)
        if r > self.overbought:
            return Order(side="SELL", qty=self.qty)
        return None


class BollingerReversion_EntryOnly:
    def __init__(self, period: int = 20, k: float = 2.0, qty: int = 1):
        self.period = int(period)
        self.k = float(k)
        self.qty = int(qty)

    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        if i < self.period + 1:
            return None
        if int(state.get("pos_qty", 0)) != 0:
            return None

        close = data["close"].iloc[:i + 1]
        mid = float(close.iloc[-self.period:].mean())
        sd = float(close.iloc[-self.period:].std(ddof=0))
        upper = mid + self.k * sd
        lower = mid - self.k * sd
        px = float(close.iloc[-1])

        if px < lower:
            return Order(side="BUY", qty=self.qty)
        if px > upper:
            return Order(side="SELL", qty=self.qty)
        return None


class DonchianBreakout_EntryOnly:
    def __init__(self, lookback: int = 20, qty: int = 1):
        self.lookback = int(lookback)
        self.qty = int(qty)

    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        if i < self.lookback + 2:
            return None
        if int(state.get("pos_qty", 0)) != 0:
            return None

        high = data["high"].iloc[:i + 1]
        low = data["low"].iloc[:i + 1]
        close = data["close"].iloc[:i + 1]

        hh = donchian_high(high, self.lookback)
        ll = donchian_low(low, self.lookback)
        px = float(close.iloc[-1])

        if np.isnan(hh) or np.isnan(ll):
            return None

        if px > hh:
            return Order(side="BUY", qty=self.qty)
        if px < ll:
            return Order(side="SELL", qty=self.qty)
        return None


class VWAPReversion_EntryOnly:
    def __init__(self, band_points: float = 10.0, qty: int = 1):
        self.band_points = float(band_points)
        self.qty = int(qty)

    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        if i < 50:
            return None
        if int(state.get("pos_qty", 0)) != 0:
            return None

        sub = data.iloc[:i + 1]
        vwap = float(vwap_series(sub).iloc[-1])
        px = float(sub["close"].iloc[-1])

        if px < vwap - self.band_points:
            return Order(side="BUY", qty=self.qty)
        if px > vwap + self.band_points:
            return Order(side="SELL", qty=self.qty)
        return None


# Extra strategies (added)
class ATRBreakout_EntryOnly:
    """
    Enter when price breaks previous close by ATR * multiplier.
    """
    def __init__(self, period: int = 14, mult: float = 1.0, qty: int = 1):
        self.period = int(period)
        self.mult = float(mult)
        self.qty = int(qty)

    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        if i < self.period + 2:
            return None
        if int(state.get("pos_qty", 0)) != 0:
            return None

        sub = data.iloc[:i + 1]
        atr = atr_last(sub, self.period)
        if np.isnan(atr):
            return None

        px = float(sub["close"].iloc[-1])
        prev = float(sub["close"].iloc[-2])

        up = prev + atr * self.mult
        down = prev - atr * self.mult

        if px > up:
            return Order(side="BUY", qty=self.qty)
        if px < down:
            return Order(side="SELL", qty=self.qty)
        return None


class TrendFilter_RSI50_EntryOnly:
    """
    Simple trend filter:
    - if RSI > 55 => long
    - if RSI < 45 => short
    """
    def __init__(self, period: int = 14, upper: float = 55, lower: float = 45, qty: int = 1):
        self.period = int(period)
        self.upper = float(upper)
        self.lower = float(lower)
        self.qty = int(qty)

    def on_bar(self, i: int, data: pd.DataFrame, state: Dict[str, Any]) -> Optional[Order]:
        if int(state.get("pos_qty", 0)) != 0:
            return None

        close = data["close"].iloc[:i + 1]
        r = rsi_last(close, self.period)
        if np.isnan(r):
            return None

        if r > self.upper:
            return Order(side="BUY", qty=self.qty)
        if r < self.lower:
            return Order(side="SELL", qty=self.qty)
        return None


# =========================
# Execution / Simulation (2:1 bracket exits)
# =========================

class BrokerSimulator:
    def __init__(self, spec: ContractSpec):
        self.spec = spec

    def _apply_slippage(self, side: str, price: float) -> float:
        slip = self.spec.slippage_ticks * self.spec.tick_size
        return price + slip if side == "BUY" else price - slip

    def fill_market(self, order: Order, bar_open_price: float, timestamp) -> Fill:
        px = self._apply_slippage(order.side, float(bar_open_price))
        commission = self.spec.commission_per_side * int(order.qty)
        return Fill(side=order.side, qty=int(order.qty), price=float(px), commission=float(commission), timestamp=timestamp)

    def fill_exit_at_price(self, side: str, qty: int, price: float, timestamp) -> Fill:
        # Side is the action to CLOSE (SELL closes long, BUY closes short)
        px = self._apply_slippage(side, float(price))
        commission = self.spec.commission_per_side * int(qty)
        return Fill(side=side, qty=int(qty), price=float(px), commission=float(commission), timestamp=timestamp)


class Backtester:
    """
    Bar-by-bar simulator:
    - Entries filled at bar OPEN
    - Stops/TP checked intrabar using HIGH/LOW
    - Equity marked at bar CLOSE
    """
    def __init__(self, spec: ContractSpec, initial_cash: float = 50_000.0):
        self.spec = spec
        self.initial_cash = float(initial_cash)

    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        risk_ticks: float = 20.0,      # stop distance in ticks
        rr: float = 2.0               # take-profit = rr * stop
    ) -> Dict[str, Any]:

        broker = BrokerSimulator(self.spec)

        cash = self.initial_cash
        pos = Position()
        equity_curve: List[Dict[str, Any]] = []
        trades: List[Trade] = []

        # bracket levels for current position
        stop_price: Optional[float] = None
        tp_price: Optional[float] = None

        # trade tracking
        open_entry_time = None
        open_entry_price = None
        open_entry_qty = 0

        # pending order to execute at NEXT bar open
        pending_order: Optional[Order] = None

        state: Dict[str, Any] = {}

        # Iterate bars
        for i in range(len(data)):
            ts = data.index[i]
            o = float(data["open"].iloc[i])
            h = float(data["high"].iloc[i])
            l = float(data["low"].iloc[i])
            c = float(data["close"].iloc[i])

            # 1) Execute pending entry at open
            if pending_order is not None and pos.qty == 0:
                fill = broker.fill_market(pending_order, o, ts)
                cash -= fill.commission

                pos.qty = fill.qty if fill.side == "BUY" else -fill.qty
                pos.avg_price = fill.price

                open_entry_time = fill.timestamp
                open_entry_price = fill.price
                open_entry_qty = pos.qty

                # set 2:1 bracket
                stop_dist = float(risk_ticks) * self.spec.tick_size
                tp_dist = float(risk_ticks) * self.spec.tick_size * float(rr)

                if pos.qty > 0:
                    stop_price = pos.avg_price - stop_dist
                    tp_price = pos.avg_price + tp_dist
                else:
                    stop_price = pos.avg_price + stop_dist
                    tp_price = pos.avg_price - tp_dist

                pending_order = None

            # 2) If in position, check bracket intrabar
            exited_this_bar = False
            if pos.qty != 0 and stop_price is not None and tp_price is not None:
                # Determine if stop or tp hit within this bar
                # Worst-case tie-break: if both touched, assume STOP hits first.
                if pos.qty > 0:
                    stop_hit = (l <= stop_price)
                    tp_hit = (h >= tp_price)
                    if stop_hit or tp_hit:
                        if stop_hit:  # worst-case first
                            exit_reason = "STOP"
                            exit_level = stop_price
                        else:
                            exit_reason = "TAKE_PROFIT"
                            exit_level = tp_price

                        exit_side = "SELL"
                        exit_qty = abs(pos.qty)
                        exit_fill = broker.fill_exit_at_price(exit_side, exit_qty, exit_level, ts)
                        cash -= exit_fill.commission

                        ticks = (exit_fill.price - pos.avg_price) / self.spec.tick_size
                        realized = ticks * self.spec.tick_value * pos.qty
                        cash += realized

                        trades.append(
                            Trade(
                                entry_time=open_entry_time,
                                entry_price=float(open_entry_price),
                                exit_time=exit_fill.timestamp,
                                exit_price=float(exit_fill.price),
                                qty=int(open_entry_qty),
                                pnl=float(realized - exit_fill.commission),  # commissions included on exit
                                exit_reason=exit_reason
                            )
                        )

                        pos.qty = 0
                        pos.avg_price = 0.0
                        stop_price = None
                        tp_price = None
                        open_entry_time = None
                        open_entry_price = None
                        open_entry_qty = 0
                        exited_this_bar = True

                else:  # short
                    stop_hit = (h >= stop_price)
                    tp_hit = (l <= tp_price)
                    if stop_hit or tp_hit:
                        if stop_hit:
                            exit_reason = "STOP"
                            exit_level = stop_price
                        else:
                            exit_reason = "TAKE_PROFIT"
                            exit_level = tp_price

                        exit_side = "BUY"
                        exit_qty = abs(pos.qty)
                        exit_fill = broker.fill_exit_at_price(exit_side, exit_qty, exit_level, ts)
                        cash -= exit_fill.commission

                        ticks = (exit_fill.price - pos.avg_price) / self.spec.tick_size
                        realized = ticks * self.spec.tick_value * pos.qty
                        cash += realized

                        trades.append(
                            Trade(
                                entry_time=open_entry_time,
                                entry_price=float(open_entry_price),
                                exit_time=exit_fill.timestamp,
                                exit_price=float(exit_fill.price),
                                qty=int(open_entry_qty),
                                pnl=float(realized - exit_fill.commission),
                                exit_reason=exit_reason
                            )
                        )

                        pos.qty = 0
                        pos.avg_price = 0.0
                        stop_price = None
                        tp_price = None
                        open_entry_time = None
                        open_entry_price = None
                        open_entry_qty = 0
                        exited_this_bar = True

            # 3) Mark-to-market at close
            unrealized = 0.0
            if pos.qty != 0:
                ticks = (c - pos.avg_price) / self.spec.tick_size
                unrealized = ticks * self.spec.tick_value * pos.qty

            equity_now = cash + unrealized
            equity_curve.append({
                "time": ts,
                "equity": equity_now,
                "cash": cash,
                "pos_qty": pos.qty,
                "stop_price": stop_price if stop_price is not None else np.nan,
                "tp_price": tp_price if tp_price is not None else np.nan,
            })

            # 4) Strategy decides for NEXT bar (only if not last bar)
            state["pos_qty"] = pos.qty
            state["avg_price"] = pos.avg_price
            state["cash"] = cash
            state["equity"] = equity_now

            if i < len(data) - 1:
                # Only allow new entry if flat and no pending order
                if pos.qty == 0 and pending_order is None:
                    order = strategy.on_bar(i, data, state)
                    if order is not None and order.qty > 0:
                        pending_order = order

        eq = pd.DataFrame(equity_curve).set_index("time") if len(equity_curve) else pd.DataFrame()
        trade_df = pd.DataFrame([t.__dict__ for t in trades])

        ending_cash = float(eq["cash"].iloc[-1]) if len(eq) else float(cash)
        ending_equity = float(eq["equity"].iloc[-1]) if len(eq) else float(cash)

        return {
            "equity": eq,
            "trades": trade_df,
            "ending_cash": ending_cash,
            "ending_equity": ending_equity,
        }


# =========================
# CSV loading + metrics
# =========================

def load_tradingview_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(file_bytes))
    df.columns = [c.lower().strip() for c in df.columns]

    required = ["time", "open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing '{col}'. Found columns: {list(df.columns)}")

    # parse time
    if np.issubdtype(df["time"].dtype, np.number):
        df["time"] = pd.to_datetime(df["time"], unit="s")
    else:
        df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").set_index("time")

    if "volume" not in df.columns:
        df["volume"] = 1
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1)

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[["open", "high", "low", "close", "volume"]].copy()


def compute_metrics(initial_cash: float, results: Dict[str, Any]) -> Dict[str, float]:
    trades = results["trades"]
    eq = results["equity"]["equity"] if "equity" in results["equity"].columns else pd.Series(dtype=float)

    start_equity = float(initial_cash)
    end_equity = float(results["ending_equity"])
    net_profit = end_equity - start_equity

    if len(eq) > 0:
        peak = eq.cummax()
        drawdown = eq - peak
        max_dd = float(drawdown.min())
    else:
        max_dd = 0.0

    if len(trades) > 0:
        wins = int((trades["pnl"] > 0).sum())
        total = int(len(trades))
        win_rate = (wins / total) * 100.0

        avg_trade = float(trades["pnl"].mean())
        biggest_win = float(trades["pnl"].max())
        biggest_loss = float(trades["pnl"].min())

        gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
        gross_loss = float(trades.loc[trades["pnl"] < 0, "pnl"].sum())

        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
        expectancy = avg_trade
    else:
        win_rate = 0.0
        avg_trade = 0.0
        biggest_win = 0.0
        biggest_loss = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        profit_factor = float("inf")
        expectancy = 0.0

    return {
        "start_equity": start_equity,
        "end_equity": end_equity,
        "net_profit": net_profit,
        "trades": float(len(trades)),
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "biggest_win": biggest_win,
        "biggest_loss": biggest_loss,
        "max_drawdown": max_dd,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }
