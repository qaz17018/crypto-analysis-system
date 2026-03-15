"""
回测模块
用历史数据验证信号引擎的有效性
逻辑：逐根K线回放 → 信号引擎判断 → 模拟开仓 → 统计结果
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """一笔回测交易记录"""

    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: str  # LONG / SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float = 0.0
    exit_reason: str = ""  # SL / TP / TIMEOUT
    pnl_pct: float = 0.0  # 盈亏百分比
    confidence_score: int = 0
    market_state: str = ""
    timeframe_conflicts: int = 0


@dataclass
class BacktestResult:
    """回测统计结果"""

    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    timeout_trades: int = 0

    total_pnl_pct: float = 0.0
    max_win_pct: float = 0.0
    max_loss_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0  # 总盈利 / 总亏损
    avg_rr_ratio: float = 0.0  # 平均实际盈亏比

    max_consecutive_losses: int = 0
    max_drawdown_pct: float = 0.0

    # 按置信度分层的胜率
    score_65_75_winrate: float = 0.0
    score_75_85_winrate: float = 0.0
    score_85_plus_winrate: float = 0.0

    # 按市场状态分层的胜率
    trending_winrate: float = 0.0
    ranging_winrate: float = 0.0

    trades: list = field(default_factory=list)


class Backtester:
    """
    回测器

    不直接调用 SignalEngine（太慢，每根K线都要请求API）
    而是用相同的指标逻辑在本地历史数据上快速回放
    """

    def __init__(
        self,
        client: Client,
        symbol: str = "BTCUSDT",
        min_confidence: int = 65,
        max_hold_candles: int = 20,  # 最多持仓K线数，超时强制平仓
    ):
        self.client = client
        self.symbol = symbol
        self.min_confidence = min_confidence
        self.max_hold_candles = max_hold_candles

    def fetch_history(self, timeframe: str, days: int = 180) -> pd.DataFrame:
        """拉取足够长的历史K线"""
        interval_map = {
            "1d": Client.KLINE_INTERVAL_1DAY,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
        }
        # 计算需要的K线数量
        candles_per_day = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}
        limit = min(days * candles_per_day[timeframe] + 250, 1500)

        raw = self.client.futures_klines(
            symbol=self.symbol,
            interval=interval_map[timeframe],
            limit=limit,
        )
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        df = df.astype(
            {
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            }
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")
        logger.info(f"获取 {timeframe} 历史数据：{len(df)} 根K线")
        return df

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标，和 MarketAnalyzer 保持一致"""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        df["rsi"] = ta.rsi(close, length=14)

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None:
            hist_col = [c for c in macd_df.columns if c.startswith("MACDh")][0]
            df["macd_hist"] = macd_df[hist_col]
        else:
            df["macd_hist"] = 0.0

        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None:
            # 精确匹配 ADX_14，不匹配 ADXR
            adx_col = [
                c
                for c in adx_df.columns
                if c == "ADX_14" or (c.startswith("ADX_") and not c.startswith("ADXR"))
            ][0]
            df["adx"] = adx_df[adx_col]
        else:
            df["adx"] = 20.0

        df["atr"] = ta.atr(high, low, close, length=14)
        df["atr_pct"] = df["atr"] / close

        df["ema200"] = ta.ema(close, length=200)
        df["ema50"] = ta.ema(close, length=50)

        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None:
            lower_col = [c for c in bb_df.columns if c.startswith("BBL")][0]
            upper_col = [c for c in bb_df.columns if c.startswith("BBU")][0]
            bb_range = bb_df[upper_col] - bb_df[lower_col]
            df["bb_position"] = (close - bb_df[lower_col]) / bb_range.replace(0, np.nan)
        else:
            df["bb_position"] = 0.5

        return df

    def _get_signal_at(self, df: pd.DataFrame, i: int) -> tuple:
        """
        在第 i 根K线处计算信号
        返回 (direction, confidence, market_state, stop_loss, take_profit)
        """
        row = df.iloc[i]
        price = row["close"]

        rsi = row["rsi"] if not pd.isna(row["rsi"]) else 50.0
        macd_hist = row["macd_hist"] if not pd.isna(row["macd_hist"]) else 0.0
        adx = row["adx"] if not pd.isna(row["adx"]) else 20.0
        atr_pct = row["atr_pct"] if not pd.isna(row["atr_pct"]) else 0.01
        ema200 = row["ema200"] if not pd.isna(row["ema200"]) else price
        ema50 = row["ema50"] if not pd.isna(row["ema50"]) else price
        bb_pos = row["bb_position"] if not pd.isna(row["bb_position"]) else 0.5

        above_ema200 = price > ema200
        above_ema50 = price > ema50

        # 市场状态
        if atr_pct > 0.03:
            market_state = "高波动危机"
        elif adx > 25:
            market_state = "趋势上涨" if above_ema50 else "趋势下跌"
        else:
            market_state = "震荡盘整"

        # 指标投票（和 MarketAnalyzer 完全一致）
        votes = []
        votes.append(1 if rsi > 60 else (-1 if rsi < 40 else 0))  # RSI  权重0.25
        votes.append(1 if macd_hist > 0 else -1)  # MACD 权重0.30
        votes.append(1 if above_ema200 else -1)  # EMA  权重0.25
        votes.append(-1 if bb_pos > 0.8 else (1 if bb_pos < 0.2 else 0))  # BB 权重0.20

        weights = [0.25, 0.30, 0.25, 0.20]
        score = sum(v * w for v, w in zip(votes, weights))
        normalized = score / sum(weights)

        if normalized > 0.15:
            direction = "LONG"
        elif normalized < -0.15:
            direction = "SHORT"
        else:
            return "NEUTRAL", 0, market_state, 0, 0

        # 置信度（简化版，只用单时间框架）
        strength = abs(normalized)
        base = strength * 50
        bonus = 10 if adx > 25 else 0
        penalty = 15 if market_state == "高波动危机" else 0

        # RSI/MACD 冲突检测
        rsi_sig = 1 if rsi > 55 else (-1 if rsi < 45 else 0)
        macd_sig = 1 if macd_hist > 0 else -1
        conflict_penalty = 8 if (rsi_sig != 0 and rsi_sig != macd_sig) else 0

        confidence = int(min(100, max(0, base + bonus - penalty - conflict_penalty)))

        # 止损止盈：基于 ATR
        atr = row["atr"] if not pd.isna(row["atr"]) else price * 0.01
        if direction == "LONG":
            stop_loss = price - atr * 1.5
            take_profit = price + atr * 3.0
        else:
            stop_loss = price + atr * 1.5
            take_profit = price - atr * 3.0

        return direction, confidence, market_state, stop_loss, take_profit

    def run(self, timeframe: str = "1h", days: int = 180) -> BacktestResult:
        """
        执行回测
        timeframe: 回测的主时间框架
        days: 回测天数
        """
        logger.info(
            f"开始回测 {self.symbol} {timeframe} | {days}天 | 最低置信度:{self.min_confidence}"
        )

        df = self.fetch_history(timeframe, days)
        df = self._calc_indicators(df)

        # 跳过前250根（指标预热期）
        start_idx = 250
        trades = []
        in_position = False
        current_trade: Optional[BacktestTrade] = None
        hold_count = 0

        for i in range(start_idx, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # ── 持仓中：检查止损止盈 ─────────────────────────────────
            if in_position and current_trade:
                hold_count += 1
                high = next_row["high"]
                low = next_row["low"]
                close = next_row["close"]

                hit_sl = hit_tp = False
                if current_trade.direction == "LONG":
                    hit_sl = low <= current_trade.stop_loss
                    hit_tp = high >= current_trade.take_profit
                else:
                    hit_sl = high >= current_trade.stop_loss
                    hit_tp = low <= current_trade.take_profit

                if hit_tp and hit_sl:
                    # 同根K线同时触及，按开盘价判断先触发哪个
                    # 简化处理：假设不利方向先触发（保守估计）
                    hit_sl, hit_tp = True, False

                if hit_sl:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = "SL"
                    current_trade.exit_time = next_row.name
                elif hit_tp:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.exit_reason = "TP"
                    current_trade.exit_time = next_row.name
                elif hold_count >= self.max_hold_candles:
                    current_trade.exit_price = close
                    current_trade.exit_reason = "TIMEOUT"
                    current_trade.exit_time = next_row.name

                if current_trade.exit_price > 0:
                    ep = current_trade.entry_price
                    xp = current_trade.exit_price
                    if current_trade.direction == "LONG":
                        current_trade.pnl_pct = (xp - ep) / ep * 100
                    else:
                        current_trade.pnl_pct = (ep - xp) / ep * 100

                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                    hold_count = 0
                continue

            # ── 空仓：寻找入场信号 ────────────────────────────────────
            direction, confidence, market_state, sl, tp = self._get_signal_at(df, i)

            if direction == "NEUTRAL" or confidence < self.min_confidence:
                continue

            current_trade = BacktestTrade(
                entry_time=row.name,
                exit_time=None,
                direction=direction,
                entry_price=row["close"],
                stop_loss=sl,
                take_profit=tp,
                confidence_score=confidence,
                market_state=market_state,
            )
            in_position = True
            hold_count = 0

        # ── 统计结果 ──────────────────────────────────────────────────
        return self._calc_stats(trades)

    def _calc_stats(self, trades: list) -> BacktestResult:
        """统计回测结果"""
        result = BacktestResult(trades=trades)

        if not trades:
            logger.warning("回测没有产生任何交易，请检查置信度阈值或数据量")
            return result

        result.total_trades = len(trades)
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0 and t.exit_reason == "SL"]
        timeouts = [t for t in trades if t.exit_reason == "TIMEOUT"]

        result.win_trades = len(wins)
        result.loss_trades = len(losses)
        result.timeout_trades = len(timeouts)
        result.win_rate = len(wins) / len(trades) * 100

        result.total_pnl_pct = sum(t.pnl_pct for t in trades)
        result.max_win_pct = max(t.pnl_pct for t in trades)
        result.max_loss_pct = min(t.pnl_pct for t in trades)
        result.avg_win_pct = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
        result.avg_loss_pct = (
            sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
        )

        total_profit = sum(t.pnl_pct for t in wins)
        total_loss = abs(sum(t.pnl_pct for t in losses))
        result.profit_factor = total_profit / total_loss if total_loss > 0 else 999

        result.avg_rr_ratio = (
            abs(result.avg_win_pct / result.avg_loss_pct)
            if result.avg_loss_pct != 0
            else 0
        )

        # 最大连续亏损
        consecutive = max_consecutive = 0
        for t in trades:
            if t.pnl_pct <= 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        result.max_consecutive_losses = max_consecutive

        # 最大回撤
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cumulative += t.pnl_pct
            peak = max(peak, cumulative)
            max_dd = max(max_dd, peak - cumulative)
        result.max_drawdown_pct = max_dd

        # 按置信度分层
        def winrate_for(subset):
            if not subset:
                return 0.0
            return len([t for t in subset if t.pnl_pct > 0]) / len(subset) * 100

        s65 = [t for t in trades if 65 <= t.confidence_score < 75]
        s75 = [t for t in trades if 75 <= t.confidence_score < 85]
        s85 = [t for t in trades if t.confidence_score >= 85]
        result.score_65_75_winrate = winrate_for(s65)
        result.score_75_85_winrate = winrate_for(s75)
        result.score_85_plus_winrate = winrate_for(s85)

        # 按市场状态分层
        trending = [t for t in trades if "趋势" in t.market_state]
        ranging = [t for t in trades if "震荡" in t.market_state]
        result.trending_winrate = winrate_for(trending)
        result.ranging_winrate = winrate_for(ranging)

        return result

    def print_report(self, result: BacktestResult, timeframe: str):
        """打印回测报告"""
        print("\n" + "═" * 60)
        print(f"  回测报告 | {self.symbol} {timeframe}")
        print("═" * 60)

        if result.total_trades == 0:
            print("  没有产生交易，请降低置信度阈值或增加回测天数")
            return

        print(f"\n  基础统计")
        print(f"  {'总交易次数:':<20} {result.total_trades}")
        print(f"  {'盈利次数:':<20} {result.win_trades}")
        print(f"  {'亏损次数:':<20} {result.loss_trades}")
        print(f"  {'超时平仓:':<20} {result.timeout_trades}")
        print(f"  {'胜率:':<20} {result.win_rate:.1f}%")

        print(f"\n  盈亏分析")
        print(f"  {'总盈亏:':<20} {result.total_pnl_pct:+.2f}%")
        print(f"  {'平均盈利:':<20} {result.avg_win_pct:+.2f}%")
        print(f"  {'平均亏损:':<20} {result.avg_loss_pct:+.2f}%")
        print(f"  {'最大单笔盈利:':<20} {result.max_win_pct:+.2f}%")
        print(f"  {'最大单笔亏损:':<20} {result.max_loss_pct:+.2f}%")
        print(f"  {'盈利因子:':<20} {result.profit_factor:.2f}")
        print(f"  {'平均盈亏比:':<20} {result.avg_rr_ratio:.2f}:1")

        print(f"\n  风险指标")
        print(f"  {'最大连续亏损:':<20} {result.max_consecutive_losses} 次")
        print(f"  {'最大回撤:':<20} {result.max_drawdown_pct:.2f}%")

        print(f"\n  置信度分层胜率")
        print(f"  {'65-75分:':<20} {result.score_65_75_winrate:.1f}%")
        print(f"  {'75-85分:':<20} {result.score_75_85_winrate:.1f}%")
        print(f"  {'85分以上:':<20} {result.score_85_plus_winrate:.1f}%")

        print(f"\n  市场状态分层胜率")
        print(f"  {'趋势市:':<20} {result.trending_winrate:.1f}%")
        print(f"  {'震荡市:':<20} {result.ranging_winrate:.1f}%")

        # 综合评估
        print(f"\n  综合评估")
        issues = []
        if result.win_rate < 45:
            issues.append("胜率偏低（<45%），建议提高置信度阈值")
        if result.profit_factor < 1.0:
            issues.append("盈利因子<1，策略整体亏损，需要调整参数")
        if result.avg_rr_ratio < 1.5:
            issues.append("平均盈亏比不足1.5:1，止盈空间可以放大")
        if result.max_consecutive_losses >= 5:
            issues.append(
                f"最大连续亏损{result.max_consecutive_losses}次，注意资金管理"
            )

        if not issues:
            print("  ✓ 策略表现良好，可以考虑在测试网实盘验证")
        else:
            for issue in issues:
                print(f"  ⚠ {issue}")

        print("═" * 60)
