"""
单时间框架市场分析器
负责：K线数据获取、技术指标计算、市场状态分类、单框架信号输出
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import pandas_ta as ta
import numpy as np
from binance.client import Client

logger = logging.getLogger(__name__)


class MarketState(Enum):
    TRENDING_UP = "趋势上涨"
    TRENDING_DOWN = "趋势下跌"
    RANGING = "震荡盘整"
    VOLATILE = "高波动危机"


class Signal(Enum):
    LONG = 1
    NEUTRAL = 0
    SHORT = -1


@dataclass
class FrameAnalysis:
    """单个时间框架的完整分析结果"""

    timeframe: str
    market_state: MarketState
    signal: Signal
    strength: float  # 信号强度 0.0 - 1.0

    # 核心指标值
    rsi: float
    macd_hist: float  # MACD 柱状图（正=看多，负=看空）
    adx: float  # 趋势强度（>25 趋势，<20 震荡）
    atr_pct: float  # ATR 占价格百分比（波动率）
    price: float  # 当前收盘价
    above_ema200: bool  # 价格是否在 EMA200 上方
    bb_position: float  # 布林带位置 0=下轨 0.5=中轨 1=上轨

    # 支撑压力
    support: float  # 近期支撑位（止损参考）
    resistance: float  # 近期压力位（止盈参考）

    # 异常标记
    has_spike: bool  # 是否检测到插针
    conflict: bool  # 指标是否存在内部冲突


class MarketAnalyzer:
    """
    单时间框架分析器
    每个时间框架创建一个实例
    """

    # 时间框架映射到币安 interval 常量
    INTERVAL_MAP = {
        "1d": Client.KLINE_INTERVAL_1DAY,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
    }

    def __init__(self, client: Client, symbol: str, timeframe: str):
        if timeframe not in self.INTERVAL_MAP:
            raise ValueError(
                f"不支持的时间框架：{timeframe}，可选：{list(self.INTERVAL_MAP.keys())}"
            )
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe
        self.interval = self.INTERVAL_MAP[timeframe]

    def fetch_klines(self, limit: int = 250) -> pd.DataFrame:
        """
        拉取K线数据，返回干净的 DataFrame
        limit=250 确保 EMA200 有足够的历史数据
        """
        raw = self.client.futures_klines(
            symbol=self.symbol,
            interval=self.interval,
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

        # 过滤插针：影线超过实体3倍的K线，用前后均值修复收盘价
        df = self._filter_spikes(df)
        return df

    def _filter_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        插针过滤：只标记极端异常K线
        条件：影线超过实体5倍，且价格偏离20周期均值超过3个标准差
        """
        body = (df["close"] - df["open"]).abs()
        wick = df["high"] - df["low"]

        # 条件1：影线超过实体5倍（从3倍提高到5倍）
        wick_ratio = (wick > body * 5) & (body > 0)

        # 条件2：收盘价偏离20周期滚动均值超过3个标准差
        roll_mean = df["close"].rolling(20).mean()
        roll_std = df["close"].rolling(20).std()
        z_score = ((df["close"] - roll_mean) / roll_std).abs()
        z_extreme = z_score > 3.0

        # 两个条件同时满足才算插针
        df["is_spike"] = wick_ratio & z_extreme

        spike_count = df["is_spike"].sum()
        if spike_count > 0:
            logger.info(f"{self.timeframe} 检测到 {spike_count} 根插针K线")
        return df

    def _calc_support_resistance(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        用近期高低点计算支撑压力位
        取最近 50 根K线的局部极值
        """
        recent = df.tail(50)
        # 局部最低点（前后各2根更低）
        lows = recent["low"]
        support_candidates = []
        for i in range(2, len(lows) - 2):
            if lows.iloc[i] == lows.iloc[i - 2 : i + 3].min():
                support_candidates.append(lows.iloc[i])

        # 局部最高点
        highs = recent["high"]
        resistance_candidates = []
        for i in range(2, len(highs) - 2):
            if highs.iloc[i] == highs.iloc[i - 2 : i + 3].max():
                resistance_candidates.append(highs.iloc[i])

        current_price = df["close"].iloc[-1]

        # 取当前价格下方最近的支撑，上方最近的压力
        supports_below = [s for s in support_candidates if s < current_price]
        resistance_above = [r for r in resistance_candidates if r > current_price]

        support = max(supports_below) if supports_below else current_price * 0.97
        resistance = min(resistance_above) if resistance_above else current_price * 1.03

        return support, resistance

    def analyze(self) -> Optional[FrameAnalysis]:
        """
        执行完整的单时间框架分析
        返回 FrameAnalysis，失败返回 None
        """
        try:
            df = self.fetch_klines()
        except Exception as e:
            logger.error(f"{self.timeframe} 获取K线失败：{e}")
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ── 计算技术指标 ──────────────────────────────────────────────

        # ── 计算技术指标 ──────────────────────────────────────────────

        # RSI(14) - RSI返回的是Series，没有列名问题，保持原样
        rsi_series = ta.rsi(close, length=14)
        rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else 50.0

        # MACD(12,26,9) - 动态获取 Histogram (柱状图) 列
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            # 匹配以 'MACDh' 开头的列名（忽略后面的参数后缀）
            hist_col = [col for col in macd_df.columns if col.startswith("MACDh")][0]
            macd_hist = float(macd_df[hist_col].iloc[-1])
        else:
            macd_hist = 0.0

        # ADX(14) — 趋势强度 - 动态获取 ADX 主线列
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and not adx_df.empty:
            # 匹配以 'ADX' 开头的列名（区分于 DPM 或 DMN）
            adx_col = [col for col in adx_df.columns if col.startswith("ADX")][0]
            adx = float(adx_df[adx_col].iloc[-1])
        else:
            adx = 20.0

        # ATR(14) — 波动率 - 返回的是Series，保持原样
        atr_series = ta.atr(high, low, close, length=14)
        atr = float(atr_series.iloc[-1]) if atr_series is not None else 0.0
        atr_pct = atr / float(close.iloc[-1])

        # EMA200 — 长期趋势方向 - 返回的是Series，保持原样
        ema200 = ta.ema(close, length=200)
        above_ema200 = (
            float(close.iloc[-1]) > float(ema200.iloc[-1])
            if ema200 is not None
            else True
        )

        # EMA50 — 中期趋势 - 返回的是Series，保持原样
        ema50 = ta.ema(close, length=50)
        above_ema50 = (
            float(close.iloc[-1]) > float(ema50.iloc[-1]) if ema50 is not None else True
        )

        # 布林带(20,2) — 价格相对位置 - 动态获取上下轨
        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            lower_col = [col for col in bb_df.columns if col.startswith("BBL")][0]
            upper_col = [col for col in bb_df.columns if col.startswith("BBU")][0]
            bb_lower = float(bb_df[lower_col].iloc[-1])
            bb_upper = float(bb_df[upper_col].iloc[-1])
            bb_range = bb_upper - bb_lower
            bb_position = (
                (float(close.iloc[-1]) - bb_lower) / bb_range if bb_range > 0 else 0.5
            )
        else:
            bb_position = 0.5

        current_price = float(close.iloc[-1])
        has_spike = bool(df["is_spike"].iloc[-3:].any())

        # ── 市场状态分类 ──────────────────────────────────────────────
        if atr_pct > 0.03:
            market_state = MarketState.VOLATILE
        elif adx > 25:
            market_state = (
                MarketState.TRENDING_UP if above_ema50 else MarketState.TRENDING_DOWN
            )
        else:
            market_state = MarketState.RANGING

        # ── 信号生成 ──────────────────────────────────────────────────
        # 每个指标独立投票，加权求和
        votes = []

        # RSI 投票（震荡市权重更高）
        if rsi > 60:
            votes.append(("rsi", Signal.LONG, 0.25))
        elif rsi < 40:
            votes.append(("rsi", Signal.SHORT, 0.25))
        else:
            votes.append(("rsi", Signal.NEUTRAL, 0.25))

        # MACD 柱状图投票
        if macd_hist > 0:
            votes.append(("macd", Signal.LONG, 0.30))
        elif macd_hist < 0:
            votes.append(("macd", Signal.SHORT, 0.30))
        else:
            votes.append(("macd", Signal.NEUTRAL, 0.30))

        # EMA200 位置投票（趋势市权重更高）
        if above_ema200:
            votes.append(("ema200", Signal.LONG, 0.25))
        else:
            votes.append(("ema200", Signal.SHORT, 0.25))

        # 布林带位置投票
        if bb_position > 0.8:
            votes.append(("bb", Signal.SHORT, 0.20))  # 靠近上轨，超买
        elif bb_position < 0.2:
            votes.append(("bb", Signal.LONG, 0.20))  # 靠近下轨，超卖
        else:
            votes.append(("bb", Signal.NEUTRAL, 0.20))

        # 加权计算总分
        score = sum(v.value * w for _, v, w in votes)
        total_weight = sum(w for _, _, w in votes)
        normalized = score / total_weight  # -1 到 +1

        if normalized > 0.15:
            signal = Signal.LONG
        elif normalized < -0.15:
            signal = Signal.SHORT
        else:
            signal = Signal.NEUTRAL

        strength = min(abs(normalized), 1.0)

        # ── 冲突检测 ──────────────────────────────────────────────────
        # RSI 和 MACD 方向相反 = 内部冲突
        rsi_signal = (
            Signal.LONG if rsi > 55 else (Signal.SHORT if rsi < 45 else Signal.NEUTRAL)
        )
        macd_signal = Signal.LONG if macd_hist > 0 else Signal.SHORT
        conflict = (
            rsi_signal != Signal.NEUTRAL
            and macd_signal != Signal.NEUTRAL
            and rsi_signal != macd_signal
        )

        # ── 支撑压力位 ────────────────────────────────────────────────
        support, resistance = self._calc_support_resistance(df)

        result = FrameAnalysis(
            timeframe=self.timeframe,
            market_state=market_state,
            signal=signal,
            strength=strength,
            rsi=rsi,
            macd_hist=macd_hist,
            adx=adx,
            atr_pct=atr_pct,
            price=current_price,
            above_ema200=above_ema200,
            bb_position=bb_position,
            support=support,
            resistance=resistance,
            has_spike=has_spike,
            conflict=conflict,
        )

        logger.info(
            f"{self.timeframe} 分析完成 | 状态:{market_state.value} | "
            f"信号:{signal.name} | 强度:{strength:.2f} | "
            f"RSI:{rsi:.1f} | ADX:{adx:.1f} | 冲突:{conflict}"
        )
        return result
