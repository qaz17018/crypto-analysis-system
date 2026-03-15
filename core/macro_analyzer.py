"""
宏观市场分析模块
数据来源：yfinance（美股/美元/黄金）+ alternative.me（恐慌贪婪指数）
输出：宏观评分（-100 到 +100）+ 信号方向 + 摘要
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"

logger = logging.getLogger(__name__)


@dataclass
class MacroSignal:
    """单个宏观指标的信号"""

    name: str
    value: float  # 当前值
    change_pct: float  # 日涨跌幅
    signal: int  # +1 看多 / 0 中性 / -1 看空
    weight: float  # 权重
    description: str  # 信号解读


@dataclass
class MacroAnalysis:
    """宏观分析完整结果"""

    score: int  # -100 到 +100，正数对BTC看多
    direction: str  # BULLISH / NEUTRAL / BEARISH
    signals: list[MacroSignal]
    fear_greed_index: int  # 0-100，越高越贪婪
    fear_greed_label: str
    risk_on: bool  # 当前是否风险偏好模式
    summary: str  # 给AI层的摘要
    data_time: str  # 数据时间


class MacroAnalyzer:
    """
    宏观市场分析器

    评分逻辑：
        纳斯达克走势    权重 35%  — 与BTC相关性最高
        美元指数(DXY)  权重 25%  — 美元强则BTC弱
        标普500走势    权重 15%  — 整体风险偏好
        VIX恐慌指数    权重 15%  — 高VIX = 市场恐慌
        恐慌贪婪指数   权重 10%  — 加密市场情绪
    """

    TICKERS = {
        "nasdaq": "QQQ",
        "sp500": "SPY",
        "dxy": "DX-Y.NYB",
        "gold": "GLD",
        "vix": "^VIX",
    }

    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 300  # 5分钟缓存

    def _fetch_all(self) -> dict:
        """一次性批量下载所有ticker，只发一个请求"""
        cache_key = "all_tickers"
        now = datetime.now().timestamp()

        if cache_key in self._cache:
            if now - self._cache_time.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]

        tickers = "QQQ SPY DX-Y.NYB ^VIX"
        try:
            data = yf.download(
                tickers,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
            result = {}
            for ticker in ["QQQ", "SPY", "DX-Y.NYB", "^VIX"]:
                try:
                    df = data[ticker].dropna()
                    if not df.empty:
                        result[ticker] = df
                except Exception:
                    pass
            self._cache[cache_key] = result
            self._cache_time[cache_key] = now
            logger.info(f"宏观数据批量获取成功：{list(result.keys())}")
            return result
        except Exception as e:
            logger.error(f"批量获取宏观数据失败：{e}")
            return {}

    def _fetch_fear_greed(self) -> tuple[int, str]:
        """
        获取加密恐慌贪婪指数
        来源：alternative.me（免费，无需Key）
        """
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10,
            )
            data = resp.json()["data"][0]
            value = int(data["value"])
            label = data["value_classification"]
            return value, label
        except Exception as e:
            logger.error(f"获取恐慌贪婪指数失败：{e}")
            return 50, "Neutral"

    def _nasdaq_signal(self, data: dict) -> MacroSignal:
        df = data.get("QQQ")
        if df is None or len(df) < 2:
            return MacroSignal("纳斯达克(QQQ)", 0, 0, 0, 0.35, "数据获取失败")
        close = df["Close"].dropna()
        change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
        value = float(close.iloc[-1])
        trend_5d = (
            float((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100)
            if len(close) >= 5
            else 0
        )

        if change > 1.0 or trend_5d > 2.0:
            signal, desc = (
                1,
                f"上涨 {change:+.2f}%，5日趋势 {trend_5d:+.2f}%，风险偏好改善",
            )
        elif change < -1.0 or trend_5d < -2.0:
            signal, desc = (
                -1,
                f"下跌 {change:+.2f}%，5日趋势 {trend_5d:+.2f}%，风险偏好恶化",
            )
        else:
            signal, desc = 0, f"震荡 {change:+.2f}%，方向不明确"
        return MacroSignal("纳斯达克(QQQ)", value, change, signal, 0.35, desc)

    def _dxy_signal(self, data: dict) -> MacroSignal:
        df = data.get("DX-Y.NYB")
        if df is None or len(df) < 2:
            return MacroSignal("美元指数(DXY)", 0, 0, 0, 0.25, "数据获取失败")
        close = df["Close"].dropna()
        change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
        value = float(close.iloc[-1])

        if change > 0.3:
            signal, desc = -1, f"美元走强 {change:+.2f}%，对BTC形成压力"
        elif change < -0.3:
            signal, desc = 1, f"美元走弱 {change:+.2f}%，对BTC形成支撑"
        else:
            signal, desc = 0, f"美元平稳 {change:+.2f}%，影响中性"
        return MacroSignal("美元指数(DXY)", value, change, signal, 0.25, desc)

    def _vix_signal(self, data: dict) -> MacroSignal:
        df = data.get("^VIX")
        if df is None or len(df) < 2:
            return MacroSignal("VIX恐慌指数", 0, 0, 0, 0.15, "数据获取失败")
        close = df["Close"].dropna()
        value = float(close.iloc[-1])
        change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)

        if value > 30:
            signal, desc = -1, f"VIX={value:.1f} 极度恐慌，市场系统性风险高"
        elif value > 20:
            signal, desc = -1, f"VIX={value:.1f} 恐慌上升，风险偏好降低"
        elif value < 15:
            signal, desc = 1, f"VIX={value:.1f} 市场平静，风险偏好良好"
        else:
            signal, desc = 0, f"VIX={value:.1f} 正常波动区间"
        return MacroSignal("VIX恐慌指数", value, change, signal, 0.15, desc)

    def _sp500_signal(self, data: dict) -> MacroSignal:
        df = data.get("SPY")
        if df is None or len(df) < 2:
            return MacroSignal("标普500(SPY)", 0, 0, 0, 0.15, "数据获取失败")
        close = df["Close"].dropna()
        change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
        value = float(close.iloc[-1])

        if change > 0.8:
            signal, desc = 1, f"标普上涨 {change:+.2f}%，整体风险偏好良好"
        elif change < -0.8:
            signal, desc = -1, f"标普下跌 {change:+.2f}%，整体风险偏好恶化"
        else:
            signal, desc = 0, f"标普平稳 {change:+.2f}%"

        return MacroSignal("标普500(SPY)", value, change, signal, 0.15, desc)

    def _fear_greed_signal(self, fg_value: int) -> MacroSignal:
        """加密恐慌贪婪指数信号"""
        if fg_value >= 75:
            signal, desc = -1, f"极度贪婪({fg_value})，市场过热，回调风险上升"
        elif fg_value >= 55:
            signal, desc = 1, f"贪婪({fg_value})，市场情绪积极"
        elif fg_value <= 25:
            signal, desc = 1, f"极度恐慌({fg_value})，历史上往往是买入时机"
        elif fg_value <= 45:
            signal, desc = -1, f"恐慌({fg_value})，市场情绪悲观"
        else:
            signal, desc = 0, f"中性({fg_value})，情绪平衡"

        return MacroSignal("恐慌贪婪指数", fg_value, 0, signal, 0.10, desc)

    def analyze(self) -> MacroAnalysis:
        """执行完整宏观分析"""
        logger.info("开始获取宏观数据...")

        # 并行获取所有信号
        data = self._fetch_all()

        nasdaq_sig = self._nasdaq_signal(data)
        dxy_sig = self._dxy_signal(data)
        vix_sig = self._vix_signal(data)
        sp500_sig = self._sp500_signal(data)
        fg_value, fg_label = self._fetch_fear_greed()
        fg_sig = self._fear_greed_signal(fg_value)

        signals = [nasdaq_sig, dxy_sig, vix_sig, sp500_sig, fg_sig]

        # 加权评分
        total_weight = sum(s.weight for s in signals)
        weighted_score = sum(s.signal * s.weight for s in signals)
        normalized = weighted_score / total_weight  # -1 到 +1

        # 转换到 -100 到 +100
        score = int(normalized * 100)

        if score >= 20:
            direction = "BULLISH"
        elif score <= -20:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # 风险偏好判断：纳指和VIX都支持才算真正 risk-on
        risk_on = nasdaq_sig.signal >= 0 and vix_sig.signal >= 0

        # 生成摘要
        summary_lines = [
            f"宏观评分：{score:+d}/100（{direction}）",
            f"风险偏好：{'开启 Risk-On' if risk_on else '关闭 Risk-Off'}",
            f"恐慌贪婪：{fg_value} - {fg_label}",
            "",
            "各指标信号：",
        ]
        for s in signals:
            icon = "↑" if s.signal == 1 else ("↓" if s.signal == -1 else "→")
            summary_lines.append(f"  {icon} {s.name}：{s.description}")

        summary = "\n".join(summary_lines)
        logger.info(
            f"宏观分析完成 | 评分:{score} | 方向:{direction} | Risk-On:{risk_on}"
        )

        return MacroAnalysis(
            score=score,
            direction=direction,
            signals=signals,
            fear_greed_index=fg_value,
            fear_greed_label=fg_label,
            risk_on=risk_on,
            summary=summary,
            data_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
