"""
多时间框架信号引擎
负责：四个时间框架聚合、置信度评分、冲突识别、止损止盈建议
"""

import logging
from dataclasses import dataclass
from typing import Optional

from binance.client import Client

from core.market_analyzer import MarketAnalyzer, FrameAnalysis, Signal, MarketState

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """信号引擎最终输出"""

    symbol: str
    confidence_score: int  # 0-100，直接输入风控层
    direction: Signal  # 最终方向

    # 四个时间框架结果
    daily: Optional[FrameAnalysis]
    h4: Optional[FrameAnalysis]
    h1: Optional[FrameAnalysis]
    m15: Optional[FrameAnalysis]

    # 风控所需参数
    entry_price: float
    suggested_stop_loss: float  # 基于支撑压力位计算
    suggested_take_profit: float

    # 冲突分析
    timeframe_conflicts: list[str]  # 哪些时间框架方向不一致
    has_major_conflict: bool  # 是否存在重大冲突

    # 市场状态摘要
    dominant_state: str  # 当前主导市场状态
    analysis_summary: str  # 给 AI 层的结构化摘要


class SignalEngine:
    """
    多时间框架信号引擎

    时间框架权重设计：
        日线  30% — 定大方向，权重最高
        4小时 25% — 确认结构
        1小时 25% — 找入场区域
        15分钟 20% — 精确入场，权重最低（噪音最多）

    置信度评分规则：
        基础分 = 加权方向得分（0-60分）
        一致性加分 = 时间框架方向一致数量（最多+25分）
        冲突扣分 = 存在冲突的时间框架数量（每个-10分）
        危机状态扣分 = 高波动市场（-15分）
        插针扣分 = 近期有插针（-5分）
    """

    WEIGHTS = {
        "1d": 0.30,
        "4h": 0.25,
        "1h": 0.25,
        "15m": 0.20,
    }

    def __init__(self, client: Client, symbol: str):
        self.symbol = symbol
        self.analyzers = {
            tf: MarketAnalyzer(client, symbol, tf) for tf in ["1d", "4h", "1h", "15m"]
        }

    def analyze(self) -> SignalResult:
        """执行完整的多时间框架分析，返回 SignalResult"""
        logger.info(f"开始分析 {self.symbol} 多时间框架信号...")

        # ── 各时间框架独立分析 ────────────────────────────────────────
        results = {}
        for tf, analyzer in self.analyzers.items():
            results[tf] = analyzer.analyze()

        daily = results["1d"]
        h4 = results["4h"]
        h1 = results["1h"]
        m15 = results["15m"]

        # 用最小时间框架的价格作为入场参考
        entry_price = m15.price if m15 else (h1.price if h1 else 0.0)

        # ── 加权方向得分 ──────────────────────────────────────────────
        weighted_score = 0.0
        valid_frames = 0
        for tf, frame in results.items():
            if frame is not None:
                weighted_score += frame.signal.value * self.WEIGHTS[tf]
                valid_frames += 1

        # 归一化到 -1 到 +1
        if valid_frames > 0:
            total_weight = sum(self.WEIGHTS[tf] for tf in results if results[tf])
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0

        # 确定最终方向
        if normalized_score > 0.15:
            direction = Signal.LONG
        elif normalized_score < -0.15:
            direction = Signal.SHORT
        else:
            direction = Signal.NEUTRAL

        # ── 置信度计算 ────────────────────────────────────────────────
        # 基础分（0-60）
        base_score = abs(normalized_score) * 60

        # 一致性加分（最多25分）
        signals = [f.signal for f in results.values() if f is not None]
        long_count = signals.count(Signal.LONG)
        short_count = signals.count(Signal.SHORT)
        dominant_count = max(long_count, short_count)
        consistency_bonus = (dominant_count / len(signals)) * 25 if signals else 0

        # 冲突检测
        timeframe_conflicts = []
        if (
            daily
            and h4
            and daily.signal != Signal.NEUTRAL
            and h4.signal != Signal.NEUTRAL
        ):
            if daily.signal != h4.signal:
                timeframe_conflicts.append("日线与4小时方向相反")
        if h4 and h1 and h4.signal != Signal.NEUTRAL and h1.signal != Signal.NEUTRAL:
            if h4.signal != h1.signal:
                timeframe_conflicts.append("4小时与1小时方向相反")
        if h1 and m15 and h1.signal != Signal.NEUTRAL and m15.signal != Signal.NEUTRAL:
            if h1.signal != m15.signal:
                timeframe_conflicts.append("1小时与15分钟方向相反")

        # 指标内部冲突
        internal_conflicts = [tf for tf, f in results.items() if f and f.conflict]
        if internal_conflicts:
            timeframe_conflicts.append(f"{','.join(internal_conflicts)} 指标内部冲突")

        conflict_penalty = len(timeframe_conflicts) * 10
        has_major_conflict = len(timeframe_conflicts) >= 2

        # 高波动扣分
        volatile_frames = [
            f for f in results.values() if f and f.market_state == MarketState.VOLATILE
        ]
        volatile_penalty = 15 if volatile_frames else 0

        # 插针扣分
        spike_penalty = 5 if any(f.has_spike for f in results.values() if f) else 0

        # 最终置信度
        confidence = int(
            base_score
            + consistency_bonus
            - conflict_penalty
            - volatile_penalty
            - spike_penalty
        )
        confidence = max(0, min(100, confidence))

        # ── 止损止盈建议 ──────────────────────────────────────────────
        # 综合多个时间框架的支撑压力位
        supports = [f.support for f in results.values() if f]
        resistances = [f.resistance for f in results.values() if f]

        if direction == Signal.LONG:
            # 做多：取最近支撑位（偏保守，取中间值）
            supports_sorted = sorted(supports, reverse=True)
            suggested_sl = (
                supports_sorted[1] if len(supports_sorted) > 1 else supports_sorted[0]
            )
            suggested_sl = min(suggested_sl, entry_price * 0.98)  # 最少2%空间

            resistances_sorted = sorted(resistances)
            suggested_tp = (
                resistances_sorted[0] if resistances_sorted else entry_price * 1.04
            )
            suggested_tp = max(suggested_tp, entry_price * 1.03)  # 最少3%空间

        elif direction == Signal.SHORT:
            resistances_sorted = sorted(resistances)
            suggested_sl = (
                resistances_sorted[1]
                if len(resistances_sorted) > 1
                else resistances_sorted[-1]
            )
            suggested_sl = max(suggested_sl, entry_price * 1.02)

            supports_sorted = sorted(supports, reverse=True)
            suggested_tp = supports_sorted[0] if supports_sorted else entry_price * 0.96
            suggested_tp = min(suggested_tp, entry_price * 0.97)

        else:
            suggested_sl = entry_price * 0.98
            suggested_tp = entry_price * 1.03

        # ── 主导市场状态 ──────────────────────────────────────────────
        states = [f.market_state.value for f in results.values() if f]
        dominant_state = max(set(states), key=states.count) if states else "未知"

        # ── 分析摘要（给 AI 层用）────────────────────────────────────
        summary_lines = [
            f"交易对：{self.symbol}",
            f"当前价格：{entry_price:.2f}",
            f"最终方向：{direction.name}（置信度 {confidence}/100）",
            f"主导市场状态：{dominant_state}",
            "",
            "各时间框架信号：",
        ]
        for tf, frame in results.items():
            if frame:
                summary_lines.append(
                    f"  {tf:4s}：{frame.signal.name:7s} | "
                    f"RSI {frame.rsi:.0f} | ADX {frame.adx:.0f} | "
                    f"状态:{frame.market_state.value}"
                )
        if timeframe_conflicts:
            summary_lines.append("\n检测到的信号冲突：")
            for c in timeframe_conflicts:
                summary_lines.append(f"  ⚠ {c}")

        summary_lines += [
            f"\n建议止损：{suggested_sl:.2f}（距入场 {abs(suggested_sl-entry_price)/entry_price:.2%}）",
            f"建议止盈：{suggested_tp:.2f}（距入场 {abs(suggested_tp-entry_price)/entry_price:.2%}）",
        ]

        analysis_summary = "\n".join(summary_lines)

        logger.info(
            f"信号引擎完成 | 方向:{direction.name} | "
            f"置信度:{confidence} | 冲突:{len(timeframe_conflicts)}个"
        )

        return SignalResult(
            symbol=self.symbol,
            confidence_score=confidence,
            direction=direction,
            daily=daily,
            h4=h4,
            h1=h1,
            m15=m15,
            entry_price=entry_price,
            suggested_stop_loss=suggested_sl,
            suggested_take_profit=suggested_tp,
            timeframe_conflicts=timeframe_conflicts,
            has_major_conflict=has_major_conflict,
            dominant_state=dominant_state,
            analysis_summary=analysis_summary,
        )
