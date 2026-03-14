"""
风控核心模块
职责：仓位计算、入场条件验证、资金费率熔断
这是整个系统最重要的模块，所有交易必须经过这里
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class RejectionReason(Enum):
    FUNDING_RATE_TOO_HIGH = "资金费率过高，做多成本不合理"
    FUNDING_RATE_TOO_LOW = "资金费率过低，做空成本不合理"
    INSUFFICIENT_BALANCE = "账户余额不足"
    SIGNAL_CONFLICT = "多维度信号存在冲突，置信度不足"
    DAILY_LOSS_LIMIT = "今日亏损已达上限，禁止继续开仓"
    POSITION_ALREADY_OPEN = "当前已有持仓，不重复开仓"
    STOP_LOSS_TOO_WIDE = "止损距离过大，风险敞口超标"


@dataclass
class TradeParams:
    """一笔交易的完整参数，由风控模块计算产生"""
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_pct: float      # 移动止盈回撤百分比
    position_usdt: float          # 建议开仓金额（USDT）
    leverage: int
    max_loss_usdt: float          # 本次交易最大亏损（USDT）
    risk_reward_ratio: float      # 盈亏比
    confidence_score: int         # 信号置信度 0-100
    funding_rate_annual: float    # 当前年化资金费率


@dataclass
class TradeDecision:
    """风控模块的最终决定"""
    approved: bool
    params: Optional[TradeParams]
    rejection_reason: Optional[RejectionReason]
    message: str


class RiskManager:
    """
    仓位计算和风控验证

    核心公式：
        建议仓位(USDT) = 总资金 × 单笔最大亏损比例 ÷ 止损距离百分比

    例：总资金 1000 USDT，最大亏损 1.5%，止损距离 2%
        建议仓位 = 1000 × 1.5% ÷ 2% = 750 USDT
        加 3 倍杠杆，实际保证金 = 750 ÷ 3 = 250 USDT
    """

    def __init__(
        self,
        total_capital_usdt: float,
        max_loss_pct_per_trade: float = 0.015,   # 单笔最大亏损 1.5%
        max_daily_loss_pct: float = 0.05,         # 日最大亏损 5%
        max_leverage: int = 5,                    # 最大杠杆倍数
        min_risk_reward_ratio: float = 1.5,       # 最低盈亏比 1.5:1
        min_confidence_score: int = 65,           # 最低信号置信度
        funding_rate_long_threshold: float = 0.50,   # 做多费率年化上限 50%
        funding_rate_short_threshold: float = -0.30, # 做空费率年化下限 -30%
        max_stop_loss_pct: float = 0.03,          # 止损距离上限 3%
    ):
        self.total_capital = total_capital_usdt
        self.max_loss_pct = max_loss_pct_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_leverage = max_leverage
        self.min_rr_ratio = min_risk_reward_ratio
        self.min_confidence = min_confidence_score
        self.funding_long_threshold = funding_rate_long_threshold
        self.funding_short_threshold = funding_rate_short_threshold
        self.max_stop_loss_pct = max_stop_loss_pct

        # 运行时状态
        self.daily_loss_usdt: float = 0.0
        self.has_open_position: bool = False

    def evaluate_trade(
        self,
        symbol: str,
        direction: TradeDirection,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        leverage: int,
        confidence_score: int,
        funding_rate_8h: float,          # 币安返回的每8小时费率，如 0.0001
        trailing_stop_pct: float = 0.008, # 移动止盈回撤幅度，默认 0.8%
    ) -> TradeDecision:
        """
        评估一笔交易是否通过风控
        返回 TradeDecision，approved=True 时包含完整交易参数
        """

        # 年化资金费率（方便人类理解）
        funding_annual = funding_rate_8h * 3 * 365

        # ── 检查 1：资金费率熔断 ──────────────────────────────────────
        if direction == TradeDirection.LONG and funding_annual > self.funding_long_threshold:
            return TradeDecision(
                approved=False,
                params=None,
                rejection_reason=RejectionReason.FUNDING_RATE_TOO_HIGH,
                message=(
                    f"当前资金费率年化 {funding_annual:.1%}，"
                    f"超过做多阈值 {self.funding_long_threshold:.1%}，"
                    f"持仓成本过高，拒绝开仓"
                ),
            )

        if direction == TradeDirection.SHORT and funding_annual < self.funding_short_threshold:
            return TradeDecision(
                approved=False,
                params=None,
                rejection_reason=RejectionReason.FUNDING_RATE_TOO_LOW,
                message=(
                    f"当前资金费率年化 {funding_annual:.1%}，"
                    f"低于做空阈值 {self.funding_short_threshold:.1%}，"
                    f"做空需支付过高费率，拒绝开仓"
                ),
            )

        # ── 检查 2：信号置信度 ────────────────────────────────────────
        if confidence_score < self.min_confidence:
            return TradeDecision(
                approved=False,
                params=None,
                rejection_reason=RejectionReason.SIGNAL_CONFLICT,
                message=(
                    f"信号置信度 {confidence_score}/100，"
                    f"低于最低要求 {self.min_confidence}，"
                    f"多维度信号可能存在冲突，拒绝开仓"
                ),
            )

        # ── 检查 3：日亏损上限 ────────────────────────────────────────
        max_daily_loss = self.total_capital * self.max_daily_loss_pct
        if self.daily_loss_usdt >= max_daily_loss:
            return TradeDecision(
                approved=False,
                params=None,
                rejection_reason=RejectionReason.DAILY_LOSS_LIMIT,
                message=(
                    f"今日已亏损 {self.daily_loss_usdt:.2f} USDT，"
                    f"达到日亏损上限 {max_daily_loss:.2f} USDT，"
                    f"今日禁止继续开仓"
                ),
            )

        # ── 检查 4：已有持仓 ──────────────────────────────────────────
        if self.has_open_position:
            return TradeDecision(
                approved=False,
                params=None,
                rejection_reason=RejectionReason.POSITION_ALREADY_OPEN,
                message="当前已有持仓，等待平仓后再开新仓",
            )

        # ── 检查 5：杠杆上限 ──────────────────────────────────────────
        leverage = min(leverage, self.max_leverage)

        # ── 计算止损距离 ──────────────────────────────────────────────
        if direction == TradeDirection.LONG:
            stop_loss_pct = (entry_price - stop_loss_price) / entry_price
            take_profit_pct = (take_profit_price - entry_price) / entry_price
        else:
            stop_loss_pct = (stop_loss_price - entry_price) / entry_price
            take_profit_pct = (entry_price - take_profit_price) / entry_price

        # ── 检查 6：止损距离上限 ──────────────────────────────────────
        if stop_loss_pct > self.max_stop_loss_pct:
            return TradeDecision(
                approved=False,
                params=None,
                rejection_reason=RejectionReason.STOP_LOSS_TOO_WIDE,
                message=(
                    f"止损距离 {stop_loss_pct:.2%} 超过上限 {self.max_stop_loss_pct:.2%}，"
                    f"单笔风险敞口过大，请缩小止损或等待更好入场点"
                ),
            )

        # ── 检查 7：盈亏比 ────────────────────────────────────────────
        rr_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
        if rr_ratio < self.min_rr_ratio:
            return TradeDecision(
                approved=False,
                params=None,
                rejection_reason=RejectionReason.SIGNAL_CONFLICT,
                message=(
                    f"盈亏比 {rr_ratio:.2f}:1，"
                    f"低于最低要求 {self.min_rr_ratio}:1，"
                    f"止盈空间不足，不值得冒险"
                ),
            )

        # ── 核心计算：仓位大小 ────────────────────────────────────────
        max_loss_usdt = self.total_capital * self.max_loss_pct
        # 开仓名义价值 = 最大亏损 ÷ 止损距离百分比
        position_notional = max_loss_usdt / stop_loss_pct
        # 实际保证金 = 名义价值 ÷ 杠杆
        position_margin = position_notional / leverage

        # ── 检查 8：余额是否足够 ──────────────────────────────────────
        available = self.total_capital - self.daily_loss_usdt
        if position_margin > available * 0.95:   # 留 5% 缓冲
            return TradeDecision(
                approved=False,
                params=None,
                rejection_reason=RejectionReason.INSUFFICIENT_BALANCE,
                message=(
                    f"需要保证金 {position_margin:.2f} USDT，"
                    f"但可用余额仅 {available:.2f} USDT"
                ),
            )

        params = TradeParams(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            trailing_stop_pct=trailing_stop_pct,
            position_usdt=position_notional,
            leverage=leverage,
            max_loss_usdt=max_loss_usdt,
            risk_reward_ratio=rr_ratio,
            confidence_score=confidence_score,
            funding_rate_annual=funding_annual,
        )

        logger.info(
            f"[风控通过] {symbol} {direction.value} | "
            f"仓位 {position_notional:.2f} USDT (保证金 {position_margin:.2f}) | "
            f"止损 {stop_loss_pct:.2%} | 止盈 {take_profit_pct:.2%} | "
            f"盈亏比 {rr_ratio:.2f}:1 | 置信度 {confidence_score}"
        )

        return TradeDecision(
            approved=True,
            params=params,
            rejection_reason=None,
            message=f"风控通过 | 最大亏损 {max_loss_usdt:.2f} USDT | 盈亏比 {rr_ratio:.2f}:1",
        )

    def record_trade_result(self, pnl_usdt: float):
        """记录交易结果，更新日亏损统计"""
        if pnl_usdt < 0:
            self.daily_loss_usdt += abs(pnl_usdt)
        self.has_open_position = False
        logger.info(f"交易结果记录：{pnl_usdt:+.2f} USDT | 今日累计亏损：{self.daily_loss_usdt:.2f} USDT")

    def reset_daily_stats(self):
        """每天UTC 00:00 调用，重置日统计"""
        self.daily_loss_usdt = 0.0
        logger.info("日统计已重置")

    def print_risk_summary(self):
        """打印当前风控参数摘要"""
        print("\n" + "═" * 50)
        print("  风控参数摘要")
        print("═" * 50)
        print(f"  总资金:           {self.total_capital:>10.2f} USDT")
        print(f"  单笔最大亏损:     {self.max_loss_pct:>9.1%}  ({self.total_capital * self.max_loss_pct:.2f} USDT)")
        print(f"  日最大亏损:       {self.max_daily_loss_pct:>9.1%}  ({self.total_capital * self.max_daily_loss_pct:.2f} USDT)")
        print(f"  最大杠杆:         {self.max_leverage:>10}x")
        print(f"  最低盈亏比:       {self.min_rr_ratio:>9.1f}:1")
        print(f"  最低置信度:       {self.min_confidence:>10}/100")
        print(f"  做多费率上限:     {self.funding_long_threshold:>9.1%} (年化)")
        print(f"  止损距离上限:     {self.max_stop_loss_pct:>9.1%}")
        print(f"  今日已亏损:       {self.daily_loss_usdt:>10.2f} USDT")
        print("═" * 50 + "\n")
