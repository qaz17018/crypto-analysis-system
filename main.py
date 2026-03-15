"""
主入口
完整流程：信号引擎分析 → 风控评估 → 确认入场 → 自动下单
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()  # 自动读取 .env 文件

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")


def run():
    if not API_KEY or not API_SECRET:
        print("未设置 API Key，请检查 .env 文件")
        return

    from binance.client import Client
    from core.signal_engine import SignalEngine, SignalResult
    from core.risk_manager import RiskManager, TradeDirection
    from core.binance_executor import BinanceFuturesExecutor
    from core.market_analyzer import Signal

    client = Client(API_KEY, API_SECRET, testnet=True)
    risk = RiskManager(
        total_capital_usdt=500,
        max_loss_pct_per_trade=0.015,
        max_daily_loss_pct=0.05,
        max_leverage=3,
        min_risk_reward_ratio=1.5,
        min_confidence_score=65,
    )
    executor = BinanceFuturesExecutor(API_KEY, API_SECRET, testnet=True)

    # ── 第一步：信号引擎分析 ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  第一步：多时间框架信号分析")
    print("═" * 60)

    symbol = "BTCUSDT"  # 改这里切换交易对，如 ETHUSDT
    engine = SignalEngine(client, symbol)
    signal_result: SignalResult = engine.analyze()

    print(signal_result.analysis_summary)

    # ── 第二步：风控评估 ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  第二步：风控评估")
    print("═" * 60)

    if signal_result.direction == Signal.NEUTRAL:
        print("当前信号为中性，无方向，系统建议等待")
        return

    direction = (
        TradeDirection.LONG
        if signal_result.direction == Signal.LONG
        else TradeDirection.SHORT
    )

    funding_rate = executor.get_funding_rate(symbol)

    decision = risk.evaluate_trade(
        symbol=symbol,
        direction=direction,
        entry_price=signal_result.entry_price,
        stop_loss_price=signal_result.suggested_stop_loss,
        take_profit_price=signal_result.suggested_take_profit,
        leverage=3,
        confidence_score=signal_result.confidence_score,
        funding_rate_8h=funding_rate,
    )

    print(f"风控结果：{'✓ 通过' if decision.approved else '✗ 拒绝'}")
    print(f"原因：{decision.message}")

    if not decision.approved:
        return

    p = decision.params
    margin = p.position_usdt / p.leverage
    print(f"\n交易参数：")
    print(f"  方向:         {direction.value}")
    print(f"  置信度:       {signal_result.confidence_score}/100")
    print(f"  入场价:       ${p.entry_price:,.2f}")
    print(
        f"  止损价:       ${p.stop_loss_price:,.2f}  ({abs(p.stop_loss_price - p.entry_price)/p.entry_price:.2%})"
    )
    print(
        f"  止盈价:       ${p.take_profit_price:,.2f}  ({abs(p.take_profit_price - p.entry_price)/p.entry_price:.2%})"
    )
    print(f"  盈亏比:       {p.risk_reward_ratio:.2f}:1")
    print(f"  开仓名义值:   ${p.position_usdt:.2f} USDT")
    print(f"  实际保证金:   ${margin:.2f} USDT")
    print(f"  最大亏损:     ${p.max_loss_usdt:.2f} USDT")

    if signal_result.has_major_conflict:
        print(f"\n  ⚠ 警告：检测到重大信号冲突，建议谨慎")
        for c in signal_result.timeframe_conflicts:
            print(f"    - {c}")

    # ── 第三步：人工确认入场 ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  第三步：确认入场")
    print("═" * 60)
    print("  系统已完成分析，等待你的决定")
    print("  输入 y 确认入场，其他任意键跳过：", end="")
    confirm = input().strip().lower()

    if confirm != "y":
        print("已跳过，等待下一个信号")
        return

    # ── 第四步：自动下单 ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  第四步：执行下单")
    print("═" * 60)

    result = executor.open_position_with_guards(p)

    if result.success:
        print(f"\n开仓成功 ✓")
        print(f"  主单ID:   {result.entry_order.order_id}")
        print(f"  止损单ID: {result.stop_loss_order.order_id}")
        print(
            f"  止盈单ID: {result.take_profit_order.order_id if result.take_profit_order and result.take_profit_order.success else '未挂，请手动补挂'}"
        )
        print(f"\n  止损和止盈已自动挂单，无需再盯盘")
    else:
        print(f"\n开仓失败：{result.error_message}")


def run_backtest():
    """回测入口"""
    if not API_KEY:
        print("未设置 API Key")
        return

    from binance.client import Client
    from core.backtest import Backtester

    print("\n" + "═" * 60)
    print("  回测模式")
    print("═" * 60)

    client = Client(API_KEY, API_SECRET, testnet=True)
    backtester = Backtester(
        client=client,
        symbol="BTCUSDT",
        min_confidence=30,
        max_hold_candles=20,
    )

    # 分别回测 1h 和 15m
    for tf in ["1h", "15m"]:
        result = backtester.run(timeframe=tf, days=180)
        backtester.print_report(result, tf)


if __name__ == "__main__":
    run()
    # run_backtest()
