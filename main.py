"""
主入口 - 风控层完整演示
展示从信号输入到下单执行的完整流程

运行方式：
    python main.py

测试网 API Key 申请：
    https://testnet.binancefuture.com → 注册 → API Management
"""

import logging
import os
from core.risk_manager import RiskManager, TradeDirection
from core.binance_executor import BinanceFuturesExecutor

# ── 日志配置 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def demo_risk_calculations():
    """
    演示1：风控计算逻辑（不需要 API Key）
    验证各种情况下的拒绝和通过逻辑
    """
    print("\n" + "═" * 60)
    print("  演示：风控计算逻辑验证")
    print("═" * 60)

    # 假设你重新开始，初始资金 500 USDT
    risk = RiskManager(
        total_capital_usdt=500,
        max_loss_pct_per_trade=0.015,   # 单笔最大亏损 1.5% = 7.5 USDT
        max_daily_loss_pct=0.05,        # 日亏损上限 5% = 25 USDT
        max_leverage=3,                 # 最高 3 倍，保守起步
        min_risk_reward_ratio=1.5,
        min_confidence_score=65,
    )
    risk.print_risk_summary()

    # ── 场景1：正常入场，应该通过 ─────────────────────────────────────
    print("场景1：BTC 正常做多信号")
    decision = risk.evaluate_trade(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_price=95000,
        stop_loss_price=93100,   # 止损距离 2%
        take_profit_price=98850, # 止盈距离 4.05%，盈亏比约 2:1
        leverage=3,
        confidence_score=72,
        funding_rate_8h=0.0001,  # 年化约 10.95%，正常水平
    )
    _print_decision(decision)

    # ── 场景2：资金费率过高，拒绝做多 ────────────────────────────────
    print("\n场景2：资金费率过高（牛市顶部常见）")
    decision = risk.evaluate_trade(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_price=95000,
        stop_loss_price=93100,
        take_profit_price=98850,
        leverage=3,
        confidence_score=70,
        funding_rate_8h=0.0015,  # 年化约 164%，过热信号
    )
    _print_decision(decision)

    # ── 场景3：置信度不足，信号冲突 ──────────────────────────────────
    print("\n场景3：多维度信号冲突，置信度不足")
    decision = risk.evaluate_trade(
        symbol="ETHUSDT",
        direction=TradeDirection.LONG,
        entry_price=3500,
        stop_loss_price=3430,
        take_profit_price=3640,
        leverage=2,
        confidence_score=55,     # 低于最低要求 65
        funding_rate_8h=0.0001,
    )
    _print_decision(decision)

    # ── 场景4：盈亏比不足 ─────────────────────────────────────────────
    print("\n场景4：止盈太近，盈亏比不合格")
    decision = risk.evaluate_trade(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_price=95000,
        stop_loss_price=93100,   # 止损 2%
        take_profit_price=96000, # 止盈只有 1%，盈亏比 0.5:1
        leverage=3,
        confidence_score=75,
        funding_rate_8h=0.0001,
    )
    _print_decision(decision)

    # ── 场景5：模拟日亏损触发上限 ────────────────────────────────────
    print("\n场景5：今日已亏损较多，触发日亏损保护")
    risk.daily_loss_usdt = 24.0  # 模拟今天已亏 24 USDT（接近 5% = 25 USDT 上限）
    decision = risk.evaluate_trade(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_price=95000,
        stop_loss_price=93100,
        take_profit_price=98850,
        leverage=3,
        confidence_score=80,
        funding_rate_8h=0.0001,
    )
    _print_decision(decision)
    risk.daily_loss_usdt = 0  # 重置


def demo_full_flow_dry_run():
    """
    演示2：完整流程空跑（使用假数据，不真实下单）
    展示信号 → 风控 → 执行的完整链路
    """
    print("\n" + "═" * 60)
    print("  演示：完整交易流程（空跑模式）")
    print("═" * 60)

    # 假设分析系统输出了以下信号
    signal = {
        "symbol": "BTCUSDT",
        "direction": TradeDirection.LONG,
        "entry_price": 95000.0,
        "stop_loss_price": 93100.0,   # 关键支撑位下方
        "take_profit_price": 98900.0, # 前高附近
        "leverage": 3,
        "confidence_score": 73,       # 多时间框架对齐，无明显信号冲突
        "funding_rate_8h": 0.00008,   # 当前费率健康
    }

    print(f"\n收到分析信号：")
    print(f"  方向:     {signal['direction'].value}")
    print(f"  入场价:   ${signal['entry_price']:,.0f}")
    print(f"  止损价:   ${signal['stop_loss_price']:,.0f}  (距离 {(signal['entry_price']-signal['stop_loss_price'])/signal['entry_price']:.2%})")
    print(f"  止盈价:   ${signal['take_profit_price']:,.0f}  (距离 {(signal['take_profit_price']-signal['entry_price'])/signal['entry_price']:.2%})")
    print(f"  置信度:   {signal['confidence_score']}/100")
    print(f"  资金费率: {signal['funding_rate_8h']*3*365:.2%} 年化")

    # 风控评估
    risk = RiskManager(total_capital_usdt=500)
    decision = risk.evaluate_trade(**signal)

    print(f"\n风控结果：{'✓ 通过' if decision.approved else '✗ 拒绝'}")
    print(f"  {decision.message}")

    if decision.approved and decision.params:
        p = decision.params
        margin = p.position_usdt / p.leverage
        print(f"\n计算出的交易参数：")
        print(f"  开仓名义价值:  {p.position_usdt:>10.2f} USDT")
        print(f"  实际保证金:    {margin:>10.2f} USDT  ({margin/500:.1%} 的总资金)")
        print(f"  最大亏损:      {p.max_loss_usdt:>10.2f} USDT  ({p.max_loss_usdt/500:.1%} 的总资金)")
        print(f"  盈亏比:        {p.risk_reward_ratio:>9.2f}:1")
        print(f"  移动止盈回撤:  {p.trailing_stop_pct:>9.1%}")
        print(f"\n  → 如果确认入场，系统将自动：")
        print(f"     1. 以市价开 {p.position_usdt:.0f} USDT {p.direction.value} 仓（{p.leverage}x 杠杆）")
        print(f"     2. 同步挂止损单（触发价 ${p.stop_loss_price:,.0f}，标记价格触发）")
        print(f"     3. 同步挂止盈单（触发价 ${p.take_profit_price:,.0f}）")
        print(f"     4. 你不需要做任何其他操作")


def _print_decision(decision):
    status = "✓ 通过" if decision.approved else "✗ 拒绝"
    print(f"  结果：{status}")
    print(f"  信息：{decision.message}")
    if decision.approved and decision.params:
        p = decision.params
        margin = p.position_usdt / p.leverage
        print(f"  建议仓位：{p.position_usdt:.2f} USDT | 保证金：{margin:.2f} USDT | 盈亏比：{p.risk_reward_ratio:.2f}:1")


def demo_real_trading():
    """
    演示3：真实下单（需要测试网 API Key）
    取消注释并填入你的测试网 API Key 才能运行
    """
    print("\n" + "═" * 60)
    print("  真实下单演示（测试网）")
    print("═" * 60)

    # 填入你的测试网 API Key
    # API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
    # API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    API_KEY = "oGu4GwBib9t9RSH7AT1foVPb9oAu26ZwunLP9RrXz5Nmd6YfU6xJXnWxHWDcXuSe"
    API_SECRET = "DroKnMLPkkZf3Jv1Iq1itXIl10zadIcU1VuXdjr3mEZIfx4bdb6qaHu7eRt3izSF"

    if not API_KEY or not API_SECRET:
        print("  未设置测试网 API Key，跳过真实下单演示")
        print("  设置方法：")
        print("    export BINANCE_TESTNET_API_KEY=你的Key")
        print("    export BINANCE_TESTNET_API_SECRET=你的Secret")
        print("  申请地址：https://testnet.binancefuture.com")
        return

    executor = BinanceFuturesExecutor(API_KEY, API_SECRET, testnet=True)
    risk = RiskManager(total_capital_usdt=500)

    # 获取当前实时数据
    symbol = "BTCUSDT"
    funding_rate = executor.get_funding_rate(symbol)
    mark_price = executor.get_mark_price(symbol)

    if mark_price <= 0:
        print("无法获取价格，取消演示")
        return

    print(f"\n当前 {symbol} 标记价格: ${mark_price:,.2f}")
    print(f"当前资金费率: {funding_rate:.6f} (年化 {funding_rate*3*365:.2%})")

    # 基于实时价格构建测试信号
    stop_loss = mark_price * 0.98     # 止损 2%
    take_profit = mark_price * 1.04   # 止盈 4%

    decision = risk.evaluate_trade(
        symbol=symbol,
        direction=TradeDirection.LONG,
        entry_price=mark_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        leverage=3,
        confidence_score=70,
        funding_rate_8h=funding_rate,
    )

    if not decision.approved:
        print(f"\n风控拒绝：{decision.message}")
        return

    print(f"\n风控通过：{decision.message}")
    print("准备在测试网下单...")

    result = executor.open_position_with_guards(decision.params)

    if result.success:
        print(f"\n开仓成功！")
        print(f"  主单ID:   {result.entry_order.order_id}")
        print(f"  止损单ID: {result.stop_loss_order.order_id}")
        print(f"  止盈单ID: {result.take_profit_order.order_id if result.take_profit_order else '未挂'}")
    else:
        print(f"\n开仓失败：{result.error_message}")


if __name__ == "__main__":
    # 演示1：风控逻辑验证（始终运行，不需要 API Key）
    demo_risk_calculations()

    # 演示2：完整流程空跑
    demo_full_flow_dry_run()

    # 演示3：真实测试网下单（需要 API Key）
    demo_real_trading()

    print("\n" + "═" * 60)
    print("  下一步：")
    print("  1. 申请币安测试网 API Key 跑通真实下单流程")
    print("  2. 开发多时间框架信号引擎（15m/1h/4h）")
    print("  3. 接入宏观数据和链上数据")
    print("  4. 集成 Claude API 生成分析报告")
    print("═" * 60 + "\n")
