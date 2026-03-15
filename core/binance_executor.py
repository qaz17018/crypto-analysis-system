"""
币安永续合约执行模块
职责：开仓、同步挂止损单和移动止盈单，确保风控单不遗漏
重要原则：开仓和挂止损是原子操作，止损挂单失败必须立即平仓
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

from core.risk_manager import TradeParams, TradeDirection

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """下单结果"""
    success: bool
    order_id: Optional[str]
    filled_price: Optional[float]
    filled_qty: Optional[float]
    error_message: Optional[str]


@dataclass
class PositionResult:
    """完整开仓结果（主单 + 止损单 + 止盈单）"""
    success: bool
    entry_order: Optional[OrderResult]
    stop_loss_order: Optional[OrderResult]
    take_profit_order: Optional[OrderResult]
    error_message: Optional[str]


class BinanceFuturesExecutor:
    """
    币安永续合约执行器

    使用说明：
        正式交易前，先用 testnet=True 在测试网跑通全流程
        测试网地址：https://testnet.binancefuture.com
        测试网 API Key 在该网站单独申请，与正式网不通用

    开仓流程（严格顺序）：
        1. 设置杠杆
        2. 市价开仓
        3. 挂止损单（STOP_MARKET）
        4. 挂止盈单（TAKE_PROFIT_MARKET）
        5. 如果 3 或 4 失败，立即市价平仓并报警
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化币安客户端"""
        try:
            from binance.client import Client
            from binance import AsyncClient

            if self.testnet:
                self.client = Client(
                    self.api_key,
                    self.api_secret,
                    testnet=True
                )
                logger.info("已连接到币安测试网（资金安全，可放心测试）")
            else:
                self.client = Client(self.api_key, self.api_secret)
                logger.warning("已连接到币安正式网，请谨慎操作")

        except ImportError:
            logger.error("请先安装 python-binance：pip install python-binance")
            raise
        except Exception as e:
            logger.error(f"客户端初始化失败：{e}")
            raise

    def get_funding_rate(self, symbol: str) -> float:
        """
        获取当前资金费率（每8小时）
        返回值示例：0.0001 表示 0.01%
        """
        try:
            result = self.client.futures_funding_rate(symbol=symbol, limit=1)
            rate = float(result[0]['fundingRate'])
            logger.info(f"{symbol} 当前资金费率：{rate:.6f} (年化 {rate * 3 * 365:.2%})")
            return rate
        except Exception as e:
            logger.error(f"获取资金费率失败：{e}")
            return 0.0

    def get_mark_price(self, symbol: str) -> float:
        """获取标记价格（用于计算盈亏，比最新价更稳定）"""
        try:
            result = self.client.futures_mark_price(symbol=symbol)
            return float(result['markPrice'])
        except Exception as e:
            logger.error(f"获取标记价格失败：{e}")
            return 0.0

    def get_symbol_precision(self, symbol: str) -> tuple[int, int]:
        """
        获取交易对精度
        返回 (价格小数位, 数量小数位)
        """
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    price_precision = s['pricePrecision']
                    qty_precision = s['quantityPrecision']
                    return price_precision, qty_precision
        except Exception as e:
            logger.error(f"获取精度信息失败：{e}")
        return 2, 3  # 默认值

    def _round_price(self, price: float, precision: int) -> str:
        """格式化价格到正确精度"""
        return f"{price:.{precision}f}"

    def _round_qty(self, qty: float, precision: int) -> str:
        """格式化数量到正确精度"""
        return f"{qty:.{precision}f}"

    def open_position_with_guards(self, params: TradeParams) -> PositionResult:
        """
        核心方法：开仓并同时设置止损止盈
        止损和止盈单失败会触发自动平仓保护
        """
        symbol = params.symbol
        price_precision, qty_precision = self.get_symbol_precision(symbol)
        mark_price = self.get_mark_price(symbol)

        if mark_price <= 0:
            return PositionResult(
                success=False,
                entry_order=None,
                stop_loss_order=None,
                take_profit_order=None,
                error_message="无法获取标记价格，取消开仓",
            )

        # 计算开仓数量
        qty = params.position_usdt / mark_price
        qty_str = self._round_qty(qty, qty_precision)
        sl_str = self._round_price(params.stop_loss_price, price_precision)
        tp_str = self._round_price(params.take_profit_price, price_precision)

        side = "BUY" if params.direction == TradeDirection.LONG else "SELL"
        close_side = "SELL" if params.direction == TradeDirection.LONG else "BUY"

        logger.info(
            f"准备开仓 {symbol} {side} | "
            f"数量 {qty_str} | 止损 {sl_str} | 止盈 {tp_str} | "
            f"杠杆 {params.leverage}x"
        )

        # ── 步骤 1：设置杠杆 ──────────────────────────────────────────
        try:
            self.client.futures_change_leverage(
                symbol=symbol,
                leverage=params.leverage
            )
        except Exception as e:
            return PositionResult(
                success=False, entry_order=None,
                stop_loss_order=None, take_profit_order=None,
                error_message=f"设置杠杆失败：{e}",
            )

        # ── 步骤 2：市价开仓 ──────────────────────────────────────────
        entry_order = None
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty_str,
            )
            entry_order = OrderResult(
                success=True,
                order_id=str(order.get('orderId', order.get('order_id', 'unknown'))),
                filled_price=float(order.get('avgPrice') or order.get('price') or mark_price),
                filled_qty=float(order.get('executedQty') or order.get('origQty') or qty),
                error_message=None,
            )
            logger.info(f"开仓成功 | 订单ID {entry_order.order_id} | 成交价 {entry_order.filled_price}")
            time.sleep(1.0)  # 等待成交完全确认后再挂条件单

        except Exception as e:
            return PositionResult(
                success=False, entry_order=None,
                stop_loss_order=None, take_profit_order=None,
                error_message=f"开仓失败：{e}",
            )

        # ── 步骤 3：挂止损单（STOP_MARKET）────────────────────────────
        # 止损单失败 = 资金暴露在风险中，必须立即平仓
        sl_order = None
        try:
            sl = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=sl_str,
                quantity=qty_str,
                reduceOnly="true",
                workingType="MARK_PRICE",
            )
            # 测试网条件单返回 algoId，正式网返回 orderId，兼容两种格式
            order_id = sl.get('orderId') or sl.get('algoId') or sl.get('order_id') or 'unknown'
            sl_order = OrderResult(
                success=True,
                order_id=str(order_id),
                filled_price=None,
                filled_qty=None,
                error_message=None,
            )
            logger.info(f"止损单已挂 | 触发价 {sl_str} | 订单ID {sl_order.order_id}")

        except Exception as e:
            # 止损挂单失败，立即平仓保护
            logger.error(f"止损单挂单失败：{e}，触发紧急平仓保护")
            self._emergency_close(symbol, close_side, qty_str)
            return PositionResult(
                success=False,
                entry_order=entry_order,
                stop_loss_order=None,
                take_profit_order=None,
                error_message=f"止损单失败已触发紧急平仓：{e}",
            )

        # ── 步骤 4：挂止盈单（TAKE_PROFIT_MARKET）────────────────────
        tp_order = None
        try:
            tp = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp_str,
                quantity=qty_str,
                reduceOnly="true",
                workingType="MARK_PRICE",
            )
            tp_order = OrderResult(
                success=True,
                order_id=str(tp.get('orderId') or tp.get('algoId') or 'unknown'),
                filled_price=None,
                filled_qty=None,
                error_message=None,
            )
            logger.info(f"止盈单已挂 | 触发价 {tp_str} | 订单ID {tp_order.order_id}")

        except Exception as e:
            # 止盈单失败不平仓（止损单还在保护），但记录警告
            logger.warning(f"止盈单挂单失败：{e}，止损单仍有效，请手动补挂止盈")
            tp_order = OrderResult(
                success=False,
                order_id=None,
                filled_price=None,
                filled_qty=None,
                error_message=str(e),
            )

        logger.info(
            f"开仓完成 ✓ | {symbol} {params.direction.value} | "
            f"止损 {sl_str} | 止盈 {tp_str}"
        )

        return PositionResult(
            success=True,
            entry_order=entry_order,
            stop_loss_order=sl_order,
            take_profit_order=tp_order,
            error_message=None,
        )

    def _emergency_close(self, symbol: str, side: str, qty: str):
        """紧急平仓，止损挂单失败时调用"""
        try:
            self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty,
                reduceOnly=True,
            )
            logger.warning(f"紧急平仓成功 {symbol}")
        except Exception as e:
            logger.critical(f"紧急平仓也失败了！请立即手动平仓！{e}")

    def cancel_all_orders(self, symbol: str):
        """取消某交易对所有挂单（平仓后清理用）"""
        try:
            # 取消普通挂单
            self.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"{symbol} 所有普通挂单已取消")
        except Exception as e:
            logger.error(f"取消普通挂单失败：{e}")
        try:
            # 取消条件单（测试网止损止盈属于此类）
            algo_orders = self.client.futures_get_all_open_orders(symbol=symbol)
            for o in algo_orders:
                algo_id = o.get('algoId')
                if algo_id:
                    self.client.futures_cancel_order(symbol=symbol, orderId=algo_id)
            logger.info(f"{symbol} 条件单已取消")
        except Exception as e:
            logger.warning(f"取消条件单失败（可忽略）：{e}")

    def get_position(self, symbol: str) -> dict:
        """查询当前持仓"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for p in positions:
                if float(p['positionAmt']) != 0:
                    return p
        except Exception as e:
            logger.error(f"查询持仓失败：{e}")
        return {}
