import os
from dotenv import load_dotenv
import okx.Account as Account
import okx.Trade as Trade
import logging

load_dotenv()

class OrderLogic:
    def __init__(self):
        key = os.getenv("OKX_API_KEY", "")
        secret = os.getenv("OKX_API_SECRET", "")
        passphrase = os.getenv("OKX_API_PASSPHRASE", "")
        flag = '1'
        self.account_api = Account.AccountAPI(key, secret, passphrase, False, flag)
        self.tradeAPI = Trade.TradeAPI(key, secret, passphrase, False, flag)


    def _get_balance(self):
        """
        Get the available balance of USDT
        """
        balance_data = self.account_api.get_account_balance()
        try:
            usdt_detail = next(
                d for d in balance_data['data'][0]['details'] if d['ccy'] == 'USDT'
            )
            avail_bal = float(usdt_detail['availBal'])  # 轉成 float
            return avail_bal
        except Exception as e:
            raise ValueError(f"Error: {e}")

    def _caculate_position(self,entry_price: float,stop_loss_price: float,loss_pct: float = 2.0,leverage: float = 5.0,) -> float:
        """
        Calculate the position size based on entry price, stop loss price, and risk percentage.
        :param entry_price: Entry price
        :param stop_loss_price: Stop loss price
        :param loss_pct: Risk percentage (default is 2.0)
        :param leverage: Leverage (default is 5.0)
        """
        try:
            principal = self._get_balance()
        except ValueError as e:
            raise ValueError(f"Error: {e}")
        fee_rate = 0.05
        if entry_price == stop_loss_price:
            raise ValueError("Stop-loss price Error")
        direction = "Long" if stop_loss_price < entry_price else "Short"
        risk_pct = (entry_price - stop_loss_price) / entry_price if direction == "Long" else (stop_loss_price - entry_price) / entry_price
        if risk_pct <= 0:
            raise ValueError("Risk rito Error")
        max_loss_total = principal * (loss_pct / 100)
        fee_contribution = 2 * (fee_rate / 100)
        denominator = leverage * (abs(risk_pct) + fee_contribution)
        if denominator == 0:
            raise ZeroDivisionError("Denominator Error")
        margin = max_loss_total / denominator /entry_price
        logging.info(f'margin: {margin}')
        return round(margin, 1)

    def create_order(self,instId, leverage, clOrdId, direction, entry_price, stop_loss_price) -> dict:
        """
        Create an order
        :param instId: Instrument ID
        :param leverage: Leverage
        :param clOrdId: Client Order ID
        :param direction: Order direction (Long/Short)
        :param entry_price: Entry price
        :param stop_loss_price: Stop loss price 
        """
        self.account_api.set_leverage(
            instId=instId,
            lever=leverage,
            mgnMode="cross"
        )
        margin = self._caculate_position(entry_price, stop_loss_price, leverage=leverage)
        result = self.tradeAPI.place_order(
            instId=instId,
            tdMode="cross",
            clOrdId=clOrdId,
            side="buy" if direction == "Long" else "sell",
            posSide=direction.lower(),
            ordType="market",
            sz=margin
        )
        logging.info(f"Order created: {result}")
        return result

    def close_order(self, instId, clOrdId, direction) -> dict:
        """
        Close an order
        :param instId: Instrument ID
        :param clOrdId: Client Order ID
        :param direction: Order direction (Long/Short)
        """
        result = self.tradeAPI.close_positions(
            instId=instId,
            mgnMode="cross",
            posSide=direction.lower(),
            clOrdId=clOrdId
        )
        return result
if __name__ == "__main__":
    order_logic = OrderLogic()
    try:
        r = order_logic.create_order("SOL-USDT-SWAP", 10, "1", "short", 30000, 29000)
        print(r)
    except Exception as e:
        print(f"Error: {e}")