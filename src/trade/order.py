class OrderLogic:
    def __init__(self,symbol, platform='binance'):
        if platform not in {"binance", "okx"}:
            raise ValueError(f"binance and okx only")
    def caculate_stop_loss(self):
        pass

    def caculate_take_profit(self):
        pass
    
    def caculate_position(self):
        pass

    def create_order(self,direction):
        pass