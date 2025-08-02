import logging
from typing import Optional, Dict, Any

import ccxt

class PortfolioManager:
    """
    사용자의 거래소 포트폴리오(계좌 잔고)를 관리합니다.
    API 키는 설정 파일에서 로드합니다.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        PortfolioManager를 초기화합니다.
        API 키는 반드시 설정 파일에 설정되어 있어야 합니다:
        - api_key
        - secret_key
        """
        api_key = config.get("api_key")
        secret_key = config.get("secret_key")

        if not api_key or not secret_key:
            logging.warning("Binance API 키가 설정 파일에 설정되지 않았습니다. 포트폴리오 기반 기능이 동작하지 않습니다.")
            self.exchange = None
            self.balance: Dict[str, float] = {}
        else:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
            })
            self.exchange.timeout = 30000
            self.balance = self._fetch_balance()

    def _fetch_balance(self) -> Dict[str, float]:
        """바이낸스에서 사용 가능한 잔고를 가져와 캐시합니다."""
        if not self.exchange:
            return {}
        try:
            logging.info("Fetching account balance from Binance...")
            balance_data = self.exchange.fetch_balance()
            # ccxt는 잔고를 문자열로 반환할 수 있으므로, float으로 변환하고 0보다 큰 경우만 필터링합니다.
            # balance_data.get('free', {})의 반환값은 Dict[str, Any]에 가까우므로,
            # 명시적 타입 변환을 통해 타입 안정성을 높이고 잠재적 런타임 오류를 방지합니다.
            free_balances = balance_data.get('free', {})
            logging.info("Successfully fetched account balance.")
            processed_balances = {}
            for coin, amount in free_balances.items():
                # ccxt는 잔고를 문자열, 숫자 등 다양한 타입으로 반환할 수 있습니다.
                # 타입 체커를 만족시키고 런타임 안정성을 높이기 위해 타입을 명시적으로 확인합니다.
                if isinstance(amount, (int, float)):
                    if amount > 0:
                        processed_balances[coin] = float(amount)
                elif isinstance(amount, str):
                    if not amount:  # 빈 문자열은 건너뜁니다.
                        continue
                    try:
                        amount_float = float(amount)
                        if amount_float > 0:
                            processed_balances[coin] = amount_float
                    except ValueError:
                        logging.debug(f"Could not convert string balance for '{coin}' to float. Value was: '{amount}'. Skipping.")
                elif amount:  # None이나 0이 아닌 다른 예기치 않은 타입은 로그를 남깁니다.
                    logging.debug(f"Balance for '{coin}' has an unsupported type: {type(amount)}. Value: '{amount}'. Skipping.")
            return processed_balances
        except Exception as e:
            logging.error(f"Failed to fetch balance from Binance: {e}")
            return {}

    def get_coin_amount(self, coin_symbol: str) -> Optional[float]:
        """캐시된 잔고에서 특정 코인의 사용 가능한 수량을 반환합니다."""
        return self.balance.get(coin_symbol)