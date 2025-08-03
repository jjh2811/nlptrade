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

    def _parse_and_validate_amount(self, coin: str, amount: Any) -> Optional[float]:
        """
        다양한 타입의 amount 값을 float으로 변환하고 유효성을 검사합니다.
        0보다 큰 경우에만 유효한 값으로 간주합니다.
        """
        try:
            # ccxt는 잔고를 문자열, 숫자 등 다양한 타입으로 반환할 수 있습니다.
            # float()으로 변환을 시도하여 대부분의 경우를 처리합니다.
            amount_float = float(amount)
        except (ValueError, TypeError):
            # float으로 변환할 수 없는 경우 (e.g., 빈 문자열, None, 지원하지 않는 타입)
            # 0, 0.0, '', None 등은 일반적인 경우이므로 로그를 남기지 않습니다.
            if amount:
                logging.debug(
                    f"Could not convert balance for '{coin}' to float. "
                    f"Value was: '{amount}' (type: {type(amount)}). Skipping."
                )
            return None

        if amount_float > 0:
            return amount_float
        return None

    def _fetch_balance(self) -> Dict[str, float]:
        """바이낸스에서 사용 가능한 잔고를 가져와 캐시합니다."""
        if not self.exchange:
            return {}
        try:
            logging.info("Fetching account balance from Binance...")
            balance_data = self.exchange.fetch_balance()
            free_balances = balance_data.get('free', {})
            logging.info("Successfully fetched account balance.")

            processed_balances = {}
            for coin, amount in free_balances.items():
                valid_amount = self._parse_and_validate_amount(coin, amount)
                if valid_amount is not None:
                    processed_balances[coin] = valid_amount
            return processed_balances
        except Exception as e:
            logging.error(f"Failed to fetch balance from Binance: {e}")
            return {}

    def get_coin_amount(self, coin_symbol: str) -> Optional[float]:
        """캐시된 잔고에서 특정 코인의 사용 가능한 수량을 반환합니다."""
        return self.balance.get(coin_symbol)