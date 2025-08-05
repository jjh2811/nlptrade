import logging
from typing import Any, Dict, Optional
from .types import Exchange



class PortfolioManager:
    """
    사용자의 거래소 포트폴리오(계좌 잔고)를 관리합니다.
    API 키는 설정 파일에서 로드합니다.
    """

    def __init__(self, exchange: Exchange):
        """
        PortfolioManager를 초기화합니다.

        Args:
            exchange: ccxt 거래소 인스턴스.
        """
        self.exchange = exchange
        self.balance = self._fetch_balance()

    def _parse_and_validate_amount(self, coin: str, amount: Any) -> Optional[float]:
        """
        다양한 타입의 amount 값을 float으로 변환하고 유효성을 검사합니다.
        0보다 큰 경우에만 유효한 값으로 간주합니다.
        """
        try:
            amount_float = float(amount)
        except (ValueError, TypeError):
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
        """선택된 거래소에서 사용 가능한 잔고를 가져와 캐시합니다."""
        if not self.exchange:
            return {}
        try:
            logging.info(f"Fetching account balance from {self.exchange.name}...")
            balance_data = self.exchange.fetch_balance()
            free_balances = balance_data.get('free', {})
            logging.info(f"Successfully fetched account balance from {self.exchange.name}.")
            return {
                coin: valid_amount
                for coin, amount in free_balances.items()
                if (valid_amount := self._parse_and_validate_amount(coin, amount)) is not None
            }
        except Exception as e:
            logging.error(f"Failed to fetch balance from {self.exchange.name}: {e}")
            return {}

    def get_coin_amount(self, coin_symbol: str) -> Optional[float]:
        """캐시된 잔고에서 특정 코인의 사용 가능한 수량을 반환합니다."""
        return self.balance.get(coin_symbol)