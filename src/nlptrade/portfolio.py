import logging
from decimal import Decimal, InvalidOperation
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

    def refresh_balance(self):
        """거래소에서 최신 잔고를 가져와 업데이트합니다."""
        logging.info("Refreshing portfolio balance...")
        self.balance = self._fetch_balance()

    def _parse_and_validate_amount(self, coin: str, amount: Any) -> Optional[Decimal]:
        """
        다양한 타입의 amount 값을 Decimal으로 변환하고 유효성을 검사합니다.
        0보다 큰 경우에만 유효한 값으로 간주합니다.
        """
        try:
            # str으로 먼저 변환하여 부동소수점 부정확성을 방지합니다.
            amount_decimal = Decimal(str(amount))
        except (InvalidOperation, TypeError, ValueError):
            if amount:
                logging.debug(
                    f"Could not convert balance for '{coin}' to Decimal. "
                    f"Value was: '{amount}' (type: {type(amount)}). Skipping."
                )
            return None

        if amount_decimal > Decimal("0"):
            return amount_decimal
        return None

    def _fetch_balance(self) -> Dict[str, str]:
        """선택된 거래소에서 사용 가능한 잔고를 가져와 캐시합니다."""
        if not self.exchange:
            return {}
        try:
            logging.info(f"Fetching account balance from {self.exchange.name}...")
            balance_data = self.exchange.fetch_balance()
            free_balances = balance_data.get('free', {})
            logging.info(f"Successfully fetched account balance from {self.exchange.name}.")
            return {
                coin: str(valid_amount)
                for coin, amount in free_balances.items()
                if (valid_amount := self._parse_and_validate_amount(coin, amount)) is not None
            }
        except Exception as e:
            logging.error(f"Failed to fetch balance from {self.exchange.name}: {e}")
            return {}

    def get_coin_amount(self, coin_symbol: str) -> Optional[Decimal]:
        """캐시된 잔고에서 특정 코인의 사용 가능한 수량을 반환합니다."""
        amount_str = self.balance.get(coin_symbol)
        if amount_str is None:
            return None
        try:
            return Decimal(amount_str)
        except InvalidOperation:
            logging.warning(
                f"Could not convert cached balance for '{coin_symbol}' to Decimal. "
                f"Value was: '{amount_str}'. Returning None."
            )
            return None
