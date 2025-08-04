import logging
from typing import Optional, Dict, Any, Type, Protocol, TYPE_CHECKING
import ccxt

from ccxt.base.types import Balances

class Exchange(Protocol):
    """코드에 사용될 ccxt.Exchange의 메서드에 대한 프로토콜 정의(타입 확인용)"""
    def fetch_balance(self) -> Balances: ...
    @property
    def name(self) -> Optional[str]: ...



class PortfolioManager:
    """
    사용자의 거래소 포트폴리오(계좌 잔고)를 관리합니다.
    API 키는 설정 파일에서 로드합니다.
    """
    def __init__(self, exchange_id: str, config: Dict[str, Any], use_testnet: bool = False):
        """
        PortfolioManager를 초기화합니다.

        Args:
            exchange_id: 사용할 거래소의 ID (예: 'binance', 'upbit').
            config: 전체 설정 객체. 'exchanges' 키 아래에 거래소별 설정이 있어야 합니다.
            use_testnet: 테스트넷 사용 여부. 바이낸스에만 적용됩니다.
        """
        self.exchange_id = exchange_id
        self.use_testnet = use_testnet
        self.exchange: Optional[Exchange] = self._initialize_exchange(exchange_id, config, use_testnet)
        self.balance = self._fetch_balance()

    def _initialize_exchange(self, exchange_id: str, config: Dict[str, Any], use_testnet: bool) -> Optional[Exchange]:
        """거래소 ID와 설정에 따라 ccxt 거래소 인스턴스를 생성하고 초기화합니다."""
        exchanges_config = config.get("exchanges", {})
        
        # 테스트넷 사용 시 설정 키를 변경합니다.
        if use_testnet and exchange_id == 'binance':
            config_key = 'binance_testnet'
        else:
            config_key = exchange_id

        exchange_config = exchanges_config.get(config_key)

        if not exchange_config:
            logging.warning(f"'{config_key}'에 대한 설정이 'exchanges' 섹션에 없습니다. 포트폴리오 기능이 비활성화됩니다.")
            return None

        # 테스트넷은 현재 바이낸스만 지원합니다.
        if use_testnet and exchange_id != 'binance':
            logging.warning(f"'{exchange_id}' 거래소는 테스트넷을 지원하지 않습니다. 메인넷으로 계속 진행합니다.")
            use_testnet = False

        api_key = exchange_config.get("api_key")
        secret_key = exchange_config.get("secret_key")
        
        network_name = f"{exchange_id.capitalize()}"
        if exchange_id == 'binance':
            network_name += " Testnet" if use_testnet else " Mainnet"

        if not api_key or not secret_key:
            logging.warning(f"{network_name} API 키가 설정 파일에 없습니다. 포트폴리오 기능이 비활성화됩니다.")
            return None

        try:
            exchange_class: Type[ccxt.Exchange] = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret_key,
                'timeout': 30000,
            })

            if exchange_id == 'binance' and use_testnet:
                exchange.set_sandbox_mode(True)

            logging.info(f"PortfolioManager for {network_name} initialized.")
            return exchange
        except AttributeError:
            logging.error(f"'{exchange_id}'는 지원되지 않는 거래소입니다. ccxt에 해당 거래소가 있는지 확인해주세요.")
            return None
        except Exception as e:
            logging.error(f"'{exchange_id}' 거래소 초기화 중 오류 발생: {e}")
            return None

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