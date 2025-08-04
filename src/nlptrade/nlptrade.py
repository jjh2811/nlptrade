import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from pathlib import Path

import unicodedata
from rapidfuzz import fuzz
import ccxt
from mecab import MeCab
from .portfolio import PortfolioManager

# MeCab 객체 초기화
mecab = MeCab()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TradeCommand:
    intent: str  # "buy" or "sell"
    coin: str  # e.g., "BTC", "ETH" (영문 심볼로 통일)
    amount: Optional[float]  # 거래 수량
    price: Optional[float]  # 지정가 가격 (시장가의 경우 None)
    order_type: str  # "market" or "limit"
    total_cost: Optional[float] = None # 총 주문 비용

def clean_text(text: str) -> str:
    """유효하지 않은 Unicode 문자를 제거하거나 대체"""
    # 유니코드 정규화 (NFKC: 호환성 문자 처리)
    text = unicodedata.normalize('NFKC', text)
    # 유효하지 않은 문자 (surrogate 등) 제거
    text = ''.join(c for c in text if c.isprintable() and ord(c) < 0x10000)
    return text

class EntityExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.coins: List[str] = config.get("coins", [])
        self.intent_map: Dict[str, str] = config.get("intent_map", {})
        self.fuzzy_threshold: int = config.get("fuzzy_threshold", 80)
        self.custom_mapping: Dict[str, str] = config.get("custom_mapping", {})

    def refresh_coins(self, executor: 'TradeExecutor'):
        """거래소에서 최신 코인 목록을 가져와 업데이트합니다."""
        try:
            updated_coins = fetch_exchange_coins(executor.exchange_id)
            self.coins = updated_coins
            logging.info(f"EntityExtractor: 코인 목록이 거래소에서 성공적으로 업데이트되었습니다. 총 {len(self.coins)}개 코인.")
        except Exception as e:
            logging.error(f"EntityExtractor: 코인 목록 업데이트 실패: {e}")

    def find_closest_symbol(self, input_symbol: str) -> Optional[str]:
        """입력된 심볼과 가장 유사한 심볼을 찾음"""
        if not input_symbol:
            return None

        # 대문자로 변환하여 일관성 유지
        input_symbol = input_symbol.upper()

        if input_symbol in self.coins:
            return input_symbol
        if input_symbol in self.custom_mapping:
            logging.info(f"Custom mapping found: {input_symbol} -> {self.custom_mapping[input_symbol]}")
            return self.custom_mapping[input_symbol]

        max_ratio = 0
        closest_symbol = None
        for symbol in self.coins:
            ratio = fuzz.WRatio(input_symbol, symbol)
            if ratio > max_ratio and ratio >= self.fuzzy_threshold:
                max_ratio = ratio
                closest_symbol = symbol

        if closest_symbol:
            # 추가적인 검증 로직: 'BAI' -> 'A' 와 같이, 길이 차이가 크고 매칭된 심볼이 매우 짧은 경우의 오류를 방지.
            # 1. 입력과 찾은 심볼의 길이 차이가 2 이상이고,
            # 2. 찾은 심볼의 길이가 2 이하인 경우, 잘못된 매칭으로 간주.
            len_diff = abs(len(input_symbol) - len(closest_symbol))
            if len_diff >= 2 and len(closest_symbol) <= 2:
                logging.warning(
                    f"Fuzzy match for '{input_symbol}' -> '{closest_symbol}' (ratio: {max_ratio}) rejected due to "
                    f"significant length difference with a very short symbol."
                )
                return None  # 매칭 거부

            logging.info(f"Fuzzy matching: '{input_symbol}' -> '{closest_symbol}' (ratio: {max_ratio})")
        return closest_symbol

    def _extract_intent(self, text: str) -> Optional[str]:
        """텍스트에서 거래 의도(매수/매도)를 추출"""
        # MeCab이 '사줘' -> ['사', '줘']로 분리하여 기존 로직에서 매칭이 어려운 문제 해결
        # 간단한 명령어에서는 키워드 검색이 더 안정적임
        for keyword, intent in self.intent_map.items():
            if keyword in text:
                logging.info(f"Intent matched: '{keyword}' in text -> '{intent}'")
                return intent
        return None

    def _extract_coin(self, text: str) -> Optional[str]:
        """텍스트에서 코인 심볼 또는 한글 이름(별칭)을 추출"""
        # 1. 영문 심볼 우선 추출 (e.g., BTC, ETH)
        # SUSHIDOWN, 1000SATS 등 더 길거나 숫자가 포함된 심볼을 처리하도록 정규식 수정
        symbol_pattern = r'\b[A-Z0-9]{2,12}\b'
        symbol_match = re.search(symbol_pattern, text.upper())
        if symbol_match:
            input_symbol = symbol_match.group(0)
            found_coin = self.find_closest_symbol(input_symbol)
            if found_coin:
                return found_coin

        # 2. 영문 심볼이 없는 경우, 등록된 한글/커스텀 매핑 키워드 검색
        # 긴 이름부터 매칭해야 '비트코인 캐시'와 '비트코인'을 구분할 수 있음
        sorted_custom_keys = sorted(self.custom_mapping.keys(), key=len, reverse=True)
        for coin_name in sorted_custom_keys:
            if coin_name in text:
                # custom_mapping에 있는 키는 find_closest_symbol에서 바로 찾아줌
                return self.find_closest_symbol(coin_name)

        # 3. 등록되지 않은 한글 이름의 경우, MeCab으로 명사를 추출하여 fuzzy matching 시도
        tokens = mecab.pos(text)
        logging.debug(f"Tokens for coin extraction: {tokens}")
        for token, pos in tokens:
            # 고유명사(NNP) 또는 일반명사(NNG)를 코인 이름 후보로 간주
            if pos.startswith('NN'): # NNP, NNG 등 명사류
                # 일반적인 거래 단위 등은 제외
                if token in ['개', '달러', '시장가', '지정가']:
                    continue
                found_coin = self.find_closest_symbol(token)
                if found_coin:
                    return found_coin
        return None

    def _extract_amount(self, text: str) -> Optional[float]:
        """텍스트에서 거래 수량을 추출"""
        amount_match = re.search(r'(\d+(?:\.\d+)?)\s*개', text)
        if amount_match:
            return float(amount_match.group(1))
        return None

    def _extract_price(self, text: str) -> Optional[float]:
        """텍스트에서 지정가 가격을 추출 (e.g., '1000원에', '50달러에', '50usdt에')"""
        # '현재가에'가 있으면 가격 추출을 무시
        if '현재가에' in text:
            return None
        price_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:원|달러|usdt)에', text, re.IGNORECASE)
        if price_match:
            return float(price_match.group(1))
        return None

    def _extract_total_cost(self, text: str) -> Optional[float]:
        """텍스트에서 '얼마어치' 또는 '얼마' 같은 총 비용을 추출"""
        # '...어치'를 먼저 찾고, 없으면 '...'을 찾는다.
        cost_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:원|달러|usdt)어치', text, re.IGNORECASE)
        if cost_match:
            return float(cost_match.group(1))
        
        # '...에'가 붙지 않은 경우를 비용으로 간주
        cost_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:원|달러|usdt)(?!\s*에)', text, re.IGNORECASE)
        if cost_match:
            return float(cost_match.group(1))

        return None

    def _extract_current_price_order(self, text: str) -> bool:
        """'현재가에' 키워드가 있는지 확인"""
        return '현재가에' in text

    def _extract_relative_amount(self, text: str) -> Optional[Dict[str, Any]]:
        """텍스트에서 '전부', '절반', '20%' 같은 상대적 수량을 추출"""
        # 키워드 기반 추출
        if '전부' in text or '전량' in text:
            return {'type': 'percentage', 'value': 100.0}
        if '절반' in text or '반' in text:
            return {'type': 'percentage', 'value': 50.0}

        # 퍼센트 기반 추출
        percentage_match = re.search(r'(\d+\.?\d*)\s*(%|퍼센트)', text)
        if percentage_match:
            return {'type': 'percentage', 'value': float(percentage_match.group(1))}
        return None

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """주어진 텍스트에서 거래 관련 모든 엔터티를 추출"""
        clean_input = clean_text(text)
        logging.info(f"Original text: '{text}', Cleaned text: '{clean_input}'")

        entities: Dict[str, Any] = {
            "intent": self._extract_intent(clean_input),
            "coin": self._extract_coin(clean_input),
            "amount": self._extract_amount(clean_input),
            "price": self._extract_price(clean_input),
            "relative_amount": self._extract_relative_amount(clean_input),
            "total_cost": self._extract_total_cost(clean_input),
            "current_price_order": self._extract_current_price_order(clean_input),
            "order_type": "market"
        }

        # 절대적 수량이 명시된 경우, 상대적 수량은 무시
        if entities["amount"] is not None:
            entities["relative_amount"] = None

        # 가격이 있거나 '현재가에' 주문이면 지정가
        if entities["price"] is not None or entities["current_price_order"]:
            entities["order_type"] = "limit"

        return entities

class TradeCommandParser:
    def __init__(self, extractor: EntityExtractor, portfolio_manager: PortfolioManager, executor: 'TradeExecutor'):
        """
        TradeCommandParser를 초기화합니다.

        Args:
            extractor: 엔터티 추출을 담당하는 EntityExtractor 객체.
            portfolio_manager: 포트폴리오 관리를 담당하는 PortfolioManager 객체.
            executor: TradeExecutor 객체.
        """
        self.extractor = extractor
        self.portfolio_manager = portfolio_manager
        self.executor = executor

    def parse(self, text: str) -> Optional[TradeCommand]:
        """주어진 텍스트를 파싱하여 TradeCommand 객체로 변환합니다."""
        entities = self.extractor.extract_entities(text)

        # 필수 엔터티 검증
        if not entities["intent"] or not entities["coin"]:
            logging.warning(
                f"Parse failed for text: '{text}'. "
                f"Missing intent ('{entities['intent']}') or coin ('{entities['coin']}')."
            )
            # 코인 정보가 없을 경우, 거래소에서 코인 목록을 새로고침하고 다시 시도
            if not entities["coin"]:
                logging.info("코인 정보를 찾을 수 없습니다. 거래소에서 코인 목록을 새로고침합니다.")
                self.extractor.refresh_coins(self.executor)
                entities = self.extractor.extract_entities(text) # 다시 엔터티 추출 시도
                if not entities["coin"]:
                    logging.warning(f"코인 목록 새로고침 후에도 코인 정보를 찾을 수 없습니다: '{text}'.")
                    return None
            else:
                return None

        # '현재가에' 주문 처리
        if entities.get("current_price_order"):
            coin_symbol = str(entities["coin"])
            intent = str(entities["intent"])
            order_book = self.executor.get_order_book(coin_symbol)

            if order_book:
                if intent == 'buy':
                    entities['price'] = order_book['bid']
                    logging.info(f"현재가 매수: {coin_symbol}의 1호 매수호가({order_book['bid']})로 지정가 설정")
                elif intent == 'sell':
                    entities['price'] = order_book['bid']
                    logging.info(f"현재가 매도: {coin_symbol}의 1호 매수호가({order_book['bid']})로 지정가 설정")
            else:
                logging.error(f"'{coin_symbol}'의 호가를 가져올 수 없어 '현재가에' 주문을 처리할 수 없습니다.")
                return None

        # 수량 정보 검증 (절대 수량, 상대 수량, 총 비용 중 하나는 있어야 함)
        if entities["amount"] is None and entities["relative_amount"] is None and entities["total_cost"] is None:
            logging.warning(
                f"Parse failed for text: '{text}'. "
                f"Missing amount information (e.g., '10개', '전부', '50%', '10000원어치')."
            )
            return None

        final_amount = entities.get("amount")
        total_cost = entities.get("total_cost")

        # 총 비용이 명시된 경우, 수량 계산
        if total_cost is not None:
            coin_symbol = str(entities["coin"])
            # '현재가에' 주문으로 가격이 이미 결정되었는지 확인
            price_to_use = entities.get("price")
            
            # 가격이 아직 결정되지 않았다면 (e.g. "비트코인 10000원어치"), 현재가 조회
            if price_to_use is None:
                price_to_use = self.executor.get_current_price(coin_symbol)

            if price_to_use is not None and price_to_use > 0:
                final_amount = total_cost / price_to_use
                quote_currency = self.executor._get_quote_currency()
                logging.info(f"계산된 수량: {total_cost} {quote_currency} / {price_to_use} {quote_currency}/coin -> {final_amount} {coin_symbol}")
            else:
                logging.error(f"'{coin_symbol}'의 현재 가격을 가져올 수 없어 총 비용 기반 주문을 처리할 수 없습니다.")
                return None

        # 상대적 수량이 감지된 경우, 실제 수량으로 변환
        relative_amount_info = entities.get("relative_amount")
        if relative_amount_info:
            coin_symbol = str(entities["coin"])
            current_holding = self.portfolio_manager.get_coin_amount(coin_symbol)

            if current_holding is None or current_holding <= 0:
                logging.warning(f"상대 수량을 처리할 수 없습니다. '{coin_symbol}'의 보유량이 없거나 잔고 조회에 실패했습니다.")
                return None

            percentage = relative_amount_info.get('value')
            if percentage is not None:
                calculated_amount = current_holding * (percentage / 100.0)
                final_amount = calculated_amount
                logging.info(f"계산된 수량: {percentage}% of {current_holding} {coin_symbol} -> {final_amount} {coin_symbol}")

        # 계산된 최종 수량이 0 이하인 경우 거래 중단
        if final_amount is not None and final_amount <= 0:
            logging.warning(f"계산된 거래 수량이 0 이하({final_amount})이므로 거래를 진행할 수 없습니다.")
            return None

        return TradeCommand(
            intent=str(entities["intent"]),
            coin=str(entities["coin"]),
            amount=final_amount,
            price=entities.get("price"),
            order_type=str(entities["order_type"]),
            total_cost=total_cost
        )

class TradeExecutor:
    """
    TradeCommand를 실행합니다.
    참고: 이 클래스는 실제 거래소 API와 연동하는 로직이 들어갈 위치의 예시입니다.
    """
    def __init__(self, exchange_id: str, config: Dict[str, Any]):
        self.exchange_id = exchange_id
        self.config = config
        self.quote_currency = self._get_quote_currency()
        self.exchange = self._initialize_exchange()

    def _initialize_exchange(self) -> ccxt.Exchange:
        """거래소 ID와 설정에 따라 ccxt 거래소 인스턴스를 생성하고 초기화합니다."""
        exchange_id_for_ccxt = self.exchange_id.replace('_testnet', '')
        is_testnet = self.exchange_id.endswith('_testnet')

        try:
            exchange_class = getattr(ccxt, exchange_id_for_ccxt)
            exchange = exchange_class(self.config.get('ccxt', {}))
            exchange.timeout = 30000  # 30초

            if is_testnet:
                exchange.set_sandbox_mode(True)
                logging.info(f"Initialized {exchange_id_for_ccxt.capitalize()} in testnet mode.")
            else:
                logging.info(f"Initialized {exchange_id_for_ccxt.capitalize()} in mainnet mode.")

            return exchange
        except Exception as e:
            logging.error(f"Failed to initialize exchange {self.exchange_id}: {e}")
            raise

    def _get_quote_currency(self) -> str:
        """설정에서 현재 거래소의 기본 통화를 가져옵니다."""
        try:
            return self.config['exchange_settings'][self.exchange_id]['quote_currency']
        except KeyError:
            logging.warning(
                f"'{self.exchange_id}'에 대한 'quote_currency' 설정이 없습니다. "
                f"기본값으로 'USDT'를 사용합니다."
            )
            return "USDT"

    def get_current_price(self, coin_symbol: str) -> Optional[float]:
        """지정된 코인의 현재 가격을 가져옵니다."""
        quote_currency = self._get_quote_currency()
        market_symbol = f'{coin_symbol}/{quote_currency}'
        try:
            ticker = self.exchange.fetch_ticker(market_symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Could not fetch price for {market_symbol}: {e}")
            return None

    def get_order_book(self, coin_symbol: str) -> Optional[Dict[str, float]]:
        """지정된 코인의 오더북을 가져와 1호가(매수/매도)를 반환합니다."""
        quote_currency = self._get_quote_currency()
        market_symbol = f'{coin_symbol}/{quote_currency}'
        try:
            # 페어로 오더북 조회
            order_book = self.exchange.fetch_order_book(market_symbol, limit=1)
            # 오더북에 매수/매도 오더가 있는지 확인
            if order_book['bids'] and order_book['asks']:
                best_bid = order_book['bids'][0][0]  # 가장 높은 매수 가격
                best_ask = order_book['asks'][0][0]  # 가장 낮은 매도 가격
                logging.info(f"Order book for {market_symbol}: Best Bid={best_bid}, Best Ask={best_ask}")
                return {'bid': best_bid, 'ask': best_ask}
            else:
                logging.warning(f"Order book for {market_symbol} is empty.")
                return None
        except Exception as e:
            logging.error(f"Could not fetch order book for {market_symbol}: {e}")
            return None

    def execute(self, command: TradeCommand) -> Dict:
        """주어진 명령을 실행하고 결과를 JSON 호환 딕셔너리로 반환합니다."""
        logging.info(f"Executing command: {command}")
        # 실제 거래 로직 추가 (예: self.exchange.create_market_buy_order(...))
        result = {
            "status": "success",
            "command_executed": command.__dict__
        }
        return result

def load_config(config_path: Path) -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"설정 파일의 형식이 올바르지 않습니다: {config_path}")
        raise

def load_secrets(secrets_path: Path) -> Dict[str, Any]:
    """비밀 설정 파일(API 키 등)을 로드합니다."""
    try:
        with open(secrets_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        example_secrets_path = secrets_path.with_name(secrets_path.name + '.example')
        if example_secrets_path.exists():
            logging.warning(
                f"비밀 설정 파일({secrets_path})을 찾을 수 없습니다. "
                f"예제 파일 '{example_secrets_path}'을 '{secrets_path.name}'으로 복사한 후, API 키 등 필요한 정보를 입력해주세요."
            )
        else:
            logging.warning(f"비밀 설정 파일을 찾을 수 없습니다: {secrets_path}. API 키가 필요한 기능은 동작하지 않을 수 있습니다.")
        return {}  # 키 파일이 없어도 프로그램은 계속 실행되도록 빈 딕셔너리 반환
    except json.JSONDecodeError:
        logging.error(f"비밀 설정 파일의 형식이 올바르지 않습니다: {secrets_path}")
        raise

def fetch_exchange_coins(exchange_id: str) -> List[str]:
    """ccxt를 사용하여 지정된 거래소의 모든 코인 목록을 가져옵니다."""
    try:
        logging.info(f"Fetching coin list from {exchange_id.capitalize()}...")
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        # 네트워크 타임아웃 설정 (단위: ms)
        exchange.timeout = 30000  # 30초
        markets = exchange.load_markets()
        base_coins = [market['base'] for market in markets.values()]
        unique_base_coins = sorted(list(set(base_coins)))
        logging.info(f"Successfully fetched {len(unique_base_coins)} unique coins from {exchange_id.capitalize()}.")
        return unique_base_coins
    except Exception as e:
        logging.error(f"Failed to fetch coin list from {exchange_id.capitalize()}: {e}")
        # 오류 발생 시 예외를 다시 발생시켜 main 함수에서 처리하도록 함
        raise

def main():
    """메인 실행 함수: 설정을 로드하고 대화형으로 명령을 처리합니다."""
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)
    secrets_path = Path(__file__).parent / "secrets.json"
    secrets = load_secrets(secrets_path)
    config.update(secrets)

    # 1. 기본 거래소 ID를 설정에서 가져와 코인 목록을 동적으로 로드
    default_exchange_id = config.get("default_exchange", "binance")
    try:
        exchange_coins = fetch_exchange_coins(default_exchange_id)
        config["coins"] = exchange_coins
    except Exception:
        logging.error(f"Could not start the bot because the coin list could not be fetched from {default_exchange_id.capitalize()}.")
        return  # 프로그램 종료

    # 2. 의존성 주입을 사용하여 컴포넌트 초기화
    # 2a. 포트폴리오 매니저 초기화
    portfolio_manager = PortfolioManager(default_exchange_id, config)
    executor = TradeExecutor(default_exchange_id, config)
    extractor = EntityExtractor(config)
    # 2b. 파서에 extractor와 portfolio_manager 주입
    parser = TradeCommandParser(extractor, portfolio_manager, executor)

    # 3. 대화형으로 명령어 처리
    print("--- NLP Trade-bot 시작 --- (종료하려면 'exit' 또는 'quit' 입력)")
    while True:
        try:
            text = input("\n[명령어 입력]> ")
            if text.lower() in ['exit', 'quit']:
                break

            command = parser.parse(text)
            if command:
                result = executor.execute(command)
                # ensure_ascii=False로 한글 깨짐 방지
                print(f"[실행 결과]: {json.dumps(result, indent=2, ensure_ascii=False)}")
            else:
                print("[실행 결과]: 명령을 해석하지 못했습니다.")

        except (KeyboardInterrupt, EOFError):
            # Ctrl+C 또는 Ctrl+D로 종료
            break
        except Exception as e:
            logging.error(f"오류 발생: {e}")

    print("\n--- NLP Trade-bot 종료 ---")

if __name__ == '__main__':
    main()
