from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import unicodedata
from decimal import Decimal, InvalidOperation

import ccxt
from ccxt.base.types import Num

from .portfolio import PortfolioManager
from .types import Exchange


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class TradeCommand:
    """사용자 입력에서 파생된 구조화된 거래 명령을 나타냅니다."""
    intent: str  # "buy" or "sell"
    symbol: Optional[str]  # e.g., "BTC/USDT", "ETH/USDT"
    amount: Optional[str]  # 거래 수량
    price: Optional[str]  # 지정가 가격 (시장가의 경우 None)
    order_type: str  # "market" or "limit"
    total_cost: Optional[str] = None  # 총 주문 비용


def clean_text(text: str) -> str:
    """유효하지 않은 Unicode 문자를 제거하거나 대체"""
    # 유니코드 정규화 (NFKC: 호환성 문자 처리)
    text = unicodedata.normalize('NFKC', text)
    # 유효하지 않은 문자 (surrogate 등) 제거
    text = ''.join(c for c in text if c.isprintable() and ord(c) < 0x10000)
    return text


class EntityExtractor:
    """
    자연어 텍스트 명령에서 거래 관련 엔티티(의도, 코인, 수량, 가격 등)를 추출합니다.
    정규식과 키워드 매칭을 사용하여 사용자 요청의 핵심 구성 요소를 식별합니다.
    """
    def __init__(self, config: Dict[str, Any]):
        self.coins: List[str] = config.get("coins", [])
        self.intent_map: Dict[str, str] = config.get("intent_map", {})
        self.custom_mapping: Dict[str, str] = config.get("custom_mapping", {})
        self.quote_currency: str = config.get("quote_currency", "USDT")

        self._update_max_coin_len()

    def _update_max_coin_len(self):
        if self.coins:
            self.max_coin_len = max(len(c) for c in self.coins)
        else:
            self.max_coin_len = 12  # 기본값
        logging.info(f"EntityExtractor: 코인 심볼 최대 길이 인식 수치가 {self.max_coin_len}(으)로 설정되었습니다.")

    def refresh_coins(self, executor: 'TradeExecutor'):
        """거래소에서 최신 코인 목록을 가져와 업데이트합니다."""
        try:
            executor.exchange.load_markets(reload=True)
            updated_coins = fetch_exchange_coins(executor.exchange)
            self.coins = updated_coins
            self._update_max_coin_len()
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

        return None

    def _is_english(self, text: str) -> bool:
        """입력된 텍스트가 영어 명령어인지 간단히 확인합니다."""
        english_keywords = ['market', 'limit', 'buy', 'sell']
        # 한글이 없고, 영어 키워드로 시작하는 경우 영어로 간주
        has_korean = any('가' <= char <= '힣' for char in text)
        return not has_korean and any(text.lower().startswith(keyword) for keyword in english_keywords)

    def _extract_intent(self, text: str, is_english: bool) -> Optional[str]:
        """텍스트에서 거래 의도(매수/매도)를 추출"""
        if is_english:
            # 영문: 정규식으로 buy/sell 추출
            match = re.search(r'\b(buy|sell)\b', text.lower())
            if match:
                intent = match.group(1)
                logging.info(f"Intent matched (English): '{intent}'")
                return intent
        else:
            # 한글: intent_map 기반 키워드 매칭
            for keyword, intent in self.intent_map.items():
                if keyword in text:
                    logging.info(f"Intent matched (Korean): '{keyword}' -> '{intent}'")
                    return intent
        return None

    def _extract_coin(self, text: str, is_english: bool) -> Optional[str]:
        """텍스트에서 코인 심볼 또는 한글 이름(별칭)을 추출"""
        # 영문/한글 공통: 영문 심볼 패턴으로 모든 잠재적 후보 추출
        symbol_pattern = rf'\b[A-Z0-9]{{2,{self.max_coin_len}}}(?![A-Z0-9])'
        potential_symbols = re.findall(symbol_pattern, text.upper())
        
        # 찾은 후보들 중에서 유효한 코인이 있는지 확인
        for symbol in potential_symbols:
            found_coin = self.find_closest_symbol(symbol)
            if found_coin:
                return found_coin

        if not is_english:
            # 한글: 커스텀 매핑(별칭) 검색
            sorted_custom_keys = sorted(self.custom_mapping.keys(), key=len, reverse=True)
            for coin_name in sorted_custom_keys:
                if coin_name in text:
                    found_coin = self.find_closest_symbol(coin_name)
                    if found_coin:
                        return found_coin
        
        return None

    def _extract_amount(self, text: str, is_english: bool) -> Optional[Decimal]:
        """텍스트에서 거래 수량을 추출"""
        try:
            if is_english:
                # 영문: 숫자만 추출 (나중에 tokens에서 처리)
                return None  # 영문은 토큰 기반 처리에서 담당
            else:
                # 한글: "개" 단위로 수량 추출
                amount_match = re.search(r'(\d+(?:\.\d+)?)\s*개', text)
                if amount_match:
                    return Decimal(amount_match.group(1))

                # 한글: "숫자 코인이름" 패턴으로 수량 추출
                if self.coins:
                    coin_pattern = '|'.join(re.escape(coin) for coin in self.coins)
                    pattern = rf'(\d+(?:\.\d+)?)\s*({coin_pattern})\b'
                    amount_coin_match = re.search(pattern, text, re.IGNORECASE)
                    if amount_coin_match:
                        return Decimal(amount_coin_match.group(1))
        except InvalidOperation:
            return None
        return None

    def _extract_price(self, text: str, is_english: bool) -> Optional[Decimal]:
        """텍스트에서 지정가 가격을 추출"""
        try:
            if is_english:
                # 영문: 숫자만 추출 (나중에 tokens에서 처리)
                return None  # 영문은 토큰 기반 처리에서 담당
            else:
                # 한글:
                # 상대 가격 패턴(+/- 숫자)이 있으면 일반(지정가) 가격으로 해석하지 않음
                if re.search(r'[+-]\d', text):
                    return None

                # "원에", "달러에", "usdt에" 패턴으로 가격 추출
                if '현재가' in text:
                    return None
                
                # 패턴 1: "10000에" 같이 '에'로 끝나는 명시적인 지정가
                price_match = re.search(r'(?<![+-])\b(\d+(?:\.\d+)?)\s*(?:원|달러|usdt)?에', text, re.IGNORECASE)
                if price_match:
                    return Decimal(price_match.group(1))

                # 패턴 2: '에'가 없는 경우. 수량, 총액, 퍼센트와 관련된 숫자를 제외하고 찾는다.
                # 퍼센트와 관련된 숫자는 가격이 될 수 없다.
                if re.search(r'\d+\s*(?:%|퍼센트)', text):
                    return None

                masked_text = text
                
                # 총액 패턴 마스킹 ("10000원어치")
                masked_text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:원|달러|usdt)\s*어치\b', ' MASKED_COST ', masked_text, re.IGNORECASE)

                # 수량 패턴 마스킹
                # "0.2개"
                masked_text = re.sub(r'\b\d+(?:\.\d+)?\s*개\b', ' MASKED_AMOUNT ', masked_text)
                
                # "0.2 BTC"
                if self.coins:
                    sorted_coins = sorted(self.coins, key=len, reverse=True)
                    coin_pattern = '|'.join(re.escape(coin) for coin in sorted_coins)
                    masked_text = re.sub(rf'\b\d+(?:\.\d+)?\s*({coin_pattern})\b', ' MASKED_AMOUNT ', masked_text, flags=re.IGNORECASE)

                # 마스킹된 텍스트에 남아있는 숫자 중 마지막 숫자를 가격으로 간주
                remaining_numbers = re.findall(r'(?<![+-])\b(\d+(?:\.\d+)?)\b', masked_text)
                
                if remaining_numbers:
                    return Decimal(remaining_numbers[-1])
        except InvalidOperation:
            return None
        return None

    def _extract_total_cost(self, text: str, is_english: bool) -> Optional[Decimal]:
        """텍스트에서 총 비용을 추출"""
        try:
            if is_english:
                # 영문: "with X usdt" 패턴으로 추출
                cost_match = re.search(r'with\s*(\d+(?:\.\d+)?)\s*(?:usdt|krw)?', text.lower())
                if cost_match:
                    return Decimal(cost_match.group(1))
            else:
                # 한글: "어치" 또는 통화 단위로 추출
                cost_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:원|달러|usdt)어치', text, re.IGNORECASE)
                if cost_match:
                    return Decimal(cost_match.group(1))

                cost_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:원|달러|usdt)(?!\s*에)', text, re.IGNORECASE)
                if cost_match:
                    return Decimal(cost_match.group(1))
        except InvalidOperation:
            return None
        return None

    def _extract_current_price_order(self, text: str, is_english: bool) -> bool:
        """'현재가' 키워드가 있는지 또는 암시적 현재가 주문인지 확인"""
        if is_english:
            # 영문: 'limit' 주문이면서 명시적인 가격 지정이 없는 경우 True를 반환.
            # Parser에서 최종적으로 price 존재 여부를 확인하여 처리함.
            if text.lower().startswith('limit'):
                # 상대 가격 지정(e.g. +5%)이 있으면 명시적 가격이 있는 것임
                if re.search(r'[+-]\d+(?:\.\d+)?', text):
                    return False
                return True
            return False
        else:
            # '현재가' 없는 문장도 현재가 주문하려했으나 시장가 주문하고 겹침
            return '현재가' in text

    def _extract_relative_price(self, text: str, is_english: bool) -> Optional[Decimal]:
        """텍스트에서 상대적 가격을 추출"""
        try:
            if is_english:
                # 영문: "+10%", "-5%" 패턴 (% 기호가 반드시 있어야 함)
                price_match = re.search(r'([+-]\d+(?:\.\d+)?)\s*%?', text)
                if price_match:
                    return Decimal(price_match.group(1))
            else:
                # 한글: "+10%에", "-5.5에" 패턴
                price_match = re.search(r'([+-]\d+(?:\.\d+)?)\s*(%|퍼센트)?\s*에?', text)
                if price_match:
                    return Decimal(price_match.group(1))
        except InvalidOperation:
            return None
        return None

    def _extract_relative_amount(self, text: str, is_english: bool) -> Optional[str]:
        """텍스트에서 상대적 수량을 추출"""
        if is_english:
            # 영문: "all", "50%" 패턴
            if 'all' in text.lower():
                return "100.0"
            
            percentage_match = re.search(r'![+-](\d+(?:\.\d+)?)\s*%', text)
            if percentage_match:
                return percentage_match.group(1)
        else:
            # 한글: "전부", "절반", "20%" 패턴
            if '전부' in text or '전량' in text:
                return "100.0"
            if '절반' in text or '반' in text:
                return "50.0"

            percentage_match = re.search(r'(?<![+-])(\d+\.?\d*)\s*(%|퍼센트)(?!에)', text)
            if percentage_match:
                return percentage_match.group(1)
        return None

    def _extract_order_type(self, text: str, is_english: bool) -> str:
        """주문 타입을 추출"""
        if is_english:
            # 영문: "market" 또는 "limit" 명시적 추출
            match = re.search(r'^(market|limit)', text.lower())
            if match:
                return match.group(1)
        
        # 기본값은 market (한글은 가격 조건에 따라 나중에 변경)
        return "market"

    def _process_english_tokens(self, text: str, entities: Dict[str, Any]) -> None:
        """영문 명령어의 토큰 기반 처리"""
        # order_type과 intent 제거 후 나머지 토큰 추출
        match = re.match(r'^(market|limit)?\s*(buy|sell)\s*', text.lower())
        if match:
            rest_of_text = text[match.end():]
        else:
            rest_of_text = text

        # 이미 추출된 패턴들 제거 (상대 가격 패턴도 제거)
        patterns_to_remove = [
            r'with\s*\d+(?:\.\d*)?\s*(?:usdt|krw)?',  # with X usdt/krw
            r'[+-]\d+(?:\.\d*)?\s*%?'  # +5%, -10 등
        ]
        
        for pattern in patterns_to_remove:
            rest_of_text = re.sub(pattern, '', rest_of_text, flags=re.IGNORECASE)

        # 토큰 분석
        tokens = [t for t in rest_of_text.split() if t]
        numbers = []
        potential_coins = []

        for token in tokens:
            try:
                if token.lower() == 'all':
                    if not entities.get('relative_amount'):
                        entities['relative_amount'] = '100.0'
                elif '%' in token and not re.match(r'^[+-]', token):
                    # +나 -로 시작하지 않는 %만 상대 수량으로 처리
                    value = token.replace('%', '')
                    if not entities.get('relative_amount'):
                        entities['relative_amount'] = value
                elif re.match(r'^\d+\.?\d*$', token):
                    numbers.append(Decimal(token))
                else:
                    potential_coins.append(token)
            except (InvalidOperation, ValueError):
                potential_coins.append(token)


        # 코인 심볼 확정 (마지막 유효한 코인 사용)
        if not entities.get('coin'):
            for pc in potential_coins:
                coin = self.find_closest_symbol(pc.upper())
                if coin:
                    entities['coin'] = coin

        # 숫자를 수량과 가격에 할당
        has_amount_spec = entities.get('total_cost') is not None or entities.get('relative_amount') is not None
        if numbers:
            if not has_amount_spec and not entities.get('amount'):
                entities['amount'] = numbers.pop(0)
            if numbers and not entities.get('price'):
                entities['price'] = numbers.pop(0)

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """주어진 텍스트에서 거래 관련 모든 엔터티를 통합 추출"""
        clean_input = clean_text(text)
        logging.info(f"Original text: '{text}', Cleaned text: '{clean_input}'")

        # 언어 구분
        is_english = self._is_english(clean_input)
        lang_type = "English" if is_english else "Korean"
        logging.info(f"{lang_type} command detected.")

        # 기본 엔터티 구조 초기화
        entities: Dict[str, Any] = {
            "intent": None, "coin": None, "amount": None, "price": None,
            "relative_price": None, "relative_amount": None, "total_cost": None,
            "current_price_order": False, "order_type": "market"
        }

        # 각 엔터티 추출 (언어별 로직 적용)
        entities["intent"] = self._extract_intent(clean_input, is_english)
        entities["coin"] = self._extract_coin(clean_input, is_english)
        entities["amount"] = self._extract_amount(clean_input, is_english)
        entities["price"] = self._extract_price(clean_input, is_english)
        entities["total_cost"] = self._extract_total_cost(clean_input, is_english)
        entities["current_price_order"] = self._extract_current_price_order(clean_input, is_english)
        entities["relative_price"] = self._extract_relative_price(clean_input, is_english)
        entities["relative_amount"] = self._extract_relative_amount(clean_input, is_english)
        entities["order_type"] = self._extract_order_type(clean_input, is_english)

        # 영문의 경우 토큰 기반 추가 처리
        if is_english:
            self._process_english_tokens(clean_input, entities)

        # 후처리: 조건부 로직 적용
        if entities["amount"] is not None:
            entities["relative_amount"] = None

        if (entities["price"] is not None or 
            entities["current_price_order"] or 
            entities["relative_price"] is not None):
            entities["order_type"] = "limit"
        print(entities)
        return entities


class TradeCommandParser:
    """
    추출된 엔티티를 파싱하여 최종적이고 검증된 `TradeCommand`를 구성합니다.
    상대 가격 계산, 총 비용을 수량으로 변환, 최종 주문 유형 결정과 같은 복잡한 로직을 처리합니다.
    """
    def __init__(self, extractor: EntityExtractor, portfolio_manager: PortfolioManager, executor: 'TradeExecutor'):
        self.extractor = extractor
        self.portfolio_manager = portfolio_manager
        self.executor = executor
        
    def parse(self, text: str) -> Optional[TradeCommand]:
        """주어진 텍스트를 파싱하여 TradeCommand 객체로 변환합니다."""
        entities = self.extractor.extract_entities(text)

        # 코인을 찾지 못했을 경우, 코인 목록을 갱신하고 다시 시도
        if not entities.get("coin"):
            logging.warning(f"코인을 찾지 못했습니다: '{text}'. 코인 목록을 갱신하고 다시 시도합니다.")
            self.extractor.refresh_coins(self.executor)
            entities = self.extractor.extract_entities(text)

        if not entities.get("intent") or not entities.get("coin"):
            logging.warning(
                f"Parse failed for text: '{text}'. "
                f"Missing intent ('{entities.get('intent')}') or coin ('{entities.get('coin')}')."
            )
            return None
        
        coin_symbol = str(entities["coin"]) if entities.get("coin") else None

        # 상대 가격 주문을 먼저 처리
        if entities.get("relative_price") is not None:
            if not coin_symbol:
                logging.error("상대 가격 주문은 반드시 코인이 명시되어야 합니다.")
                return None
            
            intent = str(entities["intent"])
            relative_price_percentage = entities["relative_price"]
            order_book = self.executor.get_order_book(coin_symbol)

            if order_book:
                base_price_num = order_book['bid'] if intent == 'buy' else order_book['ask']
                base_price = Decimal(str(base_price_num))
                calculated_price = base_price * (Decimal('1') + relative_price_percentage / Decimal('100'))
                entities['price'] = calculated_price
                logging.info(
                    f"상대 가격 주문: {coin_symbol}의 기준가({base_price}) 대비 {relative_price_percentage:+}% -> 지정가 {calculated_price} 설정"
                )
            else:
                logging.error(f"'{coin_symbol}'의 호가를 가져올 수 없어 상대 가격 주문을 처리할 수 없습니다.")
                return None

        # 암시적 현재가 주문 처리 (한글 '현재가' 또는 조건부 영어 limit 주문)
        elif entities.get("current_price_order") and entities.get("price") is None:
            if not coin_symbol:
                logging.error("현재가 주문은 반드시 코인이 명시되어야 합니다.")
                return None

            logging.info(f"암시적 현재가 주문 감지: {coin_symbol}")
            order_book = self.executor.get_order_book(coin_symbol)

            if order_book:
                # 매수는 매수 호가(bid), 매도는 매도 호가(ask)를 사용하는 것이 일반적
                price_to_set_num = order_book['bid'] if entities.get("intent") == 'buy' else order_book['ask']
                price_to_set = Decimal(str(price_to_set_num))
                entities['price'] = price_to_set
                logging.info(f"암시적 현재가 설정: 지정가를 {price_to_set}(으)로 설정")
            else:
                logging.error(f"'{coin_symbol}'의 호가를 가져올 수 없어 현재가 주문을 처리할 수 없습니다.")
                return None

        if entities.get("amount") is None and entities.get("relative_amount") is None and entities.get("total_cost") is None:
            logging.warning(
                f"Parse failed for text: '{text}'. "
                f"Missing amount information (e.g., '10개', '전부', '50%', '10000원어치', 'all', '50%')."
            )
            return None

        final_amount = entities.get("amount")
        total_cost = entities.get("total_cost")

        if total_cost is not None:
            if not coin_symbol:
                logging.error("비용 기반 주문은 반드시 코인이 명시되어야 합니다.")
                return None
            price_to_use = entities.get("price")

            if price_to_use is None:
                # For market orders, get current price. For limit orders, price should have been set.
                price_num = self.executor.get_current_price(coin_symbol)
                if price_num:
                    price_to_use = Decimal(str(price_num))


            if price_to_use is not None and price_to_use > Decimal('0'):
                final_amount = total_cost / price_to_use
                quote_currency = self.executor._get_quote_currency()
                logging.info(f"계산된 수량: {total_cost} {quote_currency} / {price_to_use} {quote_currency}/coin -> {final_amount} {coin_symbol}")
            else:
                logging.error(f"'{coin_symbol}'의 현재 가격을 가져올 수 없어 총 비용 기반 주문을 처리할 수 없습니다.")
                return None

        relative_amount_str = entities.get("relative_amount")
        if relative_amount_str:
            if not coin_symbol:
                logging.error("상대 수량 주문은 반드시 코인이 명시되어야 합니다.")
                return None

            # 상대 수량 계산 전에 잔고를 새로고침합니다.
            self.portfolio_manager.refresh_balance()
            
            current_holding = self.portfolio_manager.get_coin_amount(coin_symbol)
            if current_holding is None or current_holding <= Decimal('0'):
                logging.warning(f"상대 수량을 처리할 수 없습니다. '{coin_symbol}'의 보유량이 없거나 잔고 조회에 실패했습니다.")
                return None

            try:
                percentage = Decimal(relative_amount_str)
                calculated_amount = current_holding * (percentage / Decimal('100'))
                final_amount = calculated_amount
                logging.info(f"계산된 수량: {percentage}% of {current_holding} {coin_symbol} -> {final_amount} {coin_symbol}")
            except InvalidOperation:
                logging.error(f"잘못된 상대 수량 값입니다: '{relative_amount_str}'")
                return None

        if final_amount is not None and final_amount <= Decimal('0'):
            logging.warning(f"계산된 거래 수량이 0 이하({final_amount})이므로 거래를 진행할 수 없습니다.")
            return None

        market_symbol = f"{coin_symbol}/{self.executor.quote_currency}" if coin_symbol else None

        return TradeCommand(
            intent=str(entities["intent"]),
            symbol=market_symbol,
            amount=str(final_amount) if final_amount is not None else None,
            price=str(entities.get("price")) if entities.get("price") is not None else None,
            order_type=str(entities["order_type"]),
            total_cost=str(total_cost) if total_cost is not None else None
        )


class TradeExecutor:
    """
    `TradeCommand`를 거래소에 대해 실행합니다.
    이 클래스는 `ccxt` 라이브러리와 상호 작용하여 실제 주문을 생성하고,
    가격을 조회하며, 거래소와의 연결을 관리하는 역할을 담당합니다.
    """

    def __init__(self, exchange: Exchange, config: Dict[str, Any]):
        self.exchange = exchange
        self.config = config
        self.quote_currency = self._get_quote_currency()

    def _get_quote_currency(self) -> str:
        """설정에서 현재 거래소의 기본 통화를 가져옵니다."""
        try:
            return self.config['exchange_settings'][self.exchange.id]['quote_currency']
        except KeyError:
            logging.warning(
                f"'{self.exchange.id}'에 대한 'quote_currency' 설정이 없습니다. "
                f"기본값으로 'USDT'를 사용합니다."
            )
            return "USDT"

    def get_current_price(self, coin_symbol: str) -> Optional[Num]:
        """지정된 코인의 현재 가격을 가져옵니다."""
        quote_currency = self._get_quote_currency()
        market_symbol = f'{coin_symbol}/{quote_currency}'
        try:
            ticker = self.exchange.fetch_ticker(market_symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Could not fetch price for {market_symbol}: {e}")
            return None

    def get_order_book(self, coin_symbol: str) -> Optional[Dict[str, Num]]:
        """지정된 코인의 오더북을 가져와 1호가(매수/매도)를 반환합니다."""
        quote_currency = self._get_quote_currency()
        market_symbol = f'{coin_symbol}/{quote_currency}'
        try:
            order_book = self.exchange.fetch_order_book(market_symbol, limit=1)
            if order_book['bids'] and order_book['asks']:
                best_bid = order_book['bids'][0][0]
                best_ask = order_book['asks'][0][0]
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

        if not command.symbol or not command.amount:
            logging.error("거래를 실행하려면 symbol과 amount가 반드시 필요합니다.")
            return {"status": "error", "message": "Symbol or amount is missing"}

        try:
            # create_order 메서드에 필요한 파라미터들을 준비합니다.
            symbol = command.symbol
            order_type = command.order_type
            side = command.intent
            amount = float(command.amount)
            price = float(command.price) if command.price else None

            # 실제 주문을 실행합니다.
            logging.info(f"Placing order: {side} {amount} {symbol} at price {price}")
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            logging.info(f"Successfully placed order: {order}")

            return {
                "status": "success",
                "order_details": order
            }

        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error while executing command: {e}")
            return {"status": "error", "message": f"Exchange error: {e}"}
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}


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
        return {}
    except json.JSONDecodeError:
        logging.error(f"비밀 설정 파일의 형식이 올바르지 않습니다: {secrets_path}")
        raise


def initialize_exchange(exchange_id: str, config: Dict[str, Any], use_testnet: bool = False) -> Exchange:
    """거래소 ID와 설정에 따라 ccxt 거래소 인스턴스를 생성하고 초기화합니다."""
    exchange_id_for_ccxt = exchange_id.replace('_testnet', '')
    is_testnet = exchange_id.endswith('_testnet') or use_testnet

    try:
        exchange_class = getattr(ccxt, exchange_id_for_ccxt)

        # API 키 설정을 config에서 가져옵니다.
        exchange_config = config.get('exchanges', {}).get(exchange_id, {})
        api_key = exchange_config.get('api_key')
        secret_key = exchange_config.get('secret_key')

        exchange_params = {
            'timeout': 30000,
        }
        if api_key and secret_key:
            exchange_params['apiKey'] = api_key
            exchange_params['secret'] = secret_key

        exchange = exchange_class(exchange_params)

        if is_testnet:
            exchange.set_sandbox_mode(True)
            logging.info(f"Initialized {exchange_id_for_ccxt.capitalize()} in testnet mode.")
        else:
            logging.info(f"Initialized {exchange_id_for_ccxt.capitalize()} in mainnet mode.")

        return exchange
    except Exception as e:
        logging.error(f"Failed to initialize exchange {exchange_id}: {e}")
        raise


def fetch_exchange_coins(exchange: Exchange) -> List[str]:
    """ccxt를 사용하여 지정된 거래소의 모든 코인 목록을 가져옵니다."""
    try:
        logging.info(f"Fetching coin list from {exchange.id.capitalize()}...")
        markets = exchange.load_markets()
        base_coins = [market['base'] for market in markets.values()]
        unique_base_coins = sorted(list(set(base_coins)))
        logging.info(f"Successfully fetched {len(unique_base_coins)} unique coins from {exchange.id.capitalize()}.")
        return unique_base_coins
    except Exception as e:
        logging.error(f"Failed to fetch coin list from {exchange.id.capitalize()}: {e}")
        raise


def setup_trader(exchange_id: str, config: Dict[str, Any]) -> Tuple[TradeCommandParser, TradeExecutor, Exchange]:
    """거래 관련 객체들을 초기화하고 설정합니다."""
    use_testnet = exchange_id.endswith('_testnet')

    exchange = initialize_exchange(exchange_id, config, use_testnet)
    exchange_coins = fetch_exchange_coins(exchange)
    config["coins"] = exchange_coins

    portfolio_manager = PortfolioManager(exchange)
    executor = TradeExecutor(exchange, config)
    extractor = EntityExtractor(config)
    parser = TradeCommandParser(extractor, portfolio_manager, executor)

    return parser, executor, exchange


def main():
    """메인 실행 함수: 설정을 로드하고 대화형으로 명령을 처리합니다."""
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)
    secrets_path = Path(__file__).parent / "secrets.json"
    secrets = load_secrets(secrets_path)
    config.update(secrets)

    default_exchange_id = config.get("default_exchange", "binance")
    available_exchanges = list(config.get("exchange_settings", {}).keys())

    try:
        parser, executor, _ = setup_trader(default_exchange_id, config)
        current_exchange_id = default_exchange_id
    except Exception as e:
        logging.error(f"거래기 초기화 실패: {e}")
        return

    print(f"--- NLP Trade-bot 시작 (현재 거래소: {current_exchange_id}) ---" + " (종료하려면 'exit' 또는 'quit' 입력)")
    print(f"사용 가능한 거래소: {available_exchanges}")
    print("거래소 변경: /exchange [거래소 이름]")

    while True:
        try:
            text = input(f"\n[{current_exchange_id}]> ")
            if text.lower() in ['exit', 'quit']:
                break

            if text.lower().startswith("/exchange "):
                new_exchange_id = text.split(" ")[1].strip()
                if new_exchange_id in available_exchanges:
                    try:
                        parser, executor, _ = setup_trader(new_exchange_id, config)
                        current_exchange_id = new_exchange_id
                        print(f"거래소가 {current_exchange_id}(으)로 변경되었습니다.")
                    except Exception as e:
                        logging.error(f"{new_exchange_id}로 거래소 전환 실패: {e}")
                else:
                    print(f"지원하지 않는 거래소입니다: {new_exchange_id}. 사용 가능한 거래소: {available_exchanges}")
                continue

            command = parser.parse(text)
            if command:
                result = executor.execute(command)
                print(f"[실행 결과]: {json.dumps(result, indent=2, ensure_ascii=False)}")
            else:
                print("[실행 결과]: 명령을 해석하지 못했습니다.")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            logging.error(f"오류 발생: {e}")

    print("\n--- NLP Trade-bot 종료 ---")


if __name__ == '__main__':
    main()