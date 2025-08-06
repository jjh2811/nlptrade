import pytest

from nlptrade.nlptrade import (
    EntityExtractor,
    TradeCommand,
    TradeCommandParser,
    TradeExecutor,
)
from nlptrade.portfolio import PortfolioManager


@pytest.fixture
def config():
    """테스트용 설정을 반환하는 픽스처"""
    return {
        "coins": ["BTC", "ETH", "XRP", "DOGE", "SOL"],
        "intent_map": {
            "매수": "buy", "구매": "buy", "사줘": "buy",
            "매도": "sell", "판매": "sell", "팔아": "sell"
        },
        "custom_mapping": {
            "BCT": "BTC", "BTCC": "BTC", "BTCUSDT": "BTC",
            "비트코인": "BTC", "이더리움": "ETH", "이더": "ETH",
            "리플": "XRP", "도지": "DOGE", "도지코인": "DOGE", "솔라나": "SOL"
        },
        "quote_currency": "USDT"
    }


@pytest.fixture
def extractor(config):
    """EntityExtractor 인스턴스를 반환하는 픽스처"""
    return EntityExtractor(config)


@pytest.fixture
def portfolio_manager(mocker):
    """PortfolioManager의 모의(mock) 객체를 반환하는 픽스처"""
    mock_pm = mocker.Mock(spec=PortfolioManager)
    return mock_pm


@pytest.fixture
def executor(mocker, config):
    """TradeExecutor의 모의(mock) 객체를 반환하는 픽스처"""
    mock_executor = mocker.Mock(spec=TradeExecutor)
    mock_executor.get_order_book.return_value = {'ask': 50000.0, 'bid': 49999.0}
    mock_executor.get_current_price.return_value = 50000.0
    mock_executor.quote_currency = config["quote_currency"]
    return mock_executor


@pytest.fixture
def parser(extractor, portfolio_manager, executor):
    """의존성이 주입된 TradeCommandParser 인스턴스를 반환하는 픽스처"""
    return TradeCommandParser(extractor, portfolio_manager, executor)


# 파싱 성공 케이스
@pytest.mark.parametrize("input_text, expected_command", [
    # 기본 케이스
    ("비트코인 10개 사줘", TradeCommand(intent='buy', symbol='BTC/USDT', amount=10.0, price=None, order_type='market', total_cost=None)),
    ("이더 3개 4000달러에 매도해줘", TradeCommand(intent='sell', symbol='ETH/USDT', amount=3.0, price=4000.0, order_type='limit', total_cost=None)),
    ("리플 100개 시장가 매수", TradeCommand(intent='buy', symbol='XRP/USDT', amount=100.0, price=None, order_type='market', total_cost=None)),
    ("XRP 50개 구매", TradeCommand(intent='buy', symbol='XRP/USDT', amount=50.0, price=None, order_type='market', total_cost=None)),
    ("도지코인 10000개 팔아", TradeCommand(intent='sell', symbol='DOGE/USDT', amount=10000.0, price=None, order_type='market', total_cost=None)),

    # 커스텀 매핑 및 오타 교정 케이스
    ("BCT 1개 매수해줘", TradeCommand(intent='buy', symbol='BTC/USDT', amount=1.0, price=None, order_type='market', total_cost=None)),
    ("솔라나 5개 150달러에 지정가 구매", TradeCommand(intent='buy', symbol='SOL/USDT', amount=5.0, price=150.0, order_type='limit', total_cost=None)),

    # 특수문자 및 공백 처리 케이스
    ("  DOGE   500개를 😊 매수해줘  ", TradeCommand(intent='buy', symbol='DOGE/USDT', amount=500.0, price=None, order_type='market', total_cost=None)),

    # 현재가 지정가 주문 테스트
    ("비트코인 현재가에 10개 매수", TradeCommand(intent='buy', symbol='BTC/USDT', amount=10.0, price=49999.0, order_type='limit', total_cost=None)),

    # 상대 가격 지정가 주문 테스트
    ("BTC 1개를 +10%에 매도", TradeCommand(intent='sell', symbol='BTC/USDT', amount=1.0, price=50000 * 1.1, order_type='limit', total_cost=None)),
    ("BTC 1개를 -10%에 매수", TradeCommand(intent='buy', symbol='BTC/USDT', amount=1.0, price=49999 * 0.9, order_type='limit', total_cost=None)),
])
def test_parse_success(parser, input_text, expected_command):
    """다양한 성공 케이스에 대해 파싱이 정상적으로 동작하는지 테스트합니다."""
    result_command = parser.parse(input_text)
    assert result_command == expected_command


# 비용 기반 수량 계산 테스트
def test_parse_cost_based_amount(parser):
    """비용(e.g., 1000달러어치)을 기반으로 매수/매도 수량이 정확히 계산되는지 테스트합니다."""
    input_text = "비트코인 1000달러어치 사줘"
    expected_command = TradeCommand(intent='buy', symbol='BTC/USDT', amount=0.02, price=None, order_type='market', total_cost=1000.0)

    result_command = parser.parse(input_text)

    assert result_command is not None
    assert result_command.intent == expected_command.intent
    assert result_command.symbol == expected_command.symbol
    assert result_command.total_cost == expected_command.total_cost
    assert result_command.amount == pytest.approx(expected_command.amount)
    assert result_command.order_type == expected_command.order_type


# 파싱 실패 케이스
@pytest.mark.parametrize("input_text", [
    "XYZ 50개를 매수해",
    "매수해줘",
    "비트코인",
    "이건 그냥 문장입니다",
    "",
    "이더리움 팔아줘",
])
def test_parse_failure(parser, input_text):
    """필수 정보가 누락되거나 유효하지 않은 경우 None을 반환하는지 테스트합니다."""
    result_command = parser.parse(input_text)
    assert result_command is None


# 상대 수량 파싱 성공 케이스
@pytest.mark.parametrize("input_text, mock_balance, expected_intent, expected_amount", [
    ("비트코인 전부 매도", {"BTC": 0.5}, "sell", 0.5),
    ("이더리움 50% 매도", {"ETH": 10.0}, "sell", 5.0),
    ("도지코인 절반 팔아", {"DOGE": 1000.0}, "sell", 500.0),
    ("솔라나 25퍼센트 매도", {"SOL": 4.0}, "sell", 1.0),
])
def test_parse_relative_amount_success(parser, portfolio_manager, input_text, mock_balance, expected_intent, expected_amount):
    """보유 자산 기반의 상대적 수량(전부, 50%, 절반 등) 파싱을 테스트합니다."""
    coin_symbol = list(mock_balance.keys())[0]
    balance_amount = list(mock_balance.values())[0]
    portfolio_manager.get_coin_amount.return_value = balance_amount

    result_command = parser.parse(input_text)

    portfolio_manager.get_coin_amount.assert_called_once_with(coin_symbol)
    assert result_command is not None
    assert result_command.intent == expected_intent
    assert result_command.symbol == f"{coin_symbol}/USDT"
    assert result_command.amount == pytest.approx(expected_amount)
    assert result_command.order_type == 'market'


# 상대 수량 파싱 실패 케이스 (보유량 0)
def test_parse_relative_amount_failure_no_balance(parser, portfolio_manager):
    """보유량이 0일 때 상대 수량 파싱이 실패하는지 테스트합니다."""
    portfolio_manager.get_coin_amount.return_value = 0.0
    result_command = parser.parse("비트코인 전부 매도")
    assert result_command is None


# EntityExtractor 단위 테스트
@pytest.mark.parametrize("input_symbol, expected", [
    ("BTC", "BTC"),
    ("bct", "BTC"),
    ("비트코인", "BTC"),
    ("이더", "ETH"),
    ("XYZ", None),
    ("", None),
    ("BTCUSDT", "BTC")
])
def test_find_closest_symbol(extractor, input_symbol, expected):
    """코인 심볼 찾기 기능이 정확하게 동작하는지 테스트합니다."""
    assert extractor.find_closest_symbol(input_symbol) == expected


# 실행기(Executor) 테스트
def test_executor(executor):
    """실행기가 주어진 명령을 받아 표준 형식의 결과를 반환하는지 테스트합니다."""
    command = TradeCommand(intent='buy', symbol='BTC/USDT', amount=1.0, price=None, order_type='market', total_cost=None)

    expected_result = {
        "status": "success",
        "command_executed": command.__dict__
    }
    executor.execute.return_value = expected_result
    result = executor.execute(command)
    assert result == expected_result
    executor.execute.assert_called_once_with(command)
