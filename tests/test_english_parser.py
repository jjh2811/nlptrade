import re
from unittest.mock import MagicMock

import pytest

from src.nlptrade.nlptrade import EntityExtractor, TradeCommand, TradeCommandParser



# Mock objects for dependencies
@pytest.fixture
def mock_portfolio_manager():
    return MagicMock()


@pytest.fixture
def mock_trade_executor(config):
    executor = MagicMock()
    executor.get_order_book.return_value = {'bid': 50000, 'ask': 50050}
    executor.get_current_price.return_value = 50025
    executor.quote_currency = config["quote_currency"]
    return executor


@pytest.fixture
def config():
    """Provides a basic configuration for tests."""
    return {
        "coins": ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "DOT", "LTC", "BNB", "USDT", "KRE"],
        "intent_map": {
            "매수": "buy",
            "사": "buy",
            "매도": "sell",
            "팔아": "sell"
        },
        "custom_mapping": {
            "비트코인": "BTC",
            "이더리움": "ETH",
            "솔라나": "SOL",
            "리플": "XRP",
            "도지": "DOGE",
            "에이다": "ADA",
            "폴카닷": "DOT",
            "라이트코인": "LTC",
            "바이낸스코인": "BNB",
            "테더": "USDT",
            "크레이": "KRE"
        },
        "quote_currency": "USDT"
    }


@pytest.fixture
def parser(config, mock_portfolio_manager, mock_trade_executor):
    """Initializes the TradeCommandParser with mocked dependencies."""
    extractor = EntityExtractor(config)
    return TradeCommandParser(extractor, mock_portfolio_manager, mock_trade_executor)


# Test cases based on the user's examples
@pytest.mark.parametrize("command_text, expected", [
    ("market buy 1 btc", TradeCommand(intent='buy', order_type='market', symbol='BTC/USDT', amount=1.0, price=None, total_cost=None)),
    ("limit sell 2 eth 2500", TradeCommand(intent='sell', order_type='limit', symbol='ETH/USDT', amount=2.0, price=2500.0, total_cost=None)),
    ("market sell all sol", TradeCommand(intent='sell', order_type='market', symbol='SOL/USDT', amount=None, price=None, total_cost=None)),  # Amount will be calculated later
    ("limit sell 30% xrp 0.6", TradeCommand(intent='sell', order_type='limit', symbol='XRP/USDT', amount=None, price=0.6, total_cost=None)),  # Amount will be calculated later
    ("market sell 50% doge", TradeCommand(intent='sell', order_type='market', symbol='DOGE/USDT', amount=None, price=None, total_cost=None)),  # Amount will be calculated later
    ("limit buy 0.1 btc -5%", TradeCommand(intent='buy', order_type='limit', symbol='BTC/USDT', amount=0.1, price=47500.0, total_cost=None)),  # 50000 * (1 - 0.05)
    ("limit sell 1 eth +10%", TradeCommand(intent='sell', order_type='limit', symbol='ETH/USDT', amount=1.0, price=55055.0, total_cost=None)),  # 50050 * (1 + 0.10)
    ("market buy btc with 10 usdt", TradeCommand(intent='buy', order_type='market', symbol='BTC/USDT', amount=10 / 50025, price=None, total_cost=10.0)),
    ("market sell 0.01 btc", TradeCommand(intent='sell', order_type='market', symbol='BTC/USDT', amount=0.01, price=None, total_cost=None)),
    ("market sell all eth", TradeCommand(intent='sell', order_type='market', symbol='ETH/USDT', amount=None, price=None, total_cost=None)),
    ("market buy xrp with 1000 krw", TradeCommand(intent='buy', order_type='market', symbol='XRP/USDT', amount=1000 / 50025, price=None, total_cost=1000.0)),
    ("limit sell 20% btc +4%", TradeCommand(intent='sell', order_type='limit', symbol='BTC/USDT', amount=None, price=52052.0, total_cost=None)),  # 50050 * (1 + 0.04)
    ("limit buy 50 doge -7%", TradeCommand(intent='buy', order_type='limit', symbol='DOGE/USDT', amount=50.0, price=46500.0, total_cost=None)),  # 50000 * (1 - 0.07)
    ("limit sell 1 eth +15", TradeCommand(intent='sell', order_type='limit', symbol='ETH/USDT', amount=1.0, price=50050 * (1 + 0.15), total_cost=None)),
    ("market buy 2 sol", TradeCommand(intent='buy', order_type='market', symbol='SOL/USDT', amount=2.0, price=None, total_cost=None)),
    ("limit sell 5 ada 0.45", TradeCommand(intent='sell', order_type='limit', symbol='ADA/USDT', amount=5.0, price=0.45, total_cost=None)),
    ("limit buy 1 kre 400", TradeCommand(intent='buy', order_type='limit', symbol='KRE/USDT', amount=1.0, price=400.0, total_cost=None)),
    ("market buy btc with 50", TradeCommand(intent='buy', order_type='market', symbol='BTC/USDT', amount=50 / 50025, price=None, total_cost=50.0)),
    ("market sell 50% btc", TradeCommand(intent='sell', order_type='market', symbol='BTC/USDT', amount=None, price=None, total_cost=None)),
])
def test_english_command_parsing(parser, mock_portfolio_manager, command_text, expected):
    if '%' in command_text and 'with' not in command_text and expected.amount is None:
        mock_portfolio_manager.get_coin_amount.return_value = 10.0
        match = re.search(r'(\d+\.?\d*)\s*%', command_text)
        if match:
            percentage = float(match.group(1))
            expected.amount = 10.0 * (percentage / 100.0)

    if 'all' in command_text and expected.symbol:
        mock_portfolio_manager.get_coin_amount.return_value = 10.0
        expected.amount = 10.0

    result = parser.parse(command_text)

    assert result is not None
    assert result.intent == expected.intent
    assert result.order_type == expected.order_type
    assert result.symbol == expected.symbol
    assert result.total_cost == expected.total_cost

    if expected.price is not None:
        assert result.price == pytest.approx(expected.price)
    else:
        assert result.price is None

    if expected.amount is not None:
        assert result.amount == pytest.approx(expected.amount)
    else:
        if '%' in command_text or 'all' in command_text:
            pass
        else:
            assert result.amount is None
