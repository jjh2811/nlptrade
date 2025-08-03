import asyncio
import json
import logging
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from nlptrade.nlptrade import (
    load_config,
    load_secrets,
    fetch_exchange_coins,
    EntityExtractor,
    TradeCommandParser,
    TradeExecutor,
    PortfolioManager,
)

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# 전역 변수로 파서와 실행기 저장 (애플리케이션 컨텍스트를 통해 전달하는 것이 더 나은 방법일 수 있음)
parser: TradeCommandParser
executor: TradeExecutor

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start 명령어 핸들러"""
    if not update.message:
        return
    await update.message.reply_html(
        "안녕하세요! NLP 기반 트레이딩 봇입니다.\n"
        "거래 명령을 자연어 문장으로 입력해주세요.\n"
        "예: <i>비트코인 10개 사줘</i>\n"
        "도움이 필요하시면 /help 를 입력하세요."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help 명령어 핸들러"""
    if not update.message:
        return
    await update.message.reply_html(
        "<b>지원하는 명령어 형식:</b>\n"
        "- <code>[코인 이름] [수량] [매수/매도]</code> (예: 비트코인 10개 사줘)\n"
        "- <code>[코인 이름] [수량] [가격] [매수/매도]</code> (예: 이더리움 3개 4000달러에 팔아줘)\n"
        "- <code>[코인 이름] [비용]어치 [매수/매도]</code> (예: 리플 5000원어치 매수)\n"
        "- <code>[코인 이름] [상대 수량] [매도]</code> (예: 도지코인 전부 매도)\n"
        "- <code>[코인 이름] 현재가에 [수량] [매수/매도]</code> (예: 솔라나 현재가에 5개 구매)\n\n"
        "<b>코인 이름:</b> BTC, 이더리움, XRP 등 지원하는 코인의 영문/한글 이름\n"
        "<b>수량:</b> '10개', '0.5개' 등\n"
        "<b>가격:</b> '50000달러에', '600만원에' 등\n"
        "<b>상대 수량:</b> '전부', '절반', '50%' 등 (매도에만 적용)\n"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """모든 텍스트 메시지를 처리하는 핸들러"""
    global parser, executor
    if not update.message or not update.message.text:
        return

    text = update.message.text
    logger.info(f"Received message: {text}")

    command = parser.parse(text)

    if command:
        try:
            result = executor.execute(command)
            # JSON 응답을 보기 좋게 포맷팅하여 전송
            response_text = f"<pre>{json.dumps(result, indent=2, ensure_ascii=False)}</pre>"
            await update.message.reply_html(response_text)
        except Exception as e:
            logger.error(f"Error executing command: {command}, Error: {e}")
            await update.message.reply_text(f"명령 실행 중 오류가 발생했습니다: {e}")
    else:
        await update.message.reply_text("명령을 이해하지 못했습니다. /help 를 참고하세요.")

def main() -> None:
    """텔레그램 봇을 시작하고 실행합니다."""
    global parser, executor

    # 설정 로드
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)
    secrets_path = Path(__file__).parent / "secrets.json"
    secrets = load_secrets(secrets_path)
    config.update(secrets)

    telegram_token = config.get("telegram_token")
    if not telegram_token or telegram_token == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("텔레그램 봇 토큰이 secrets.json 파일에 설정되지 않았습니다.")
        return

    # nlptrade 컴포넌트 초기화
    default_exchange_id = config.get("default_exchange", "binance")
    try:
        exchange_coins = fetch_exchange_coins(default_exchange_id)
        config["coins"] = exchange_coins
    except Exception:
        logger.error(f"{default_exchange_id.capitalize()}에서 코인 목록을 가져올 수 없어 봇을 시작할 수 없습니다.")
        return

    portfolio_manager = PortfolioManager(default_exchange_id, config)
    executor = TradeExecutor(default_exchange_id, config)
    extractor = EntityExtractor(config)
    parser = TradeCommandParser(extractor, portfolio_manager, executor)

    # 텔레그램 봇 애플리케이션 생성
    application = Application.builder().token(telegram_token).build()

    # 핸들러 등록
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 봇 실행
    logger.info("Telegram bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()
