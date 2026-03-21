# LLM Чат

Streamlit-чат с поддержкой любых LLM через litellm (OpenAI, Anthropic, HuggingFace и др.).

## Установка

```bash
uv sync
cp .env.example .env
```

## Настройка `.env`

```env
LLM_MODEL=huggingface/meta-llama/Llama-4-Scout-17B-16E-Instruct
# LLM_MODEL=claude-haiku-4-5
# LLM_MODEL=gpt-4o

API_KEY=your_key_here
```

## Запуск

```bash
uv run streamlit run app.py
```

## Линтинг

```bash
uv run ruff check . --fix
uv run black .
```