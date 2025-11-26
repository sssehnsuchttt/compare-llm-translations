import csv
import json
import os
import time
from typing import TypedDict, Any
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from rich import box
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

load_dotenv()


class Backend(str, Enum):
    OPENROUTER = "openrouter"
    NEUROAPI = "neuroapi"


_clients_cache: dict[Backend, OpenAI] = {}


def make_openai_client(backend: Backend) -> OpenAI:
    """Создаёт OpenAI клиент для заданного backend.

    Поддерживаемые значения: "openrouter", "neuroapi".
    Для каждого бэкенда читаются свои переменные окружения для ключа и base_url.
    """
    
    if backend == Backend.OPENROUTER:
        key = os.getenv("OPEN_ROUTER_API_KEY")
        base = os.getenv("OPEN_ROUTER_API_BASE_URL")
        if not key:
            raise RuntimeError(
                "Требуется OPEN_ROUTER_API_KEY для openrouter"
            )
        return OpenAI(api_key=key, base_url=base)

    if backend == Backend.NEUROAPI:
        key = os.getenv("NEURO_API_KEY")
        base = os.getenv("NEURO_API_BASE_URL")
        if not key:
            raise RuntimeError(
                "Требуется NEURO_API_KEY для neuroapi"
            )
        return OpenAI(api_key=key, base_url=base)

    raise RuntimeError(f"Неизвестный backend: {backend}")


class ModelTranslation(TypedDict):
    model: str
    text: str


class DatasetItem(TypedDict):
    id: int
    original_text: str
    translations: list[ModelTranslation]


class TranslationRating(TypedDict):
    term_preservation: float
    language_naturalness: float
    meaning_accuracy: float


class RatedTranslation(TypedDict):
    id: int
    model: str
    original_text: str
    translated_text: str
    ratings: TranslationRating


# С neuroapi не работал grok, а с openrouter не работал gemini, поэтому реализована возможность выбирать бэкенд для каждой модели
MODELS: dict[str, dict[str, Any]] = {
    "gpt-5-nano": {"model": "gpt-5-nano", "backend": Backend.NEUROAPI},
    "grok-4.1-fast": {"model": "x-ai/grok-4.1-fast:free", "backend": Backend.OPENROUTER},
    "gemini-2.0-flash-lite": {"model": "gemini-2.5-flash-lite", "backend": Backend.NEUROAPI},
}

SYSTEM_PROMPT = """
You are a professional technical translator from English to Russian,
specializing in software engineering, AI/ML, and technical documentation.

Your task:
- Translate the input text from English to natural, fluent Russian.
- Preserve all technical terms, identifiers, code fragments, and product names.
- Keep the original meaning accurate, without adding or omitting information.
- If a technical term has a common Russian equivalent, use it; otherwise keep the English term.
- Do not explain the translation and do not add comments.
- Output only the translated text, without quotes or any extra formatting.
"""


def translate_dataset(console: Console) -> list[DatasetItem]:
    """
    Проходит по всем записям в dataset.csv и получает переводы от всех моделей.
    Возвращает список с оригинальными текстами и переводами.
    """
    translations: list[DatasetItem] = []

    with open("dataset.csv", "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total_tasks = len(rows) * len(MODELS)

    with Progress(
        TextColumn("{task.description} {task.completed}/{task.total}"),
        BarColumn(
            bar_width=60,
            style="grey23",
            complete_style="green",
            finished_style="bright_green",
        ),
        console=console,
    ) as progress:
        task = progress.add_task("Обработка датасета...", total=total_tasks)

        for row in rows:
            item: DatasetItem = {
                "id": int(row["id"]),
                "original_text": row["text"],
                "translations": [],
            }

            for model_name, info in MODELS.items():
                model_id = info["model"]
                backend = info.get("backend", Backend.OPENROUTER)

                client = _clients_cache.get(backend)
                if client is None:
                    client = make_openai_client(backend)
                    _clients_cache[backend] = client

                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": row["text"]},
                        ],
                    )
                except OpenAIError as e:
                    # просто падаем и завершаем скрипт
                    raise RuntimeError(f"Ошибка API для модели {model_name}: {e}")

                message = response.choices[0].message
                if not message or not message.content:
                    console.print(
                        f"[red]Пустой ответ от модели {model_name}[/red]"
                    )
                    raise RuntimeError("Пустой ответ от модели")

                translated_text = message.content

                item["translations"].append(
                    {
                        "model": model_name,
                        "text": translated_text,
                    }
                )

                progress.advance(task)

            translations.append(item)

    return translations


def _ask_rating(
    console: Console,
    label: str,
    min_value: int = 0,
    max_value: int = 10,
) -> float:
    while True:
        console.print(
            f"[bold]{label}[/bold] " f"([green]{min_value}–{max_value}[/green])"
        )

        value = IntPrompt.ask(
            "[cyan]Введите оценку[/cyan]",
            console=console,
        )

        if min_value <= value <= max_value:
            return float(value) / float(max_value)

        console.print(
            f"[red]Ошибка:[/red] введите целое число "
            f"от {min_value} до {max_value}.\n"
        )


def rate_translation(
    console: Console,
    item_id: int,
    original_text: str,
    translated_text: str,
    model_name: str,
) -> RatedTranslation:
    """Интерактивно оценивает перевод по трём критериям.
    """
    console.clear()

    console.rule(f"Оценка качества перевода модели [bold]{model_name}[/bold]")

    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        highlight=True,
    )

    table.add_column("ID", justify="center", no_wrap=True, style="cyan")
    table.add_column("Оригинальный текст", style="magenta")
    table.add_column("Переведенный текст", style="green")

    table.add_row(str(item_id), original_text, translated_text)

    console.print(table)

    console.rule(
        "[bold]Оцените перевод по трём критериям по шкале от 0 до 10[/bold]",
        style="cyan",
    )

    term_preservation = _ask_rating(
        console,
        "\nСохранение технических терминов",
    )

    language_naturalness = _ask_rating(
        console,
        "\nЕстественность языка",
    )

    meaning_accuracy = _ask_rating(
        console,
        "\nТочность передачи смысла",
    )

    rating: TranslationRating = {
        "term_preservation": term_preservation,
        "language_naturalness": language_naturalness,
        "meaning_accuracy": meaning_accuracy,
    }

    rated_translation: RatedTranslation = {
        "id": item_id,
        "model": model_name,
        "original_text": original_text,
        "translated_text": translated_text,
        "ratings": rating,
    }

    return rated_translation


def show_summary_table(console: Console, ratings: list[RatedTranslation]) -> None:
    """
    Отображает итоговую таблицу с усреднёнными результатами по моделям.
    Итоговая оценка рассчитывается как среднее арифметическое трёх критериев.
    """
    summary: dict[str, dict[str, float | int]] = {}

    for r in ratings:
        model = r["model"]
        if model not in summary:
            summary[model] = {
                "count": 0,
                "term_sum": 0.0,
                "natural_sum": 0.0,
                "meaning_sum": 0.0,
            }

        s = summary[model]
        s["count"] += 1
        s["term_sum"] += r["ratings"]["term_preservation"]
        s["natural_sum"] += r["ratings"]["language_naturalness"]
        s["meaning_sum"] += r["ratings"]["meaning_accuracy"]

    table = Table(
        title="Итоговые результаты по моделям",
        box=box.SIMPLE_HEAD,
        expand=True,
        highlight=True,
    )

    table.add_column("Модель", style="cyan", no_wrap=True)
    table.add_column("Термины (avg)", justify="right")
    table.add_column("Естеств. (avg)", justify="right")
    table.add_column("Смысл (avg)", justify="right")
    table.add_column("Общая оценка", justify="right")

    for model, s in summary.items():
        count = s["count"] or 1
        term_avg = s["term_sum"] / count
        natural_avg = s["natural_sum"] / count
        meaning_avg = s["meaning_sum"] / count
        overall = (term_avg + natural_avg + meaning_avg) / 3.0

        def fmt_percent(x: float) -> str:
            return f"{x * 100:.2f}".rstrip("0").rstrip(".") + "%"

        table.add_row(
            model,
            fmt_percent(term_avg),
            fmt_percent(natural_avg),
            fmt_percent(meaning_avg),
            fmt_percent(overall),
        )

    console.clear()
    console.rule("[bold]Итоговая таблица[/bold]", style="cyan")
    console.print(table)


def main():
    console = Console()

    translations = translate_dataset(console)

    with open("translations.json", "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)

    all_ratings: list[RatedTranslation] = []

    for item in translations:
        for model_translation in item["translations"]:
            rating = rate_translation(
                console=console,
                item_id=item["id"],
                original_text=item["original_text"],
                translated_text=model_translation["text"],
                model_name=model_translation["model"],
            )
            all_ratings.append(rating)

    show_summary_table(console, all_ratings)

    output_path = "ratings.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_ratings, f, ensure_ascii=False, indent=2)

    console.print(
        f"\n[green]Результаты сохранены в файле[/green] [bold]{output_path}[/bold]"
    )


if __name__ == "__main__":
    main()
