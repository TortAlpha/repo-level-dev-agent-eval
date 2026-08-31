"""Model token pricing for the cost_estimate metric.

Prices live in ``static/pricing.csv`` (USD per 1M tokens, input/output), pulled
from the OpenRouter models API — the authoritative source for what a run on
OpenRouter actually costs. Runs record only token counts; cost is derived here.

Refresh the table from OpenRouter (network) with:

    python -m src.metrics.pricing
"""

from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PRICING_CSV = Path(__file__).resolve().parents[2] / "static" / "pricing.csv"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float
    cache_read_input_per_1m: float | None = None
    cache_write_input_per_1m: float | None = None


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def load_prices(path: Path = PRICING_CSV) -> dict[str, ModelPrice]:
    if not path.is_file():
        return {}
    prices: dict[str, ModelPrice] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cache_read = row.get("cache_read_input_per_1m")
            cache_write = row.get("cache_write_input_per_1m")
            prices[row["model_id"]] = ModelPrice(
                float(row["input_per_1m"]),
                float(row["output_per_1m"]),
                _optional_float(cache_read),
                _optional_float(cache_write),
            )
    return prices


PRICES: dict[str, ModelPrice] = load_prices()


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    prices: dict[str, ModelPrice] | None = None,
    override: ModelPrice | None = None,
    cached_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
) -> float | None:
    """USD cost of one run's tokens, or ``None`` if the model has no price."""
    price = override or (PRICES if prices is None else prices).get(model)
    if price is None:
        return None
    cache_reads = min(max(cached_input_tokens, 0), input_tokens)
    cache_writes = min(
        max(cache_write_input_tokens, 0), max(input_tokens - cache_reads, 0)
    )
    uncached = max(input_tokens - cache_reads - cache_writes, 0)
    read_rate = (
        price.cache_read_input_per_1m
        if price.cache_read_input_per_1m is not None
        else price.input_per_1m
    )
    write_rate = (
        price.cache_write_input_per_1m
        if price.cache_write_input_per_1m is not None
        else price.input_per_1m
    )
    return (
        uncached / 1_000_000 * price.input_per_1m
        + cache_reads / 1_000_000 * read_rate
        + cache_writes / 1_000_000 * write_rate
        + output_tokens / 1_000_000 * price.output_per_1m
    )


def fetch_openrouter_prices(
    url: str = OPENROUTER_MODELS_URL,
) -> dict[str, ModelPrice]:
    """Pull per-model pricing from the OpenRouter models API (USD per 1M)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "repo-level-dev-agent-eval"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        models = json.load(response)["data"]

    prices: dict[str, ModelPrice] = {}
    for model in models:
        pricing = model.get("pricing", {})
        try:
            per_in = float(pricing.get("prompt", 0)) * 1_000_000
            per_out = float(pricing.get("completion", 0)) * 1_000_000
            cache_read_raw = pricing.get("input_cache_read")
            cache_write_raw = pricing.get("input_cache_write")
            cache_read = (
                float(cache_read_raw) * 1_000_000
                if cache_read_raw not in (None, "")
                else None
            )
            cache_write = (
                float(cache_write_raw) * 1_000_000
                if cache_write_raw not in (None, "")
                else None
            )
        except (TypeError, ValueError):
            continue
        if per_in > 0 or per_out > 0:
            prices[model["id"]] = ModelPrice(
                round(per_in, 4),
                round(per_out, 4),
                round(cache_read, 4) if cache_read is not None else None,
                round(cache_write, 4) if cache_write is not None else None,
            )
    return prices


def write_prices_csv(
    prices: dict[str, ModelPrice], path: Path = PRICING_CSV
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_id",
                "input_per_1m",
                "output_per_1m",
                "cache_read_input_per_1m",
                "cache_write_input_per_1m",
            ]
        )
        for model_id in sorted(prices):
            price = prices[model_id]
            writer.writerow(
                [
                    model_id,
                    price.input_per_1m,
                    price.output_per_1m,
                    price.cache_read_input_per_1m
                    if price.cache_read_input_per_1m is not None
                    else "",
                    price.cache_write_input_per_1m
                    if price.cache_write_input_per_1m is not None
                    else "",
                ]
            )


def main() -> int:
    prices = fetch_openrouter_prices()
    write_prices_csv(prices)
    print(f"Wrote {len(prices)} model prices to {PRICING_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
