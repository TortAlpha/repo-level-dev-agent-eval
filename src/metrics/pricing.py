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


def load_prices(path: Path = PRICING_CSV) -> dict[str, ModelPrice]:
    if not path.is_file():
        return {}
    prices: dict[str, ModelPrice] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prices[row["model_id"]] = ModelPrice(
                float(row["input_per_1m"]), float(row["output_per_1m"])
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
) -> float | None:
    """USD cost of one run's tokens, or ``None`` if the model has no price."""
    price = override or (PRICES if prices is None else prices).get(model)
    if price is None:
        return None
    return (
        input_tokens / 1_000_000 * price.input_per_1m
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
        except (TypeError, ValueError):
            continue
        if per_in > 0 or per_out > 0:
            prices[model["id"]] = ModelPrice(round(per_in, 4), round(per_out, 4))
    return prices


def write_prices_csv(
    prices: dict[str, ModelPrice], path: Path = PRICING_CSV
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "input_per_1m", "output_per_1m"])
        for model_id in sorted(prices):
            price = prices[model_id]
            writer.writerow([model_id, price.input_per_1m, price.output_per_1m])


def main() -> int:
    prices = fetch_openrouter_prices()
    write_prices_csv(prices)
    print(f"Wrote {len(prices)} model prices to {PRICING_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
