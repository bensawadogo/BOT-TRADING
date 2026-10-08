from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deriv.constants import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("BACKTEST")


def _disarm_model_saves(ensemble) -> None:
    """Empêche tout modèle de l'ensemble d'écrire pendant l'essai."""
    for name in ("hmm", "xgb", "lstm"):
        model = getattr(ensemble, name, None)
        save = getattr(model, "_save", None)
        if callable(save):
            model._save = lambda: None


@contextmanager
def _temporary_model_dir() -> Iterator[str]:
    model_dir = tempfile.mkdtemp(prefix="deriv_models_")
    import deriv.ensemble_predictor as ensemble_predictor

    previous = ensemble_predictor.MODELS_DIR
    ensemble_predictor.MODELS_DIR = model_dir
    try:
        yield model_dir
    finally:
        ensemble_predictor.MODELS_DIR = previous
        shutil.rmtree(model_dir, ignore_errors=True)


def _publish_models(source_dir: str, target_dir: str) -> list[str]:
    """Copie explicitement les artefacts validés avec un rename atomique."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    published = []
    for source in Path(source_dir).iterdir():
        if not source.is_file() or source.name.startswith("."):
            continue
        destination = target / source.name
        temporary = target / f".{source.name}.tmp"
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        published.append(source.name)
    return published


async def _load_market_data(symbol: str, granularity: int, count: int):
    from deriv.data_collector import DerivDataCollector

    collector = DerivDataCollector(symbol, granularity)
    try:
        await collector.connect()
        df = await collector.get_candle_history_full(target_count=count)
    finally:
        await collector.close()

    if df is None or df.empty:
        raise ValueError(f"Aucune bougie Deriv reçue pour {symbol}")
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Données invalides pour {symbol}: colonnes OHLC requises")
    return df


async def run(args: argparse.Namespace) -> dict:
    """Entraîne et évalue sans toucher aux modèles de production par défaut."""
    load_dotenv()
    df = await _load_market_data(args.symbol, args.granularity, args.count)
    logger.info("Données Deriv chargées : %s bougies", len(df))

    from deriv.ensemble_predictor import EnsemblePredictor
    from deriv.walk_forward_optimizer import WalkForwardOptimizer

    with _temporary_model_dir() as model_dir:
        logger.info("Répertoire d'entraînement temporaire : %s", model_dir)
        ensemble = EnsemblePredictor()
        _disarm_model_saves(ensemble)
        ensemble.train_all(df)

        optimizer = WalkForwardOptimizer(
            granularity=args.granularity,
            eval_in_sample=not args.no_in_sample,
        )
        summary = optimizer.run(df, target_mode=args.target)
        published = (
            _publish_models(model_dir, os.path.join(ROOT, "deriv", "models"))
            if args.publish
            else []
        )

    result = {
        "ok": True,
        "dry_run": args.dry_run,
        "symbol": args.symbol,
        "granularity": args.granularity,
        "n_bougies": len(df),
        "summary": summary,
        "published": published,
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entraînement et Walk-Forward sur données Deriv"
    )
    parser.add_argument("--symbol", default=Config.SYMBOL)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--granularity", type=int, default=60)
    parser.add_argument("--target", choices=("sign", "crash", "triple_barrier"),
                        default="triple_barrier")
    parser.add_argument("--no-in-sample", action="store_true")
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Mode sûr par défaut : aucun modèle n'est publié",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publier explicitement les modèles validés dans deriv/models",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.count <= 0 or args.granularity <= 0:
        logger.error("--count et --granularity doivent être strictement positifs")
        return 2
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        logger.exception("Entraînement/backtest échoué : %s", exc)
        return 1
    print(f"OK dry-run={result['dry_run']} bougies={result['n_bougies']}")
    if result.get("published"):
        print("Modèles publiés : " + ", ".join(result["published"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
