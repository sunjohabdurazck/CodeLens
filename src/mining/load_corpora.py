"""
Load raw corpora from Hugging Face.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_IDS = {
    "wildchat": "allenai/WildChat-1M",
    "sharegpt": "Hwaple/ShareGPT52K",  # working 90K mirror
}

LOG_EVERY = 500


def _json_default(o):
    """Serialize HF's datetime objects (and anything else json can't handle)."""
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    return str(o)


def load_samples(
    dataset_key: str,
    out_dir: Path,
    limit: int,
    split: str = "train",
) -> int:
    """Stream `limit` samples from the dataset and write them incrementally to disk.

    Returns the number of samples actually written.
    """
    dataset_id = DATASET_IDS[dataset_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "samples.jsonl"

    logger.info("Loading %s (%s) from Hugging Face...", dataset_key, dataset_id)
    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de charger le dataset '{dataset_id}' (split={split!r}) : {exc}"
        ) from exc

    count = 0
    # Écriture incrémentale : évite de garder tous les échantillons en mémoire
    # et préserve ce qui a déjà été écrit si le flux est interrompu en cours de route.
    with output_path.open("w", encoding="utf-8") as f:
        try:
            for item in ds:
                if limit >= 0 and count >= limit:
                    break
                f.write(json.dumps(item, default=_json_default, ensure_ascii=False) + "\n")
                count += 1
                if count % LOG_EVERY == 0:
                    logger.info("  ... %d échantillons écrits", count)
        except KeyboardInterrupt:
            logger.warning("Interrompu par l'utilisateur après %d échantillons.", count)
        except Exception as exc:
            logger.error("Erreur pendant le streaming après %d échantillons : %s", count, exc)
            raise

    return count


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Télécharge un échantillon d'un corpus Hugging Face en JSONL.")
    parser.add_argument("--dataset", choices=sorted(DATASET_IDS.keys()), required=True)
    parser.add_argument("--out", required=True, type=Path, help="Répertoire de sortie")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Nombre d'échantillons à sauvegarder (négatif ou 0 = tout le split)",
    )
    parser.add_argument("--split", default="train", help="Split du dataset à charger (par défaut: train)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.limit == 0:
        logger.warning("--limit vaut 0 : aucun échantillon ne sera sauvegardé.")

    count = load_samples(args.dataset, args.out, args.limit, split=args.split)
    output_path = args.out / "samples.jsonl"
    logger.info("Saved %d samples to %s", count, output_path)


if __name__ == "__main__":
    main()
