#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation study runner (M1-M5).

Charge un split de test (CSV) et un fichier de prédictions (JSON),
puis calcule les métriques pour chaque variante du modèle.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_split(path: str) -> List[Dict[str, str]]:
    """Charge un fichier CSV et retourne une liste de dictionnaires (une entrée par ligne)."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Fichier de test introuvable : {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Le fichier CSV est vide : {csv_path}")

    return rows


def run_ablation(test_path: str, predictions: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Calcule les métriques pour chaque ensemble de prédictions (M1, M2, ...).

    Args:
        test_path: chemin vers le CSV contenant la colonne "acqp_score".
        predictions: dict {nom_du_modèle: liste_des_prédictions}.

    Returns:
        dict {nom_du_modèle: métriques calculées}.
    """
    examples = load_split(test_path)

    try:
        y_true = [float(ex["acqp_score"]) for ex in examples]
    except KeyError:
        raise KeyError("La colonne 'acqp_score' est absente du fichier CSV.")
    except ValueError as e:
        raise ValueError(f"Valeur non convertible en float dans 'acqp_score' : {e}")

    if not predictions:
        raise ValueError("Le dictionnaire de prédictions est vide.")

    results: Dict[str, Any] = {}
    for name, y_pred in predictions.items():
        if len(y_pred) != len(y_true):
            raise AssertionError(
                f"{name}: nombre de prédictions ({len(y_pred)}) "
                f"différent du nombre de labels ({len(y_true)})"
            )
        results[name] = compute_metrics(y_true, y_pred)
        logger.info("Métriques calculées pour %s", name)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner d'étude d'ablation (M1-M5).")
    parser.add_argument("--test", required=True, help="Chemin vers le CSV de test.")
    parser.add_argument("--predictions", required=True, help="Chemin vers le JSON des prédictions.")
    parser.add_argument(
        "--output", required=False, default=None,
        help="Chemin optionnel pour sauvegarder les résultats en JSON (en plus du print)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preds_path = Path(args.predictions)
    if not preds_path.is_file():
        logger.error("Fichier de prédictions introuvable : %s", preds_path)
        sys.exit(1)

    try:
        with preds_path.open(encoding="utf-8") as f:
            preds = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("JSON de prédictions invalide : %s", e)
        sys.exit(1)

    try:
        results = run_ablation(args.test, preds)
    except (FileNotFoundError, ValueError, KeyError, AssertionError) as e:
        logger.error(str(e))
        sys.exit(1)

    output_str = json.dumps(results, indent=2)
    print(output_str)

    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
        logger.info("Résultats sauvegardés dans %s", args.output)


if __name__ == "__main__":
    main()
