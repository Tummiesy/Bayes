"""Core experiment loop for short-text Naive Bayes classification."""

from __future__ import annotations

from ast import literal_eval
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_loader import build_train_dev_text_label, load_dataset_splits
from src.evaluate import (
    compute_metrics,
    make_classification_report,
    save_confusion_matrix_plot,
    save_macro_f1_barplot,
)
from src.features import create_vectorizer
from src.models import create_nb_model
from src.preprocess import PreprocessConfig, preprocess_corpus
from src.utils import ensure_dir, log, save_dataframe, save_json, save_text


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment search space and run-time settings."""

    datasets: Sequence[str]
    vectorizers: Sequence[str]
    ngram_ranges: Sequence[Tuple[int, int]]
    min_dfs: Sequence[int]
    model_names: Sequence[str]
    alphas: Sequence[float]
    preprocess: PreprocessConfig


class ExperimentRunner:
    """Runs full model selection and evaluation for each dataset independently."""

    def __init__(self, data_root: Path, output_root: Path, config: ExperimentConfig) -> None:
        self.data_root = data_root
        self.output_root = output_root
        self.config = config

    def _fit_predict(
        self,
        train_texts: List[str],
        train_labels: Sequence[str],
        eval_texts: List[str],
        vectorizer_name: str,
        ngram_range: Tuple[int, int],
        min_df: int,
        model_name: str,
        alpha: float,
    ):
        vectorizer = create_vectorizer(
            vectorizer_name=vectorizer_name,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        model = create_nb_model(model_name=model_name, alpha=alpha)

        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("model", model),
        ])

        pipeline.fit(train_texts, train_labels)
        preds = pipeline.predict(eval_texts)
        return pipeline, preds

    @staticmethod
    def _is_count_vectorizer(vectorizer_name: str) -> int:
        return 0 if vectorizer_name == "count" else 1

    @staticmethod
    def _is_unigram_only(ngram_range: Tuple[int, int]) -> int:
        return 0 if ngram_range == (1, 1) else 1

    def _pick_best_config(self, experiment_df: pd.DataFrame) -> pd.Series:
        # tie-break priority:
        # 1) higher macro_f1
        # 2) higher accuracy
        # 3) simpler ngram -> (1,1)
        # 4) simpler vectorizer -> count
        ranked = experiment_df.sort_values(
            by=["macro_f1", "accuracy", "ngram_complexity", "vectorizer_complexity"],
            ascending=[False, False, True, True],
        )
        return ranked.iloc[0]

    def run(self) -> pd.DataFrame:
        """Execute experiments for all datasets and return global summary dataframe."""
        ensure_dir(self.output_root)
        summary_rows: List[Dict] = []
        macro_f1_by_dataset: Dict[str, float] = {}
        feature_dim_summary_rows: List[Dict] = []

        for dataset_name in self.config.datasets:
            log(f"Running dataset: {dataset_name}")
            dataset_output_dir = self.output_root / dataset_name
            ensure_dir(dataset_output_dir)

            splits = load_dataset_splits(data_root=self.data_root, dataset_name=dataset_name)
            (
                train_text,
                train_label,
                dev_text,
                dev_label,
                test_text,
                test_label,
            ) = build_train_dev_text_label(splits)

            train_text_clean = preprocess_corpus(train_text.tolist(), self.config.preprocess)
            dev_text_clean = preprocess_corpus(dev_text.tolist(), self.config.preprocess)
            test_text_clean = preprocess_corpus(test_text.tolist(), self.config.preprocess)

            all_rows: List[Dict] = []
            feature_dim_rows: List[Dict] = []

            for vectorizer_name in self.config.vectorizers:
                for ngram_range in self.config.ngram_ranges:
                    for min_df in self.config.min_dfs:
                        for model_name in self.config.model_names:
                            for alpha in self.config.alphas:
                                config_row = {
                                    "dataset": dataset_name,
                                    "vectorizer": vectorizer_name,
                                    "ngram_range": str(ngram_range),
                                    "min_df": min_df,
                                    "model": model_name,
                                    "alpha": alpha,
                                }
                                feature_dim_row = {
                                    "dataset": dataset_name,
                                    "vectorizer": vectorizer_name,
                                    "ngram_range": str(ngram_range),
                                    "min_df": min_df,
                                    "model": model_name,
                                    "alpha": alpha,
                                    "train_samples": len(train_text_clean),
                                }
                                try:
                                    pipeline, dev_pred = self._fit_predict(
                                        train_texts=train_text_clean,
                                        train_labels=train_label.tolist(),
                                        eval_texts=dev_text_clean,
                                        vectorizer_name=vectorizer_name,
                                        ngram_range=ngram_range,
                                        min_df=min_df,
                                        model_name=model_name,
                                        alpha=alpha,
                                    )
                                    # feature_dim = vocabulary size learned from train texts.
                                    feature_dim_row["feature_dim"] = len(pipeline.named_steps["vectorizer"].vocabulary_)
                                    feature_dim_row["status"] = "ok"
                                    feature_dim_row["error"] = ""

                                    metrics = compute_metrics(dev_label.tolist(), dev_pred.tolist())
                                    config_row.update(metrics)
                                    config_row["status"] = "ok"
                                except Exception as exc:
                                    config_row.update(
                                        {
                                            "accuracy": float("nan"),
                                            "macro_precision": float("nan"),
                                            "macro_recall": float("nan"),
                                            "macro_f1": float("nan"),
                                            "weighted_f1": float("nan"),
                                            "status": "failed",
                                            "error": str(exc),
                                        }
                                    )
                                    feature_dim_row["feature_dim"] = float("nan")
                                    feature_dim_row["status"] = "failed"
                                    feature_dim_row["error"] = str(exc)

                                all_rows.append(config_row)
                                feature_dim_rows.append(feature_dim_row)

            all_experiments_df = pd.DataFrame(all_rows)
            feature_dimensions_df = pd.DataFrame(feature_dim_rows)
            all_experiments_df["ngram_complexity"] = all_experiments_df["ngram_range"].apply(
                lambda s: self._is_unigram_only(literal_eval(s))
            )
            all_experiments_df["vectorizer_complexity"] = all_experiments_df["vectorizer"].apply(
                self._is_count_vectorizer
            )

            valid_df = all_experiments_df.loc[all_experiments_df["status"] == "ok"].copy()
            if valid_df.empty:
                raise RuntimeError(f"All configurations failed for dataset '{dataset_name}'.")

            best = self._pick_best_config(valid_df)

            best_config = {
                "dataset": dataset_name,
                "vectorizer": best["vectorizer"],
                "ngram_range": best["ngram_range"],
                "min_df": int(best["min_df"]),
                "model": best["model"],
                "alpha": float(best["alpha"]),
                "selection_metric": "dev macro_f1",
                "dev_metrics": {
                    "accuracy": float(best["accuracy"]),
                    "macro_precision": float(best["macro_precision"]),
                    "macro_recall": float(best["macro_recall"]),
                    "macro_f1": float(best["macro_f1"]),
                    "weighted_f1": float(best["weighted_f1"]),
                },
                "retrain_strategy": "Option B: retrain on train+dev then evaluate once on test",
                "preprocess": asdict(self.config.preprocess),
            }

            # Option B: retrain best config on train+dev and evaluate once on test.
            train_plus_dev_text = train_text_clean + dev_text_clean
            train_plus_dev_label = train_label.tolist() + dev_label.tolist()

            final_pipeline, test_pred = self._fit_predict(
                train_texts=train_plus_dev_text,
                train_labels=train_plus_dev_label,
                eval_texts=test_text_clean,
                vectorizer_name=best_config["vectorizer"],
                ngram_range=literal_eval(best_config["ngram_range"]),
                min_df=best_config["min_df"],
                model_name=best_config["model"],
                alpha=best_config["alpha"],
            )

            del final_pipeline

            test_metrics = compute_metrics(test_label.tolist(), test_pred.tolist())
            labels = sorted(pd.unique(pd.concat([train_label, dev_label, test_label], ignore_index=True)).tolist())
            report = make_classification_report(test_label.tolist(), test_pred.tolist(), labels=labels)
            save_confusion_matrix_plot(
                y_true=test_label.tolist(),
                y_pred=test_pred.tolist(),
                labels=labels,
                output_path=dataset_output_dir / "confusion_matrix.png",
                title=f"Confusion Matrix - {dataset_name}",
            )

            save_dataframe(
                all_experiments_df.drop(columns=["ngram_complexity", "vectorizer_complexity"]),
                dataset_output_dir / "all_experiments.csv",
            )
            save_dataframe(feature_dimensions_df, dataset_output_dir / "feature_dimensions.csv")
            save_json(best_config, dataset_output_dir / "best_config.json")
            save_json(test_metrics, dataset_output_dir / "test_metrics.json")
            save_text(report, dataset_output_dir / "classification_report.txt")
            feature_dim_summary_rows.extend(feature_dim_rows)

            summary_row = {
                "dataset": dataset_name,
                "best_vectorizer": best_config["vectorizer"],
                "best_ngram_setting": best_config["ngram_range"],
                "best_min_df": best_config["min_df"],
                "best_model": best_config["model"],
                "best_alpha": best_config["alpha"],
                "dev_macro_f1": best_config["dev_metrics"]["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
            }
            summary_rows.append(summary_row)
            macro_f1_by_dataset[dataset_name] = test_metrics["macro_f1"]

            log(f"Finished dataset: {dataset_name}")

        summary_df = pd.DataFrame(summary_rows).sort_values(by="dataset").reset_index(drop=True)
        feature_dim_summary_df = pd.DataFrame(feature_dim_summary_rows)
        save_dataframe(summary_df, self.output_root / "summary_results.csv")
        save_dataframe(feature_dim_summary_df, self.output_root / "feature_dimensions_summary.csv")
        save_macro_f1_barplot(macro_f1_by_dataset, self.output_root / "best_test_macro_f1_barplot.png")

        analysis_lines = ["Best test Macro-F1 by dataset:"]
        if macro_f1_by_dataset:
            sorted_items = sorted(macro_f1_by_dataset.items(), key=lambda item: item[1], reverse=True)
            easiest_dataset, easiest_score = sorted_items[0]
            hardest_dataset, hardest_score = sorted_items[-1]
            for name, score in sorted_items:
                analysis_lines.append(f"- {name}: {score:.4f}")
            analysis_lines.append("")
            analysis_lines.append(
                f"Easiest for Naive Bayes (highest test Macro-F1): {easiest_dataset} ({easiest_score:.4f})."
            )
            analysis_lines.append(
                f"Hardest for Naive Bayes (lowest test Macro-F1): {hardest_dataset} ({hardest_score:.4f})."
            )
        save_text("\n".join(analysis_lines) + "\n", self.output_root / "analysis.txt")

        return summary_df
