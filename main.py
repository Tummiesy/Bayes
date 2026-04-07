"""Entry point for Naive Bayes short-text classification experiments."""

from __future__ import annotations

from pathlib import Path

from src.experiment import ExperimentConfig, ExperimentRunner
from src.preprocess import PreprocessConfig
from src.utils import log


def main() -> None:
    data_root = Path("data")
    output_root = Path("outputs")

    config = ExperimentConfig(
        datasets=["banking", "snips", "stackoverflow"],
        vectorizers=["count", "tfidf"],
        ngram_ranges=[(1, 1), (1, 2)],
        min_dfs=[1, 2, 3],
        model_names=["multinomial", "bernoulli", "complement"],
        alphas=[0.1, 0.5, 1.0, 2.0],
        preprocess=PreprocessConfig(
            lowercase=True,
            strip=True,
            collapse_whitespace=True,
            remove_punctuation=False,
            remove_digits=False,
            remove_stopwords=False,
        ),
    )

    runner = ExperimentRunner(data_root=data_root, output_root=output_root, config=config)
    summary_df = runner.run()
    log("All experiments completed.")
    log(f"Summary saved to: {output_root / 'summary_results.csv'}")
    log("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
