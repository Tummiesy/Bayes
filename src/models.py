"""Naive Bayes model factory."""

from __future__ import annotations

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB


def create_nb_model(model_name: str, alpha: float):
    """Create a Naive Bayes classifier by name."""
    if model_name == "multinomial":
        return MultinomialNB(alpha=alpha)
    if model_name == "bernoulli":
        return BernoulliNB(alpha=alpha)
    if model_name == "complement":
        return ComplementNB(alpha=alpha)

    raise ValueError(f"Unsupported model_name: {model_name}")
