# Top-level package for the scholar-classifier project.
from __future__ import annotations

# Re-export the public ML models
from .logistic_regression import SoftmaxRegression
from .decision_tree import ID3DecisionTree
from .svm import SVM_QP, DAGSVM

# Re-export the base class (useful for type hints and subclassing)
from .core.base_model import BaseClassifier

# Re-export common utilities
from .utils import (
	one_hot_encode,
	fill_missing,
	standardize,
	f1_macro,
	save_model,
	load_model,
)

__all__ = [
	# models
	"SoftmaxRegression",
	"ID3DecisionTree",
	"SVM_QP",
	"DAGSVM",
	# core
	"BaseClassifier",
	# utils
	"one_hot_encode",
	"fill_missing",
	"standardize",
	"f1_macro",
	"save_model",
	"load_model",
]

