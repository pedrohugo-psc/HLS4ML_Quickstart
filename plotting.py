"""Funções auxiliares para visualização gráfica (curvas ROC, dicionários de config)."""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def print_dict(d, indent=''):
    """Imprime dicionários aninhados de forma legível e organizada."""
    for k, v in d.items():
        if isinstance(v, dict):
            print(f'{indent}{k}:')
            print_dict(v, indent + '  ')
        else:
            print(f'{indent}{k}: {v}')


def makeRoc(y_test, y_pred, labels, linestyle='-'):
    """Calcula e plota as curvas ROC para cada classe avaliada."""
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', linestyle=linestyle)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curvas ROC por Classe')
    plt.legend(loc='lower right')
