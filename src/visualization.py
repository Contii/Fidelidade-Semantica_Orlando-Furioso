"""
Módulo de visualização para análise de fidelidade semântica do Orlando Furioso.
Gera gráficos e visualizações para análise comparativa de traduções.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Configuração global de estilo
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


# Paleta de cores consistente
COLORS = {
    'italian': "#0AC900E3",
    'william': "#0018CE",
    'john': "#B300FF",
    'primary': "#2EAB75",
    'secondary': "#A23B61",
    'tertiary': "#F1D501"
}

LABELS = {
    'italian': 'Italiano (Original)',
    'william': 'William S. Rose',
    'john': 'John Harington'
}


def plot_training_curves(
    losses_dict: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Curvas de Erro das MLPs Autoassociativas Durante o Treinamento',
    save_path: Optional[str] = None
):
    """
    Plota curvas de treinamento dos autoencoders.
    
    Args:
        losses_dict: Dicionário {'language': [losses]}
        figsize: Tamanho da figura
        title: Título do gráfico
        save_path: Caminho para salvar (None para não salvar)
    """
    plt.figure(figsize=figsize)
    
    for lang, losses in losses_dict.items():
        epochs = range(1, len(losses) + 1)
        plt.plot(
            epochs, losses,
            label=LABELS.get(lang, lang),
            color=COLORS.get(lang, 'gray'),
            linewidth=2,
            alpha=0.8
        )
    
    plt.xlabel('Época', fontsize=14)
    plt.ylabel('Perda (MSE)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")
    
    plt.show()


def plot_fidelity_by_canto(
    df: pd.DataFrame,
    fidelity_columns: List[str],
    figsize: Tuple[int, int] = (14, 7),
    title: str = 'Fidelidade Semântica Média por Canto',
    save_path: Optional[str] = None
):
    """
    Plota fidelidade semântica média por canto.
    
    Args:
        df: DataFrame com scores de fidelidade
        fidelity_columns: Lista de colunas de fidelidade
        figsize: Tamanho da figura
        title: Título do gráfico
        save_path: Caminho para salvar
    """
    # Calcular média por canto
    fidelity_by_canto = df.groupby('canto')[fidelity_columns].mean().reset_index()
    
    plt.figure(figsize=figsize)
    
    # Plotar cada tradução
    for col in fidelity_columns:
        translator = col.replace('fidelidade_semantica_', '')
        marker = 'o'
        linestyle = '-'
        
        plt.plot(
            fidelity_by_canto['canto'],
            fidelity_by_canto[col],
            label=LABELS.get(translator, translator.upper()),
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            alpha=0.8,
            color=COLORS.get(translator, 'gray'),
            markersize=6
        )
    
    plt.xlabel('Canto', fontsize=16)
    plt.ylabel('Similaridade de Cosseno', fontsize=16)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fidelity_by_canto['canto'][::5], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")
    
    plt.show()


def plot_fidelity_distribution(
    df: pd.DataFrame,
    fidelity_columns: List[str],
    figsize: Tuple[int, int] = (14, 6),
    bins: int = 30,
    save_path: Optional[str] = None
):
    """
    Plota histogramas da distribuição de fidelidade.
    
    Args:
        df: DataFrame com scores de fidelidade
        fidelity_columns: Lista de colunas de fidelidade
        figsize: Tamanho da figura
        bins: Número de bins para o histograma
        save_path: Caminho para salvar
    """
    n_plots = len(fidelity_columns)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]

    # Calcular o valor máximo de frequência entre todos os gráficos
    max_freq = 0
    for col in fidelity_columns:
        counts, _ = np.histogram(df[col], bins=bins)
        max_freq = max(max_freq, counts.max())
    
    # Adicionar margem de 10% ao limite superior
    y_limit = max_freq * 1.1
    
    for ax, col in zip(axes, fidelity_columns):
        translator = col.replace('fidelidade_semantica_', '')
        color = COLORS.get(translator, 'skyblue')
        
        sns.histplot(
            df[col],
            bins=bins,
            kde=True,
            color=color,
            alpha=0.7,
            ax=ax
        )
        
        ax.set_title(
            f'Distribuição da Fidelidade\n{LABELS.get(translator, translator.upper())}',
            fontsize=13,
            fontweight='bold'
        )
        ax.set_xlabel('Similaridade de Cosseno', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Definir limite do eixo Y para todos os gráficos
        ax.set_ylim(0, y_limit)
        
        # Adicionar estatísticas
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Mediana: {median_val:.3f}')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")
    
    plt.show()
