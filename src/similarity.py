"""
Módulo de cálculo de similaridade semântica para Orlando Furioso.
Implementa métricas de fidelidade semântica entre traduções.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple


def calculate_cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Calcula a similaridade de cosseno entre dois vetores.
    
    Args:
        vec1: Primeiro vetor (embedding ou representação latente)
        vec2: Segundo vetor
        
    Returns:
        Similaridade de cosseno (valor entre -1 e 1)
    """
    # Reshape para garantir dimensão correta (1, n_features)
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    # Calcular similaridade de cosseno
    similarity = cosine_similarity(vec1, vec2)[0][0]
    
    return similarity


def calculate_pairwise_similarities(
    representations_original: np.ndarray,
    representations_translation: np.ndarray
) -> np.ndarray:
    """
    Calcula similaridades de cosseno entre pares de representações.
    
    Args:
        representations_original: Representações do texto original (n_samples, n_features)
        representations_translation: Representações da tradução (n_samples, n_features)
        
    Returns:
        Array com similaridades para cada par (n_samples,)
    """
    if representations_original.shape[0] != representations_translation.shape[0]:
        raise ValueError(
            f"Número de amostras diferente: "
            f"original={representations_original.shape[0]}, "
            f"tradução={representations_translation.shape[0]}"
        )
    
    n_samples = representations_original.shape[0]
    similarities = np.zeros(n_samples)
    
    # Calcular similaridade para cada par de estrofes
    for i in range(n_samples):
        similarities[i] = calculate_cosine_similarity(
            representations_original[i],
            representations_translation[i]
        )
    
    return similarities


def calculate_fidelity_scores(
    encoded_original: np.ndarray,
    encoded_translations: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calcula scores de fidelidade semântica para múltiplas traduções.
    
    Args:
        encoded_original: Representações latentes do original (n_samples, encoding_dim)
        encoded_translations: Dicionário {'translator': encoded_array}
                             Ex: {'william': array, 'john': array}
        
    Returns:
        Dicionário com scores de fidelidade para cada tradução
        Ex: {'william': similarities_array, 'john': similarities_array}
    """
    fidelity_scores = {}
    
    for translator, encoded_translation in encoded_translations.items():
        similarities = calculate_pairwise_similarities(
            encoded_original,
            encoded_translation
        )
        
        fidelity_scores[translator] = similarities
    
    return fidelity_scores


def add_fidelity_to_dataframe(
    df: pd.DataFrame,
    fidelity_scores: Dict[str, np.ndarray],
    prefix: str = 'fidelidade_semantica'
) -> pd.DataFrame:
    """
    Adiciona scores de fidelidade ao DataFrame.
    
    Args:
        df: DataFrame com dados das estrofes
        fidelity_scores: Dicionário com scores de fidelidade
        prefix: Prefixo para as colunas criadas
        
    Returns:
        DataFrame com novas colunas de fidelidade
    """
    df_copy = df.copy()
    
    for translator, scores in fidelity_scores.items():
        column_name = f"{prefix}_{translator}"
        df_copy[column_name] = scores
        print(f"✓ Coluna adicionada: {column_name}")
    
    return df_copy





def get_extreme_stanzas(
    df: pd.DataFrame,
    fidelity_column: str,
    n_top: int = 5,
    n_bottom: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifica estrofes com maior e menor fidelidade.
    
    Args:
        df: DataFrame com scores de fidelidade
        fidelity_column: Nome da coluna de fidelidade
        n_top: Número de estrofes com maior fidelidade
        n_bottom: Número de estrofes com menor fidelidade
        
    Returns:
        Tupla (top_stanzas, bottom_stanzas)
    """
    # Ordenar por fidelidade
    df_sorted = df.sort_values(by=fidelity_column, ascending=False)
    
    # Top N (maior fidelidade)
    top_stanzas = df_sorted.head(n_top).copy()
    
    # Bottom N (menor fidelidade)
    bottom_stanzas = df_sorted.tail(n_bottom).copy()
    
    return top_stanzas, bottom_stanzas


def print_fidelity_summary(
    df: pd.DataFrame,
    fidelity_columns: list
):
    """
    Imprime resumo estatístico da fidelidade semântica.
    
    Args:
        df: DataFrame com scores de fidelidade
        fidelity_columns: Lista de colunas de fidelidade
    """
    print("\n" + "="*50)
    print("     RESUMO ESTATÍSTICO DA FIDELIDADE SEMÂNTICA")
    print("="*50)
    
    for col in fidelity_columns:
        translator = col.replace('fidelidade_semantica_', '').upper()
        
        print(f"\n📊 {translator}:")
        print(f"   Média: {df[col].mean():.4f}")
        print(f"   Mediana: {df[col].median():.4f}")
        print(f"   Desvio padrão: {df[col].std():.4f}")
        print(f"   Mínimo: {df[col].min():.4f}")
        print(f"   Máximo: {df[col].max():.4f}")
        print(f"   Q1 (25%): {df[col].quantile(0.25):.4f}")
        print(f"   Q3 (75%): {df[col].quantile(0.75):.4f}")
        
        # Categorização
        high_fidelity = (df[col] >= 0.80).sum()
        medium_fidelity = ((df[col] >= 0.60) & (df[col] < 0.80)).sum()
        low_fidelity = (df[col] < 0.60).sum()
        
        total = len(df)
        print(f"\n   Distribuição por categoria:")
        print(f"      Alta fidelidade (≥0.80): {high_fidelity} ({high_fidelity/total*100:.1f}%)")
        print(f"      Média fidelidade (0.60-0.80): {medium_fidelity} ({medium_fidelity/total*100:.1f}%)")
        print(f"      Baixa fidelidade (<0.60): {low_fidelity} ({low_fidelity/total*100:.1f}%)\n")
    