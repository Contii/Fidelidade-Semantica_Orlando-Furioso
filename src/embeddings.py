"""
Módulo de geração de embeddings multilíngues para Orlando Furioso.
Utiliza SentenceTransformers para criar representações vetoriais das estrofes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import unicodedata


class EmbeddingGenerator:
    """Gerador de embeddings multilíngues para textos do Orlando Furioso."""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Inicializa o gerador de embeddings.
        
        Args:
            model_name: Nome do modelo SentenceTransformer a ser usado.
                       Padrão: 'paraphrase-multilingual-MiniLM-L12-v2'
                       (otimizado para textos multilíngues)
        """
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        
    def load_model(self):
        """Carrega o modelo SentenceTransformer."""
        print(f"📥 Carregando modelo de embeddings: {self.model_name}...")
        self.model = SentenceTransformer(f'sentence-transformers/{self.model_name}')
        print(f"✅ Modelo carregado com sucesso!")
        print(f"   Dimensão dos embeddings: {self.model.get_sentence_embedding_dimension()}")
        
    def preprocess_text(self, text: str) -> str:
        """
        Pré-processa texto para geração de embeddings.
        
        IMPORTANTE: Mantém a formatação original (acentos, apóstrofos, elisões)
        pois o modelo SentenceTransformer foi treinado para entender essas nuances.
        
        Apenas converte para minúsculas para consistência.
        
        Args:
            text: Texto a ser processado
            
        Returns:
            Texto em minúsculas (preservando pontuação e estrutura)
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalização Unicode para garantir consistência de caracteres
        # NFD (Canonical Decomposition) seguido de NFC (Canonical Composition)
        # Isso resolve problemas com caracteres italianos especiais como:
        # - à, è, ì, ò, ù (acentos graves)
        # - é (acento agudo)
        # - Apóstrofos e elisões: l'arme, ch'io, s'un, ecc.
        text = unicodedata.normalize('NFC', text)
        
        # Conversão para minúsculas (única transformação aplicada)
        text = text.lower()
        
        return text
    
    def generate_embeddings(
        self, 
        texts: list, 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.
        
        Args:
            texts: Lista de textos (estrofes)
            batch_size: Tamanho do batch para processamento
            show_progress: Se True, mostra barra de progresso
            
        Returns:
            Array numpy com embeddings (shape: [n_texts, embedding_dim])
        """
        if self.model is None:
            self.load_model()
        
        # Pré-processar textos
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Gerar embeddings
        print(f"🔄 Gerando embeddings para {len(texts)} textos...")
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"✅ Embeddings gerados! Shape: {embeddings.shape}")
        return embeddings
    
    def scale_embeddings(
        self,
        embeddings_dict: dict,
        fit_on_all: bool = True
    ) -> dict:
        """
        Escala embeddings usando StandardScaler.
        
        Args:
            embeddings_dict: Dicionário {'language': embeddings_array}
                            Ex: {'italian': array, 'william': array, 'john': array}
            fit_on_all: Se True, fita o scaler em todos os embeddings concatenados.
                       Se False, fita apenas no primeiro conjunto (italiano)
        
        Returns:
            Dicionário com embeddings escalonados
        """
        print(f"📏 Escalonando embeddings...")
        
        if fit_on_all:
            # Concatenar todos os embeddings para fitar o scaler
            all_embeddings = np.vstack(list(embeddings_dict.values()))
            self.scaler.fit(all_embeddings)
            print(f"   Scaler ajustado em {all_embeddings.shape[0]} amostras")
        else:
            # Fitar apenas no primeiro conjunto (geralmente o original)
            first_key = list(embeddings_dict.keys())[0]
            self.scaler.fit(embeddings_dict[first_key])
            print(f"   Scaler ajustado em '{first_key}' ({embeddings_dict[first_key].shape[0]} amostras)")
        
        # Transformar todos os embeddings
        scaled_embeddings = {}
        for lang, embeddings in embeddings_dict.items():
            scaled_embeddings[lang] = self.scaler.transform(embeddings)
            print(f"   ✓ {lang}: {scaled_embeddings[lang].shape}")
        
        print(f"✅ Escalonamento concluído!")
        return scaled_embeddings
    
    def save_embeddings(
        self,
        embeddings_dict: dict,
        output_dir: str = 'outputs/embeddings',
        prefix: str = 'embeddings'
    ):
        """
        Salva embeddings em arquivos .npy.
        
        Args:
            embeddings_dict: Dicionário com embeddings
            output_dir: Diretório de saída
            prefix: Prefixo dos arquivos
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Salvando embeddings em {output_path}...")
        
        for lang, embeddings in embeddings_dict.items():
            filepath = output_path / f"{prefix}_{lang}.npy"
            np.save(filepath, embeddings)
            print(f"   ✓ {filepath}")
        
        # Salvar também o scaler
        scaler_path = output_path / f"{prefix}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✓ {scaler_path}")
        
        print(f"✅ Embeddings salvos com sucesso!")
    
    def load_embeddings(
        self,
        input_dir: str = 'outputs/embeddings',
        prefix: str = 'embeddings',
        languages: list = ['italian', 'william', 'john']
    ) -> dict:
        """
        Carrega embeddings de arquivos .npy.
        
        Args:
            input_dir: Diretório de entrada
            prefix: Prefixo dos arquivos
            languages: Lista de idiomas/versões a carregar
            
        Returns:
            Dicionário com embeddings carregados
        """
        input_path = Path(input_dir)
        
        print(f"📂 Carregando embeddings de {input_path}...")
        
        embeddings_dict = {}
        for lang in languages:
            filepath = input_path / f"{prefix}_{lang}.npy"
            if filepath.exists():
                embeddings_dict[lang] = np.load(filepath)
                print(f"   ✓ {lang}: {embeddings_dict[lang].shape}")
            else:
                print(f"   ⚠️  Arquivo não encontrado: {filepath}")
        
        # Carregar scaler
        scaler_path = input_path / f"{prefix}_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"   ✓ Scaler carregado")
        
        print(f"✅ Embeddings carregados com sucesso!")
        return embeddings_dict


def generate_embeddings_from_dataframe(
    df: pd.DataFrame,
    model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    save_path: Optional[str] = 'outputs/embeddings',
    batch_size: int = 32
) -> Tuple[dict, dict]:
    """
    Função de conveniência para gerar embeddings a partir de um DataFrame.
    
    Args:
        df: DataFrame com colunas 'italian', 'william', 'john'
        model_name: Nome do modelo SentenceTransformer
        save_path: Caminho para salvar embeddings (None para não salvar)
        batch_size: Tamanho do batch para processamento
        
    Returns:
        Tupla (embeddings_raw, embeddings_scaled)
    """
    generator = EmbeddingGenerator(model_name)
    generator.load_model()
    
    # Gerar embeddings para cada versão
    print("\n" + "="*50)
    print(" "*15 + "GERANDO EMBEDDINGS" + " "*15)
    print("="*50 + "\n")
    
    embeddings_raw = {}
    
    print("\n🇮🇹 ITALIANO (Original):")
    embeddings_raw['italian'] = generator.generate_embeddings(
        df['italian'].tolist(),
        batch_size=batch_size
    )
    
    print("\n🇬🇧 WILLIAM STEWART ROSE:")
    embeddings_raw['william'] = generator.generate_embeddings(
        df['william'].tolist(),
        batch_size=batch_size
    )
    
    print("\n🇬🇧 JOHN HARINGTON:")
    embeddings_raw['john'] = generator.generate_embeddings(
        df['john'].tolist(),
        batch_size=batch_size
    )
    
    # Escalar embeddings
    print("\n" + "="*50)
    print(" "*15 + "ESCALONANDO EMBEDDINGS" + " "*15)
    print("="*50 + "\n")
    
    embeddings_scaled = generator.scale_embeddings(embeddings_raw, fit_on_all=True)
    
    # Salvar se solicitado
    if save_path:
        print("\n" + "="*50)
        print(" "*15 + "SALVANDO EMBEDDINGS" + " "*15)
        print("="*50 + "\n")
        
        generator.save_embeddings(embeddings_raw, save_path, prefix='embeddings_raw')
        generator.save_embeddings(embeddings_scaled, save_path, prefix='embeddings_scaled')
    
    return embeddings_raw, embeddings_scaled