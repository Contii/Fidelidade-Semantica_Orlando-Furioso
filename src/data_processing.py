"""
Módulo de processamento de dados para Orlando Furioso.
Parsing de cantos e estrofes dos arquivos .txt.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


class OrlandoFuriosoParser:
    """Parser para extrair cantos e estrofes do Orlando Furioso."""
    
    # Padrões regex validados no notebook de teste
    CANTO_PATTERN = re.compile(r'^CANTO\s+(\d+)\s*$', re.MULTILINE)
    ESTROFE_PATTERN = re.compile(
        r'^\s*(\d+)\s*\n'
        r'((?:(?!^\s*\d+\s*\n|^CANTO\s+\d+\s*$).)+)',
        re.MULTILINE | re.DOTALL
    )
    
    def __init__(self, filepath: str, language: str):
        """
        Inicializa o parser.
        
        Args:
            filepath: Caminho para o arquivo .txt
            language: 'italian', 'william', ou 'john'
        """
        self.filepath = Path(filepath)
        self.language = language
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        self.text = self._load_text()
    
    def _load_text(self) -> str:
        """Carrega texto do arquivo."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse(self) -> Dict[int, Dict[int, str]]:
        """
        Extrai estrutura hierárquica: {canto_num: {estrofe_num: estrofe_text}}
        
        Returns:
            Dicionário com cantos e estrofes parseados
        """
        cantos = {}
        
        # Encontrar todos os cantos
        canto_matches = list(self.CANTO_PATTERN.finditer(self.text))
        
        for i, canto_match in enumerate(canto_matches):
            canto_num = int(canto_match.group(1))
            
            # Extrair texto do canto (até o próximo canto ou fim)
            start_pos = canto_match.end()
            end_pos = (canto_matches[i + 1].start() 
                      if i + 1 < len(canto_matches) 
                      else len(self.text))
            canto_text = self.text[start_pos:end_pos]
            
            # Extrair estrofes
            estrofes = self._extract_estrofes(canto_text)
            
            if estrofes:
                cantos[canto_num] = estrofes
        
        return cantos
    
    def _extract_estrofes(self, canto_text: str) -> Dict[int, str]:
        """
        Extrai estrofes numeradas de um canto.
        
        Args:
            canto_text: Texto completo do canto
            
        Returns:
            Dicionário {estrofe_num: estrofe_text}
        """
        estrofes = {}
        
        for match in self.ESTROFE_PATTERN.finditer(canto_text):
            estrofe_num = int(match.group(1))
            estrofe_text = match.group(2).strip()
            
            if estrofe_text:
                estrofes[estrofe_num] = estrofe_text
        
        return estrofes


def create_aligned_dataframe(
    italian: Dict[int, Dict[int, str]],
    english_william: Dict[int, Dict[int, str]],
    english_john: Dict[int, Dict[int, str]]
) -> pd.DataFrame:
    """
    Cria DataFrame alinhado com todas as versões.
    
    Estrutura do DataFrame:
        | canto | estrofe | italian | william | john |
        |-------|--------|---------|---------|------|
        | 1     | 1      | texto   | text    | text |
        | 1     | 2      | texto   | NaN     | text |  <- Estrofe ausente em william
    
    Args:
        italian: Dicionário parseado do original italiano
        english_william: Dicionário da tradução de William Stewart Rose
        english_john: Dicionário da tradução de John Harington
    
    Returns:
        DataFrame alinhado com colunas: canto, estrofe, italian, william, john
    """
    records = []
    
    # Obter todos os cantos únicos
    all_cantos = sorted(set(
        list(italian.keys()) + 
        list(english_william.keys()) + 
        list(english_john.keys())
    ))
    
    for canto in all_cantos:
        # Número máximo de estrofes neste canto (entre todas as versões)
        max_estrofe = max(
            max(italian.get(canto, {}).keys(), default=0),
            max(english_william.get(canto, {}).keys(), default=0),
            max(english_john.get(canto, {}).keys(), default=0)
        )
        
        # Criar linha para cada estrofe possível
        for estrofe_num in range(1, max_estrofe + 1):
            records.append({
                'canto': canto,
                'estrofe': estrofe_num,
                'italian': italian.get(canto, {}).get(estrofe_num),
                'william': english_william.get(canto, {}).get(estrofe_num),
                'john': english_john.get(canto, {}).get(estrofe_num)
            })
    
    df = pd.DataFrame(records)
    
    # Filtrar linhas sem texto italiano (não há base de comparação)
    df = df.dropna(subset=['italian']).reset_index(drop=True)
    
    # Preencher traduções ausentes com string vazia
    df['william'] = df['william'].fillna('')
    df['john'] = df['john'].fillna('')
    
    return df


def load_and_parse_all(data_dir: str = 'data') -> pd.DataFrame:
    """
    Função de conveniência para carregar e parsear todos os arquivos.
    
    Args:
        data_dir: Diretório contendo os arquivos .txt
    
    Returns:
        DataFrame alinhado com todas as versões
    """
    data_path = Path(data_dir)
    
    # Parsear os três arquivos
    italian_parser = OrlandoFuriosoParser(
        data_path / 'Orlando Furioso - Ludovico Ariosto.txt',
        'italian'
    )
    william_parser = OrlandoFuriosoParser(
        data_path / 'Orlando Furioso - William Stewart Rose.txt',
        'william'
    )
    john_parser = OrlandoFuriosoParser(
        data_path / 'Orlando Furioso - John Harrington.txt',
        'john'
    )
    
    italian_data = italian_parser.parse()
    william_data = william_parser.parse()
    john_data = john_parser.parse()
    
    # Criar DataFrame alinhado
    df = create_aligned_dataframe(italian_data, william_data, john_data)
    
    print(f"✅ Parsing concluído: {len(df)} estrofes, {df['canto'].nunique()} cantos")
    
    return df