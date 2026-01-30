"""
Módulo de modelos de redes neurais para Orlando Furioso.
Implementa MLP Autoassociativa (Autoencoder) para redução de dimensionalidade.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    """
    MLP Autoassociativa (Autoencoder) para redução de dimensionalidade.
    
    Arquitetura:
        Input → Encoder (Linear + ReLU) → Latent Space → Decoder (Linear) → Output
    
    O autoencoder aprende a comprimir embeddings de alta dimensão em uma
    representação compacta (latent space) e depois reconstruí-los.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int):
        """
        Inicializa o Autoencoder.
        
        Args:
            input_dim: Dimensão da entrada (dimensão dos embeddings)
            encoding_dim: Dimensão da camada de codificação (latent space)
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Encoder: Comprime a entrada para a dimensão de codificação
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True)  # ReLU com inplace=True para economizar memória
        )
        
        # Decoder: Reconstrói a entrada a partir da representação codificada
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass do autoencoder.
        
        Args:
            x: Tensor de entrada (batch_size, input_dim)
            
        Returns:
            Tupla (decoded, encoded):
                - decoded: Saída reconstruída (batch_size, input_dim)
                - encoded: Representação compacta (batch_size, encoding_dim)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna apenas a representação codificada (latent space).
        
        Args:
            x: Tensor de entrada (batch_size, input_dim)
            
        Returns:
            Representação codificada (batch_size, encoding_dim)
        """
        return self.encoder(x)


class AutoencoderTrainer:
    """
    Classe para treinar e gerenciar Autoencoders.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Inicializa o treinador de Autoencoder.
        
        Args:
            input_dim: Dimensão da entrada
            encoding_dim: Dimensão da codificação
            learning_rate: Taxa de aprendizado
            device: Dispositivo ('cuda', 'cpu', ou None para auto-detectar)
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        
        # Configurar dispositivo (GPU ou CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Inicializar modelo
        self.model = Autoencoder(input_dim, encoding_dim).to(self.device)
        
        # Função de perda: MSE (Mean Squared Error)
        self.criterion = nn.MSELoss()
        
        # Otimizador: Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Histórico de treinamento
        self.train_losses = []
        
        print(f"✅ Autoencoder inicializado:")
        print(f"   Input dim: {input_dim}")
        print(f"   Encoding dim: {encoding_dim}")
        print(f"   Device: {self.device}")
        print(f"   Learning rate: {learning_rate}")
    
    def train(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        print_every: int = 20
    ) -> List[float]:
        """
        Treina o autoencoder.
        
        Args:
            data: Dados de treinamento (n_samples, input_dim)
            epochs: Número de épocas
            batch_size: Tamanho do batch
            verbose: Se True, imprime progresso
            print_every: Imprime perda a cada N épocas
            
        Returns:
            Lista com histórico de perdas por época
        """
        # Converter dados numpy para tensor PyTorch
        data_tensor = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if verbose:
            print(f"\n🔄 Iniciando treinamento:")
            print(f"   Amostras: {data.shape[0]}")
            print(f"   Épocas: {epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Batches por época: {len(dataloader)}\n")
        
        self.model.train()  # Coloca o modelo em modo de treinamento
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                
                # Calcular perda
                loss = self.criterion(outputs, inputs)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Calcular perda média da época
            avg_epoch_loss = epoch_loss / batch_count
            self.train_losses.append(avg_epoch_loss)
            
            # Imprimir progresso
            if verbose and (epoch + 1) % print_every == 0:
                print(f"   Epoch [{epoch+1}/{epochs}] - Loss: {avg_epoch_loss:.6f}")
        
        if verbose:
            print(f"\n✅ Treinamento concluído!")
            print(f"   Perda final: {self.train_losses[-1]:.6f}")
        
        return self.train_losses
    
    def encode_data(self, data: np.ndarray) -> np.ndarray:
        """
        Codifica dados usando o encoder treinado.
        
        Args:
            data: Dados a serem codificados (n_samples, input_dim)
            
        Returns:
            Representações codificadas (n_samples, encoding_dim)
        """
        self.model.eval()  # Coloca o modelo em modo de avaliação
        
        with torch.no_grad():  # Desativa cálculo de gradientes
            data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
            encoded = self.model.encode(data_tensor)
        
        return encoded.cpu().numpy()
    
    def save_model(self, filepath: str):
        """
        Salva o modelo treinado.
        
        Args:
            filepath: Caminho do arquivo para salvar
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'train_losses': self.train_losses
        }, filepath)
        print(f"💾 Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carrega um modelo treinado.
        
        Args:
            filepath: Caminho do arquivo para carregar
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        print(f"📂 Modelo carregado de: {filepath}")
    
    def plot_training_curve(
        self,
        title: str = "Training Loss Curve",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plota a curva de perda do treinamento.
        
        Args:
            title: Título do gráfico
            figsize: Tamanho da figura
            save_path: Caminho para salvar o gráfico (None para não salvar)
        """
        if not self.train_losses:
            print("⚠️  Nenhum histórico de treinamento disponível.")
            return
        
        plt.figure(figsize=figsize)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, linewidth=2, color='#2E86AB', alpha=0.8)
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Perda (MSE)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico salvo em: {save_path}")
        
        plt.show()


def train_multiple_autoencoders(
    embeddings_dict: dict,
    encoding_dim: Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    verbose: bool = True
) -> dict:
    """
    Função de conveniência para treinar múltiplos autoencoders.
    
    Args:
        embeddings_dict: Dicionário {'language': embeddings_array}
        encoding_dim: Dimensão de codificação (None para usar input_dim // 4)
        epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        verbose: Se True, imprime progresso
        
    Returns:
        Dicionário com treinadores e representações codificadas:
        {
            'trainers': {'italian': trainer, 'william': trainer, 'john': trainer},
            'encoded': {'italian': array, 'william': array, 'john': array},
            'losses': {'italian': [losses], 'william': [losses], 'john': [losses]}
        }
    """
    input_dim = list(embeddings_dict.values())[0].shape[1]
    
    if encoding_dim is None:
        encoding_dim = input_dim // 4
    
    print("\n" + "="*50)
    print(" "*15 + "TREINANDO AUTOENCODERS")
    print("="*50)
    print(f"Input dim: {input_dim} → Encoding dim: {encoding_dim}")
    print(f"Compressão: {(1 - encoding_dim/input_dim)*100:.1f}%\n")
    
    trainers = {}
    encoded_data = {}
    losses = {}
    
    for lang, embeddings in embeddings_dict.items():
        print("\n" + "=" * 50)
        print(f"🔵 Treinando Autoencoder: {lang.upper()}")
        
        # Criar e treinar autoencoder
        trainer = AutoencoderTrainer(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            learning_rate=learning_rate
        )
        
        train_losses = trainer.train(
            data=embeddings,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Codificar dados
        encoded = trainer.encode_data(embeddings)
        
        trainers[lang] = trainer
        encoded_data[lang] = encoded
        losses[lang] = train_losses
    
    return {
        'trainers': trainers,
        'encoded': encoded_data,
        'losses': losses
    }


def test_reconstruction_quality(
    trainers: Dict[str, 'AutoencoderTrainer'],
    embeddings_scaled: Dict[str, np.ndarray],
    device: Optional[torch.device] = None
) -> Dict[str, dict]:
    """
    Testa a qualidade de reconstrução dos autoencoders treinados.
    
    Args:
        trainers: Dicionário com trainers treinados {'lang': trainer}
        embeddings_scaled: Dicionário com embeddings escalonados {'lang': array}
        device: Device para computação (None = auto-detect)
        
    Returns:
        Dicionário com métricas de reconstrução para cada versão
        Ex: {'italian': {'mse': 0.003, 'loss_percentage': 0.32, ...}, ...}
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    print("\n" + "="*80)
    print("🔬 TESTE DE QUALIDADE DE RECONSTRUÇÃO")
    print("="*80)
    
    for lang, trainer in trainers.items():
        print(f"\n📊 Testando: {lang.upper()}")
        
        # Pegar primeira estrofe
        sample_input = torch.tensor(
            embeddings_scaled[lang][0:1], 
            dtype=torch.float32
        ).to(device)
        
        # Reconstruir
        trainer.model.eval()
        with torch.no_grad():
            reconstructed, latent = trainer.model(sample_input)
        
        # Calcular métricas
        mse = torch.nn.functional.mse_loss(reconstructed, sample_input).item()
        input_variance = sample_input.var().item()
        loss_percentage = (mse / input_variance) * 100 if input_variance > 0 else 0
        
        # Armazenar resultados
        results[lang] = {
            'mse': mse,
            'variance': input_variance,
            'loss_percentage': loss_percentage,
            'input_shape': tuple(sample_input.shape),
            'latent_shape': tuple(latent.shape),
            'reconstructed_shape': tuple(reconstructed.shape)
        }
        
        # Print resumido
        print(f"   MSE: {mse:.6f}")
        print(f"   Perda de informação: {loss_percentage:.2f}%")
        
        # Feedback qualitativo
        if loss_percentage < 1:
            print(f"   ✅ Excelente! (<1%)")
        elif loss_percentage < 10:
            print(f"   ✅ Muito bom! (<10%)")
        elif loss_percentage < 20:
            print(f"   ⚠️  Moderado (10-20%)")
        else:
            print(f"   ❌ Atenção! (>20%)")
    
    print("\n" + "="*80)
    return results