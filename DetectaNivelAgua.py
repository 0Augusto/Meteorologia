import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from datetime import datetime
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
import threading
from queue import Queue
import logging

@dataclass
class WaterLevelSample:
    timestamp: datetime
    water_level: float
    confidence: float
    image: np.ndarray

class WaterLevelDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa o detector de nível de água
        Args:
            model_path: Caminho para um modelo pré-treinado (opcional)
        """
        self.samples = []
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue()
        self.is_running = False
        
        # Configuração do processamento paralelo
        self.num_threads = len(os.sched_getaffinity(0))
        
        # Configuração do modelo
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        else:
            self.model = self._create_model()
            
        # Configuração do logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _create_model(self) -> keras.Model:
        """
        Cria um modelo CNN para detecção de nível de água
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(224, 224, 3)),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Pré-processa o frame para análise
        """
        # Redimensiona para o tamanho esperado pelo modelo
        processed = cv2.resize(frame, (224, 224))
        
        # Normalização
        processed = processed.astype(np.float32) / 255.0
        
        # Detecção de bordas para destacar a linha d'água
        edges = cv2.Canny(
            (processed * 255).astype(np.uint8),
            threshold1=30,
            threshold2=100
        )
        
        # Combina a imagem original com as bordas detectadas
        processed = np.dstack([processed, edges/255.0])
        
        return processed

    async def process_frame(self, frame: np.ndarray) -> WaterLevelSample:
        """
        Processa um único frame e retorna uma amostra
        """
        processed_frame = self.preprocess_frame(frame)
        
        # Faz a predição usando o modelo
        prediction = self.model.predict(
            np.expand_dims(processed_frame, axis=0),
            verbose=0
        )[0][0]
        
        # Calcula a confiança baseada na qualidade da imagem
        edges = cv2.Canny(frame, 30, 100)
        confidence = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        sample = WaterLevelSample(
            timestamp=datetime.now(),
            water_level=float(prediction),
            confidence=float(confidence),
            image=frame
        )
        
        self.samples.append(sample)
        return sample

    def start_gopro_capture(self, ip_address: str, port: int = 8554):
        """
        Inicia a captura da GoPro via RTSP
        """
        self.is_running = True
        stream_url = f"rtsp://{ip_address}:{port}/live"
        
        def capture_frames():
            cap = cv2.VideoCapture(stream_url)
            while self.is_running:
                ret, frame = cap.read()
                if ret:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                else:
                    self.logger.error("Falha ao ler frame da GoPro")
                    break
            cap.release()
        
        def process_frames():
            while self.is_running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    sample = await self.process_frame(frame)
                    self.result_queue.put(sample)
        
        # Inicia threads para captura e processamento
        capture_thread = threading.Thread(target=capture_frames)
        process_threads = [
            threading.Thread(target=process_frames)
            for _ in range(self.num_threads)
        ]
        
        capture_thread.start()
        for thread in process_threads:
            thread.start()

    def stop_capture(self):
        """
        Para a captura e processamento
        """
        self.is_running = False

    async def train_model(self, labeled_data: List[WaterLevelSample], epochs: int = 10):
        """
        Treina o modelo com dados rotulados
        """
        X = np.array([self.preprocess_frame(sample.image) for sample in labeled_data])
        y = np.array([sample.water_level for sample in labeled_data])
        
        # Divide em conjunto de treino e validação
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Treina o modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history

    def save_model(self, path: str):
        """
        Salva o modelo treinado
        """
        self.model.save(path)
        self.logger.info(f"Modelo salvo em {path}")

# Função de exemplo de uso
async def main():
    # Inicializa o detector
    detector = WaterLevelDetector()
    
    # Configura a captura da GoPro
    detector.start_gopro_capture("192.168.1.100")  # Substitua pelo IP da sua GoPro
    
    try:
        # Executa por 60 segundos
        await asyncio.sleep(60)
        
        # Para a captura
        detector.stop_capture()
        
        # Treina o modelo com as amostras coletadas
        labeled_data = detector.samples  # Em um caso real, você precisaria rotular os dados
        await detector.train_model(labeled_data)
        
        # Salva o modelo
        detector.save_model("water_level_model.h5")
        
    except KeyboardInterrupt:
        detector.stop_capture()
        print("Captura interrompida pelo usuário")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
