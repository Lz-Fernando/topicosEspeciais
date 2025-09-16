#!/usr/bin/env python3
"""
Handler Alternativo para Reconhecimento Facial usando OpenCV
Versão simplificada compatível com Python 3.13 quando face_recognition não está disponível.
"""

import cv2
import numpy as np
import os
import pickle
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io


class OpenCVFaceHandler:
    """Handler alternativo usando apenas OpenCV para detecção facial."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializa o handler alternativo.
        
        Args:
            models_dir: Diretório para armazenar modelos
        """
        self.models_dir = models_dir
        
        # Arquivos para persistir dados
        self.faces_database_file = os.path.join(models_dir, "opencv_faces.pkl")
        
        # Dados das faces conhecidas (simplificado)
        self.known_faces: Dict[str, Dict[str, Any]] = {}
        
        # Configuração de logging
        self.logger = logging.getLogger(__name__)
        
        # Cria diretório se não existir
        os.makedirs(models_dir, exist_ok=True)
        
        # Inicializa detector de faces do OpenCV
        self.face_cascade = None
        self._load_opencv_classifier()
        
    def _load_opencv_classifier(self) -> None:
        """Carrega o classificador de faces do OpenCV."""
        try:
            # Tenta carregar o classificador Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                self.logger.error("Falha ao carregar classificador de faces")
            else:
                self.logger.info("Classificador de faces OpenCV carregado com sucesso")
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar classificador: {e}")
            
    def load_known_faces(self) -> bool:
        """
        Carrega faces conhecidas do arquivo de dados.
        
        Returns:
            True se carregou com sucesso, False caso contrário
        """
        try:
            if os.path.exists(self.faces_database_file):
                with open(self.faces_database_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                    
                self.logger.info(f"Carregadas {len(self.known_faces)} faces conhecidas")
            else:
                self.logger.info("Nenhum arquivo de faces encontrado, iniciando com base vazia")
                self.known_faces = {}
                
            return True
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar faces conhecidas: {e}")
            return False
            
    def save_known_faces(self) -> bool:
        """
        Salva faces conhecidas no arquivo de dados.
        
        Returns:
            True se salvou com sucesso, False caso contrário
        """
        try:
            with open(self.faces_database_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
                
            self.logger.info(f"Salvadas {len(self.known_faces)} faces conhecidas")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar faces conhecidas: {e}")
            return False
            
    def add_known_face(self, name: str, image_data: str) -> bool:
        """
        Adiciona uma nova face conhecida (versão simplificada).
        
        Args:
            name: Nome da pessoa
            image_data: Dados da imagem em base64
            
        Returns:
            True se adicionou com sucesso, False caso contrário
        """
        try:
            # Decodifica imagem base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
            # Converte para formato OpenCV
            if len(image_array.shape) == 3:
                opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                opencv_image = image_array
                
            # Detecta faces
            faces = self._detect_faces(opencv_image)
            
            if not faces:
                self.logger.warning(f"Nenhuma face encontrada na imagem para {name}")
                return False
                
            # Salva informações da face (simplificado - apenas primeira face)
            x, y, w, h = faces[0]
            face_roi = opencv_image[y:y+h, x:x+w]
            
            # Armazena dados da face
            self.known_faces[name] = {
                'face_roi': face_roi.tolist(),  # Converte para lista para serialização
                'coordinates': (x, y, w, h),
                'added_date': str(cv2.getTickCount())
            }
            
            self.logger.info(f"Face adicionada: {name}")
            return self.save_known_faces()
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar face conhecida {name}: {e}")
            return False
            
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta faces em uma imagem usando OpenCV.
        
        Args:
            image: Imagem OpenCV
            
        Returns:
            Lista de coordenadas das faces (x, y, w, h)
        """
        if self.face_cascade is None:
            return []
            
        try:
            # Converte para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detecta faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return [(x, y, w, h) for x, y, w, h in faces]
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de faces: {e}")
            return []
            
    def recognize_faces(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detecta faces em um frame (reconhecimento simplificado).
        
        Args:
            frame: Frame de vídeo (numpy array)
            
        Returns:
            Dicionário com resultados da detecção
        """
        try:
            # Detecta faces
            face_coords = self._detect_faces(frame)
            
            faces_found = []
            confidence_scores = []
            face_coordinates = []
            
            for x, y, w, h in face_coords:
                # Como não temos reconhecimento real, marca como "Pessoa Detectada"
                faces_found.append("Pessoa Detectada")
                confidence_scores.append(0.8)  # Confiança simulada
                
                face_coordinates.append({
                    'top': y,
                    'right': x + w,
                    'bottom': y + h,
                    'left': x
                })
            
            return {
                'faces': faces_found,
                'confidence': confidence_scores,
                'coordinates': face_coordinates,
                'total_faces': len(faces_found)
            }
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de faces: {e}")
            return {
                'faces': [],
                'confidence': [],
                'coordinates': [],
                'total_faces': 0,
                'error': str(e)
            }
            
    def draw_face_rectangles(self, frame: np.ndarray, recognition_result: Dict[str, Any]) -> np.ndarray:
        """
        Desenha retângulos nas faces detectadas.
        
        Args:
            frame: Frame original
            recognition_result: Resultado da detecção
            
        Returns:
            Frame com anotações
        """
        try:
            annotated_frame = frame.copy()
            
            coordinates = recognition_result.get('coordinates', [])
            faces = recognition_result.get('faces', [])
            confidence_scores = recognition_result.get('confidence', [])
            
            for i, (coords, name, confidence) in enumerate(zip(coordinates, faces, confidence_scores)):
                # Coordenadas do retângulo
                top = coords['top']
                right = coords['right']
                bottom = coords['bottom']
                left = coords['left']
                
                # Cor do retângulo
                color = (0, 255, 0)  # Verde para faces detectadas
                
                # Desenha retângulo
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                
                # Texto
                text = f"{name} ({confidence:.2f})"
                
                # Fundo para o texto
                cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
                # Texto
                cv2.putText(
                    annotated_frame, 
                    text, 
                    (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
            
            return annotated_frame
            
        except Exception as e:
            self.logger.error(f"Erro ao desenhar retângulos: {e}")
            return frame
            
    def get_known_faces_list(self) -> List[str]:
        """
        Retorna lista de nomes das faces conhecidas.
        
        Returns:
            Lista com nomes das pessoas conhecidas
        """
        return list(self.known_faces.keys())
        
    def get_faces_count(self) -> int:
        """
        Retorna número de faces conhecidas.
        
        Returns:
            Número de faces conhecidas
        """
        return len(self.known_faces)
        
    def remove_known_face(self, name: str) -> bool:
        """
        Remove uma face conhecida.
        
        Args:
            name: Nome da pessoa a ser removida
            
        Returns:
            True se removeu com sucesso, False caso contrário
        """
        try:
            if name in self.known_faces:
                del self.known_faces[name]
                self.save_known_faces()
                self.logger.info(f"Face removida: {name}")
                return True
            else:
                self.logger.warning(f"Face não encontrada para remoção: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao remover face {name}: {e}")
            return False
            
    def clear_all_faces(self) -> bool:
        """
        Remove todas as faces conhecidas.
        
        Returns:
            True se limpou com sucesso, False caso contrário
        """
        try:
            self.known_faces.clear()
            success = self.save_known_faces()
            
            if success:
                self.logger.info("Todas as faces conhecidas foram removidas")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Erro ao limpar faces conhecidas: {e}")
            return False
