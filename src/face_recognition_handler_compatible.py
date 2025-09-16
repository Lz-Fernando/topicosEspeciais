#!/usr/bin/env python3
"""
Handler Compatível para Reconhecimento Facial
Módulo responsável pelo processamento de reconhecimento facial com fallback para OpenCV.
"""

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

import cv2
import numpy as np
import os
import pickle
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io


class FaceRecognitionHandler:
    """Gerencia o reconhecimento facial e faces conhecidas."""
    
    def __init__(self, models_dir: str = "models", tolerance: float = 0.6):
        """
        Inicializa o handler de reconhecimento facial.
        
        Args:
            models_dir: Diretório para armazenar modelos
            tolerance: Tolerância para reconhecimento (menor = mais restritivo)
        """
        self.models_dir = models_dir
        self.tolerance = tolerance
        
        # Configuração de logging
        self.logger = logging.getLogger(__name__)
        
        # Cria diretório se não existir
        os.makedirs(models_dir, exist_ok=True)
        
        if FACE_RECOGNITION_AVAILABLE:
            self.logger.info("Usando face_recognition library")
            self._init_face_recognition()
        else:
            self.logger.warning("face_recognition não disponível, usando OpenCV")
            self._init_opencv_detection()
            
    def _init_face_recognition(self) -> None:
        """Inicializa com face_recognition library."""
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.faces_database_file = os.path.join(self.models_dir, "known_faces.pkl")
        
    def _init_opencv_detection(self) -> None:
        """Inicializa detecção com OpenCV apenas."""
        self.known_faces: Dict[str, Dict[str, Any]] = {}
        self.faces_database_file = os.path.join(self.models_dir, "opencv_faces.pkl")
        
        # Carrega classificador de faces
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.logger.error("Falha ao carregar classificador de faces")
            else:
                self.logger.info("Classificador OpenCV carregado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao carregar classificador: {e}")
            self.face_cascade = None
        
    def load_known_faces(self) -> bool:
        """
        Carrega faces conhecidas do arquivo de dados.
        
        Returns:
            True se carregou com sucesso, False caso contrário
        """
        try:
            if os.path.exists(self.faces_database_file):
                with open(self.faces_database_file, 'rb') as f:
                    data = pickle.load(f)
                    
                if FACE_RECOGNITION_AVAILABLE:
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    count = len(self.known_face_names)
                else:
                    if isinstance(data, dict) and 'encodings' in data:
                        # Arquivo do face_recognition, converte para formato OpenCV
                        self.known_faces = {}
                    else:
                        self.known_faces = data if isinstance(data, dict) else {}
                    count = len(self.known_faces)
                    
                self.logger.info(f"Carregadas {count} faces conhecidas")
            else:
                self.logger.info("Nenhum arquivo de faces encontrado, iniciando com base vazia")
                if FACE_RECOGNITION_AVAILABLE:
                    self.known_face_encodings = []
                    self.known_face_names = []
                else:
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
            if FACE_RECOGNITION_AVAILABLE:
                data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }
                count = len(self.known_face_names)
            else:
                data = self.known_faces
                count = len(self.known_faces)
            
            with open(self.faces_database_file, 'wb') as f:
                pickle.dump(data, f)
                
            self.logger.info(f"Salvadas {count} faces conhecidas")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar faces conhecidas: {e}")
            return False
            
    def add_known_face(self, name: str, image_data: str) -> bool:
        """
        Adiciona uma nova face conhecida.
        
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
            
            if FACE_RECOGNITION_AVAILABLE:
                return self._add_face_with_recognition(name, image_array)
            else:
                return self._add_face_with_opencv(name, image_array)
                
        except Exception as e:
            self.logger.error(f"Erro ao adicionar face conhecida {name}: {e}")
            return False
            
    def _add_face_with_recognition(self, name: str, image_array: np.ndarray) -> bool:
        """Adiciona face usando face_recognition."""
        # Converte para RGB se necessário
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_image = image_array
        else:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Encontra faces na imagem
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            self.logger.warning(f"Nenhuma face encontrada na imagem para {name}")
            return False
            
        # Gera encoding da primeira face encontrada
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if face_encodings:
            if name in self.known_face_names:
                # Atualiza encoding existente
                index = self.known_face_names.index(name)
                self.known_face_encodings[index] = face_encodings[0]
                self.logger.info(f"Encoding atualizado para {name}")
            else:
                # Adiciona nova pessoa
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                self.logger.info(f"Nova face adicionada: {name}")
            
            return self.save_known_faces()
        else:
            self.logger.error(f"Falha ao gerar encoding para {name}")
            return False
            
    def _add_face_with_opencv(self, name: str, image_array: np.ndarray) -> bool:
        """Adiciona face usando apenas OpenCV."""
        # Converte para formato OpenCV
        if len(image_array.shape) == 3:
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            opencv_image = image_array
            
        # Detecta faces
        faces = self._detect_faces_opencv(opencv_image)
        
        if not faces:
            self.logger.warning(f"Nenhuma face encontrada na imagem para {name}")
            return False
            
        # Salva dados da face (simplificado)
        x, y, w, h = faces[0]
        self.known_faces[name] = {
            'coordinates': (x, y, w, h),
            'added_date': str(cv2.getTickCount())
        }
        
        self.logger.info(f"Face adicionada: {name}")
        return self.save_known_faces()
        
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta faces usando OpenCV."""
        if self.face_cascade is None:
            return []
            
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        Reconhece faces em um frame.
        
        Args:
            frame: Frame de vídeo (numpy array)
            
        Returns:
            Dicionário com resultados do reconhecimento
        """
        try:
            if FACE_RECOGNITION_AVAILABLE:
                return self._recognize_with_face_recognition(frame)
            else:
                return self._recognize_with_opencv(frame)
                
        except Exception as e:
            self.logger.error(f"Erro no reconhecimento facial: {e}")
            return {
                'faces': [],
                'confidence': [],
                'coordinates': [],
                'total_faces': 0,
                'error': str(e)
            }
            
    def _recognize_with_face_recognition(self, frame: np.ndarray) -> Dict[str, Any]:
        """Reconhece usando face_recognition."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        faces_found = []
        confidence_scores = []
        face_coordinates = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=self.tolerance
            )
            
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, 
                face_encoding
            )
            
            name = "Desconhecido"
            confidence = 0.0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = max(0, 1 - face_distances[best_match_index])
            
            faces_found.append(name)
            confidence_scores.append(float(confidence))
            
            top, right, bottom, left = face_location
            face_coordinates.append({
                'top': top * 4,
                'right': right * 4, 
                'bottom': bottom * 4,
                'left': left * 4
            })
        
        return {
            'faces': faces_found,
            'confidence': confidence_scores,
            'coordinates': face_coordinates,
            'total_faces': len(faces_found)
        }
        
    def _recognize_with_opencv(self, frame: np.ndarray) -> Dict[str, Any]:
        """Reconhece usando apenas OpenCV (detecção)."""
        face_coords = self._detect_faces_opencv(frame)
        
        faces_found = []
        confidence_scores = []
        face_coordinates = []
        
        for x, y, w, h in face_coords:
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
        
    def draw_face_rectangles(self, frame: np.ndarray, recognition_result: Dict[str, Any]) -> np.ndarray:
        """
        Desenha retângulos e nomes nas faces detectadas.
        
        Args:
            frame: Frame original
            recognition_result: Resultado do reconhecimento
            
        Returns:
            Frame com anotações
        """
        try:
            annotated_frame = frame.copy()
            
            faces = recognition_result.get('faces', [])
            coordinates = recognition_result.get('coordinates', [])
            confidence_scores = recognition_result.get('confidence', [])
            
            for i, (name, coords, confidence) in enumerate(zip(faces, coordinates, confidence_scores)):
                top = coords['top']
                right = coords['right']
                bottom = coords['bottom']
                left = coords['left']
                
                color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
                
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                
                text = f"{name}"
                if name != "Desconhecido":
                    text += f" ({confidence:.2f})"
                
                cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
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
        """Retorna lista de nomes das faces conhecidas."""
        if FACE_RECOGNITION_AVAILABLE:
            return self.known_face_names.copy()
        else:
            return list(self.known_faces.keys())
        
    def get_faces_count(self) -> int:
        """Retorna número de faces conhecidas."""
        if FACE_RECOGNITION_AVAILABLE:
            return len(self.known_face_names)
        else:
            return len(self.known_faces)
        
    def remove_known_face(self, name: str) -> bool:
        """Remove uma face conhecida."""
        try:
            if FACE_RECOGNITION_AVAILABLE:
                if name in self.known_face_names:
                    index = self.known_face_names.index(name)
                    del self.known_face_names[index]
                    del self.known_face_encodings[index]
                    self.save_known_faces()
                    self.logger.info(f"Face removida: {name}")
                    return True
            else:
                if name in self.known_faces:
                    del self.known_faces[name]
                    self.save_known_faces()
                    self.logger.info(f"Face removida: {name}")
                    return True
                    
            self.logger.warning(f"Face não encontrada para remoção: {name}")
            return False
                
        except Exception as e:
            self.logger.error(f"Erro ao remover face {name}: {e}")
            return False
            
    def clear_all_faces(self) -> bool:
        """Remove todas as faces conhecidas."""
        try:
            if FACE_RECOGNITION_AVAILABLE:
                self.known_face_encodings.clear()
                self.known_face_names.clear()
            else:
                self.known_faces.clear()
            
            success = self.save_known_faces()
            
            if success:
                self.logger.info("Todas as faces conhecidas foram removidas")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Erro ao limpar faces conhecidas: {e}")
            return False
