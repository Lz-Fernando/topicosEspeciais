#!/usr/bin/env python3
"""
Servidor de Reconhecimento Facial
Implementa arquitetura cliente-servidor com ThreadPool para conexões simultâneas.
"""

import socket
import threading
import logging
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
import time

from face_recognition_handler import FaceRecognitionHandler
from camera_handler import CameraHandler


class FacialRecognitionServer:
    """Servidor principal para reconhecimento facial com suporte a múltiplos clientes."""
    
    def __init__(self, host: str = 'localhost', port: int = 8888, max_workers: int = 5):
        """
        Inicializa o servidor.
        
        Args:
            host: Endereço IP do servidor
            port: Porta do servidor
            max_workers: Número máximo de threads no pool
        """
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server_socket: Optional[socket.socket] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.is_running = False
        
        # Handlers especializados
        self.face_handler = FaceRecognitionHandler()
        self.camera_handler = CameraHandler()
        
        # Controle de conexões ativas
        self.active_connections: Dict[str, socket.socket] = {}
        self.connection_lock = threading.Lock()
        
        # Configuração de logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configura o sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('server.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def start_server(self) -> None:
        """Inicia o servidor e aceita conexões."""
        try:
            # Configuração do socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            # Inicialização do ThreadPool
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
            self.is_running = True
            self.logger.info(f"Servidor iniciado em {self.host}:{self.port}")
            self.logger.info(f"ThreadPool configurado com {self.max_workers} workers")
            
            # Inicializa handlers
            self.camera_handler.initialize_camera()
            self.face_handler.load_known_faces()
            
            # Loop principal de aceitação de conexões
            while self.is_running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    self.logger.info(f"Nova conexão de {client_address}")
                    
                    # Submete a conexão para o ThreadPool
                    future = self.executor.submit(self.handle_client, client_socket, client_address)
                    
                    # Adiciona callback para limpeza quando a conexão terminar
                    future.add_done_callback(lambda f: self._cleanup_connection(client_address))
                    
                except socket.error as e:
                    if self.is_running:
                        self.logger.error(f"Erro ao aceitar conexão: {e}")
                        
        except Exception as e:
            self.logger.error(f"Erro ao iniciar servidor: {e}")
        finally:
            self.shutdown()
            
    def handle_client(self, client_socket: socket.socket, client_address: tuple) -> None:
        """
        Gerencia a comunicação com um cliente específico.
        
        Args:
            client_socket: Socket do cliente
            client_address: Endereço do cliente
        """
        client_id = f"{client_address[0]}:{client_address[1]}"
        
        try:
            # Registra a conexão
            with self.connection_lock:
                self.active_connections[client_id] = client_socket
                
            self.logger.info(f"Iniciando atendimento ao cliente {client_id}")
            
            # Envia mensagem de boas-vindas
            welcome_msg = {
                "type": "welcome",
                "message": "Conectado ao servidor de reconhecimento facial",
                "timestamp": time.time()
            }
            self._send_message(client_socket, welcome_msg)
            
            # Loop de comunicação com o cliente
            while self.is_running:
                try:
                    # Recebe dados do cliente
                    data = client_socket.recv(4096)
                    if not data:
                        break
                        
                    # Processa a mensagem
                    message = json.loads(data.decode('utf-8'))
                    response = self._process_client_message(message)
                    
                    # Envia resposta
                    self._send_message(client_socket, response)
                    
                except json.JSONDecodeError:
                    error_response = {
                        "type": "error",
                        "message": "Formato de mensagem inválido",
                        "timestamp": time.time()
                    }
                    self._send_message(client_socket, error_response)
                    
                except socket.timeout:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Erro ao atender cliente {client_id}: {e}")
        finally:
            self._disconnect_client(client_socket, client_id)
            
    def _process_client_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagens recebidas do cliente.
        
        Args:
            message: Mensagem do cliente
            
        Returns:
            Resposta para o cliente
        """
        message_type = message.get("type", "unknown")
        
        if message_type == "recognize_face":
            return self._handle_face_recognition()
            
        elif message_type == "capture_image":
            return self._handle_image_capture()
            
        elif message_type == "add_known_face":
            return self._handle_add_known_face(message)
            
        elif message_type == "list_known_faces":
            return self._handle_list_known_faces()
            
        elif message_type == "ping":
            return {
                "type": "pong",
                "timestamp": time.time()
            }
            
        else:
            return {
                "type": "error",
                "message": f"Tipo de mensagem não reconhecido: {message_type}",
                "timestamp": time.time()
            }
            
    def _handle_face_recognition(self) -> Dict[str, Any]:
        """Executa reconhecimento facial."""
        try:
            # Captura frame da câmera
            frame = self.camera_handler.capture_frame()
            if frame is None:
                return {
                    "type": "error",
                    "message": "Falha ao capturar imagem da câmera",
                    "timestamp": time.time()
                }
                
            # Executa reconhecimento
            result = self.face_handler.recognize_faces(frame)
            
            # Codifica imagem para envio (opcional)
            _, buffer = self.camera_handler.encode_frame(frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "type": "recognition_result",
                "recognized_faces": result["faces"],
                "confidence_scores": result["confidence"],
                "image_data": image_data,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Erro no reconhecimento facial: {e}")
            return {
                "type": "error",
                "message": f"Erro no reconhecimento: {str(e)}",
                "timestamp": time.time()
            }
            
    def _handle_image_capture(self) -> Dict[str, Any]:
        """Captura uma imagem da câmera."""
        try:
            frame = self.camera_handler.capture_frame()
            if frame is None:
                return {
                    "type": "error",
                    "message": "Falha ao capturar imagem",
                    "timestamp": time.time()
                }
                
            _, buffer = self.camera_handler.encode_frame(frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "type": "image_captured",
                "image_data": image_data,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "type": "error",
                "message": f"Erro na captura: {str(e)}",
                "timestamp": time.time()
            }
            
    def _handle_add_known_face(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Adiciona nova face conhecida."""
        try:
            name = message.get("name")
            image_data = message.get("image_data")
            
            if not name or not image_data:
                return {
                    "type": "error",
                    "message": "Nome e dados da imagem são obrigatórios",
                    "timestamp": time.time()
                }
                
            success = self.face_handler.add_known_face(name, image_data)
            
            if success:
                return {
                    "type": "face_added",
                    "message": f"Face de {name} adicionada com sucesso",
                    "timestamp": time.time()
                }
            else:
                return {
                    "type": "error",
                    "message": "Falha ao adicionar face",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            return {
                "type": "error",
                "message": f"Erro ao adicionar face: {str(e)}",
                "timestamp": time.time()
            }
            
    def _handle_list_known_faces(self) -> Dict[str, Any]:
        """Lista faces conhecidas."""
        try:
            known_faces = self.face_handler.get_known_faces_list()
            return {
                "type": "known_faces_list",
                "faces": known_faces,
                "count": len(known_faces),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "type": "error",
                "message": f"Erro ao listar faces: {str(e)}",
                "timestamp": time.time()
            }
            
    def _send_message(self, client_socket: socket.socket, message: Dict[str, Any]) -> None:
        """Envia mensagem para o cliente."""
        try:
            data = json.dumps(message).encode('utf-8')
            client_socket.send(data)
        except Exception as e:
            self.logger.error(f"Erro ao enviar mensagem: {e}")
            
    def _disconnect_client(self, client_socket: socket.socket, client_id: str) -> None:
        """Desconecta um cliente e limpa recursos."""
        try:
            client_socket.close()
            with self.connection_lock:
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
            self.logger.info(f"Cliente {client_id} desconectado")
        except Exception as e:
            self.logger.error(f"Erro ao desconectar cliente {client_id}: {e}")
            
    def _cleanup_connection(self, client_address: tuple) -> None:
        """Callback para limpeza quando uma conexão termina."""
        client_id = f"{client_address[0]}:{client_address[1]}"
        self.logger.info(f"Limpeza da conexão {client_id} completada")
        
    def get_server_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do servidor."""
        with self.connection_lock:
            return {
                "active_connections": len(self.active_connections),
                "max_workers": self.max_workers,
                "is_running": self.is_running,
                "connected_clients": list(self.active_connections.keys())
            }
            
    def shutdown(self) -> None:
        """Encerra o servidor de forma segura."""
        self.logger.info("Iniciando shutdown do servidor...")
        self.is_running = False
        
        # Fecha todas as conexões ativas
        with self.connection_lock:
            for client_id, client_socket in self.active_connections.items():
                try:
                    client_socket.close()
                    self.logger.info(f"Conexão {client_id} fechada")
                except Exception as e:
                    self.logger.error(f"Erro ao fechar conexão {client_id}: {e}")
            self.active_connections.clear()
        
        # Encerra o ThreadPool
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("ThreadPool encerrado")
            
        # Fecha socket do servidor
        if self.server_socket:
            self.server_socket.close()
            self.logger.info("Socket do servidor fechado")
            
        # Limpa recursos dos handlers
        self.camera_handler.cleanup()
        
        self.logger.info("Shutdown completado")


def main():
    """Função principal para executar o servidor."""
    server = FacialRecognitionServer(host='localhost', port=8888, max_workers=5)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
        server.shutdown()


if __name__ == "__main__":
    main()
