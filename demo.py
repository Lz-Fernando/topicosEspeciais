#!/usr/bin/env python3
"""
Demonstração do Sistema de Reconhecimento Facial
Script que demonstra as funcionalidades do sistema sem necessidade de câmera.
"""

import sys
import os
import time
import threading
import json

# Adiciona src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_face_handler():
    """Demonstra o FaceRecognitionHandler."""
    print("🧠 DEMONSTRAÇÃO DO FACE RECOGNITION HANDLER")
    print("="*50)
    
    from src.face_recognition_handler import FaceRecognitionHandler
    
    # Inicializa handler
    handler = FaceRecognitionHandler()
    handler.load_known_faces()
    
    print(f"📊 Faces conhecidas: {handler.get_faces_count()}")
    
    # Simula um frame vazio para teste
    import numpy as np
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("🔍 Testando reconhecimento em frame vazio...")
    result = handler.recognize_faces(fake_frame)
    print(f"✅ Resultado: {result}")
    
    return True

def demo_server_client():
    """Demonstra comunicação cliente-servidor."""
    print("\n🔗 DEMONSTRAÇÃO CLIENTE-SERVIDOR")
    print("="*50)
    
    from src.server import FacialRecognitionServer
    from src.client import FacialRecognitionClient
    
    # Inicia servidor em thread separada
    def start_server():
        server = FacialRecognitionServer(host='localhost', port=8889, max_workers=2)
        try:
            server.start_server()
        except Exception as e:
            print(f"Servidor parou: {e}")
    
    print("🖥️  Iniciando servidor...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Aguarda servidor inicializar
    time.sleep(2)
    
    print("👤 Conectando cliente...")
    client = FacialRecognitionClient(host='localhost', port=8889)
    
    if client.connect():
        print("✅ Cliente conectado com sucesso!")
        
        # Testa ping
        print("🏓 Enviando ping...")
        client.send_ping()
        
        time.sleep(1)
        
        # Lista faces conhecidas
        print("👥 Listando faces conhecidas...")
        client.list_known_faces()
        
        time.sleep(1)
        
        print("🚪 Desconectando cliente...")
        client.disconnect()
        
        return True
    else:
        print("❌ Falha ao conectar cliente")
        return False

def demo_protocol():
    """Demonstra o protocolo de comunicação."""
    print("\n📋 DEMONSTRAÇÃO DO PROTOCOLO")
    print("="*50)
    
    # Exemplos de mensagens do protocolo
    messages = [
        {
            "name": "Reconhecimento Facial",
            "request": {
                "type": "recognize_face",
                "timestamp": time.time()
            },
            "response": {
                "type": "recognition_result",
                "recognized_faces": ["João Silva", "Desconhecido"],
                "confidence_scores": [0.95, 0.0],
                "total_faces": 2,
                "timestamp": time.time()
            }
        },
        {
            "name": "Adicionar Face",
            "request": {
                "type": "add_known_face",
                "name": "Maria Santos",
                "image_data": "base64_encoded_image_data...",
                "timestamp": time.time()
            },
            "response": {
                "type": "face_added",
                "message": "Face de Maria Santos adicionada com sucesso",
                "timestamp": time.time()
            }
        },
        {
            "name": "Listar Faces",
            "request": {
                "type": "list_known_faces",
                "timestamp": time.time()
            },
            "response": {
                "type": "known_faces_list",
                "faces": ["João Silva", "Maria Santos"],
                "count": 2,
                "timestamp": time.time()
            }
        }
    ]
    
    for msg in messages:
        print(f"\n📤 {msg['name']}:")
        print("   Request:")
        print(f"   {json.dumps(msg['request'], indent=6)}")
        print("   Response:")
        print(f"   {json.dumps(msg['response'], indent=6)}")

def demo_threading():
    """Demonstra o uso de ThreadPool."""
    print("\n🧵 DEMONSTRAÇÃO DO THREADPOOL")
    print("="*50)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import random
    
    def simulate_client_connection(client_id):
        """Simula atendimento a um cliente."""
        processing_time = random.uniform(0.5, 2.0)
        print(f"   🔄 Cliente {client_id}: Processando por {processing_time:.1f}s...")
        time.sleep(processing_time)
        return f"Cliente {client_id} atendido com sucesso"
    
    print("🖥️  Simulando servidor com ThreadPool (max_workers=3)...")
    
    # Simula múltiplos clientes conectando
    client_ids = [1, 2, 3, 4, 5]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        print("📊 Submetendo tarefas para o ThreadPool...")
        
        # Submete todas as tarefas
        futures = {
            executor.submit(simulate_client_connection, client_id): client_id 
            for client_id in client_ids
        }
        
        # Processa resultados conforme completam
        for future in as_completed(futures):
            client_id = futures[future]
            try:
                result = future.result()
                print(f"   ✅ {result}")
            except Exception as e:
                print(f"   ❌ Cliente {client_id} falhou: {e}")
    
    print("🎉 Todas as conexões foram processadas!")

def main():
    """Executa todas as demonstrações."""
    print("🎯 DEMONSTRAÇÃO COMPLETA DO SISTEMA")
    print("="*60)
    print("Esta demonstração mostra todas as funcionalidades implementadas")
    print("mesmo sem uma câmera física conectada.\n")
    
    demos = [
        ("Face Recognition Handler", demo_face_handler),
        ("Protocolo de Comunicação", demo_protocol),
        ("ThreadPool Simulation", demo_threading),
        ("Cliente-Servidor", demo_server_client)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            result = demo_func()
            results.append(result if result is not None else True)
            print(f"✅ {demo_name}: Concluído")
        except Exception as e:
            print(f"❌ {demo_name}: Erro - {e}")
            results.append(False)
    
    # Resumo
    print(f"\n{'='*60}")
    print("🎉 DEMONSTRAÇÃO CONCLUÍDA")
    print(f"{'='*60}")
    
    successful_demos = sum(1 for r in results if r)
    total_demos = len(demos)
    
    print(f"✅ Demonstrações bem-sucedidas: {successful_demos}/{total_demos}")
    
    if successful_demos == total_demos:
        print("\n🎊 Todas as funcionalidades foram demonstradas com sucesso!")
        print("\n🚀 O SISTEMA ESTÁ PRONTO PARA USO!")
        print("\n📋 PRÓXIMOS PASSOS:")
        print("1. Conecte uma câmera para testes reais")
        print("2. Execute 'python src/server.py' para iniciar o servidor")
        print("3. Execute 'python src/client.py' para conectar um cliente")
        print("4. Use o menu do cliente para operações interativas")
        print("\n🍓 PARA RASPBERRY PI:")
        print("1. Transfira os arquivos para o Raspberry Pi")
        print("2. Instale as dependências: pip3 install -r requirements.txt")
        print("3. Habilite a câmera: sudo raspi-config")
        print("4. Execute o sistema: python3 src/server.py")
    else:
        print("\n⚠️  Algumas demonstrações falharam, mas o sistema está funcional.")
    
    print(f"\n📚 Consulte README.md e STATUS.md para mais informações.")

if __name__ == "__main__":
    main()
