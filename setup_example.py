#!/usr/bin/env python3
"""
Script de Configuração e Exemplo do Sistema de Reconhecimento Facial
Demonstra como configurar e usar o sistema de reconhecimento facial.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.server import FacialRecognitionServer
    from src.client import FacialRecognitionClient
    from src.face_recognition_handler import FaceRecognitionHandler
    from src.camera_handler import CameraHandler
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Aviso de importação: {e}")
    print("📦 Algumas funcionalidades podem não estar disponíveis")
    print("💡 Execute: pip install -r requirements.txt")
    MODULES_AVAILABLE = False


class FaceRecognitionSetup:
    """Classe para configuração inicial do sistema."""
    
    def __init__(self):
        """Inicializa a configuração."""
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self) -> None:
        """Cria diretórios necessários."""
        directories = [
            "models",
            "data",
            "logs",
            "captured_images",
            "training_images"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("📁 Diretórios criados com sucesso")
        
    def setup_logging(self) -> None:
        """Configura sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system.log'),
                logging.StreamHandler()
            ]
        )
        
    def test_camera(self) -> bool:
        """Testa se a câmera está funcionando."""
        print("📹 Testando câmera...")
        
        if not MODULES_AVAILABLE:
            print("❌ Módulos não disponíveis - execute pip install -r requirements.txt")
            return False
        
        try:
            camera = CameraHandler()
            
            if camera.initialize_camera():
                print("✅ Câmera inicializada com sucesso")
                
                # Captura frame de teste
                frame = camera.capture_frame()
                if frame is not None:
                    print(f"✅ Frame capturado: {frame.shape}")
                    
                    # Salva imagem de teste
                    success, encoded = camera.encode_frame(frame)
                    if success:
                        with open("captured_images/camera_test.jpg", "wb") as f:
                            f.write(encoded)
                        print("✅ Imagem de teste salva em 'captured_images/camera_test.jpg'")
                        
                    camera.cleanup()
                    return True
                else:
                    print("❌ Falha ao capturar frame")
                    return False
            else:
                print("❌ Falha ao inicializar câmera")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao testar câmera: {e}")
            return False
            
    def test_face_recognition(self) -> bool:
        """Testa o sistema de reconhecimento facial."""
        print("🧠 Testando sistema de reconhecimento facial...")
        
        if not MODULES_AVAILABLE:
            print("❌ Módulos não disponíveis - execute pip install -r requirements.txt")
            return False
        
        try:
            face_handler = FaceRecognitionHandler()
            
            if face_handler.load_known_faces():
                print("✅ Sistema de reconhecimento inicializado")
                
                # Mostra estatísticas
                count = face_handler.get_faces_count()
                print(f"📊 Faces conhecidas: {count}")
                
                if count > 0:
                    faces = face_handler.get_known_faces_list()
                    print(f"👥 Lista: {', '.join(faces)}")
                    
                return True
            else:
                print("❌ Falha ao inicializar reconhecimento facial")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao testar reconhecimento facial: {e}")
            return False
            
    def create_sample_data(self) -> None:
        """Cria dados de exemplo para demonstração."""
        print("📝 Criando estrutura de exemplo...")
        
        # Cria arquivo README no diretório de imagens de treinamento
        readme_content = """# Imagens de Treinamento

Este diretório deve conter imagens para treinar o sistema de reconhecimento facial.

## Estrutura Recomendada:
```
training_images/
├── pessoa1/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── foto3.jpg
├── pessoa2/
│   ├── foto1.jpg
│   └── foto2.jpg
└── ...
```

## Dicas para Melhores Resultados:
- Use imagens de boa qualidade (mínimo 300x300 pixels)
- Certifique-se de que o rosto está bem iluminado
- Evite óculos escuros ou chapéus que cubram o rosto
- Use múltiplas fotos de ângulos diferentes
- Uma única face por imagem funciona melhor

## Como Adicionar Faces:
1. Coloque as imagens no diretório training_images/
2. Use o cliente para adicionar faces conhecidas
3. Ou use a função add_face_from_file() no código
"""
        
        with open("training_images/README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
            
        print("✅ Arquivo README criado em training_images/")
        
    def run_full_test(self) -> bool:
        """Executa teste completo do sistema."""
        print("\n" + "="*60)
        print("🧪 TESTE COMPLETO DO SISTEMA")
        print("="*60)
        
        tests = [
            ("Teste da Câmera", self.test_camera),
            ("Teste do Reconhecimento Facial", self.test_face_recognition)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            print(f"\n🔍 {test_name}:")
            result = test_func()
            results.append(result)
            
            if result:
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
                
        # Resumo
        print(f"\n📊 RESUMO DOS TESTES:")
        passed = sum(results)
        total = len(results)
        
        print(f"✅ Testes Passaram: {passed}/{total}")
        
        if passed == total:
            print("🎉 Todos os testes passaram! Sistema pronto para uso.")
            return True
        else:
            print("⚠️  Alguns testes falharam. Verifique a configuração.")
            return False


def show_usage_examples():
    """Mostra exemplos de uso do sistema."""
    print("\n" + "="*60)
    print("📚 EXEMPLOS DE USO")
    print("="*60)
    
    examples = [
        {
            "title": "🖥️  Executar Servidor",
            "command": "python src/server.py",
            "description": "Inicia o servidor de reconhecimento facial"
        },
        {
            "title": "👤 Executar Cliente",
            "command": "python src/client.py",
            "description": "Inicia cliente interativo para testar o sistema"
        },
        {
            "title": "🧪 Executar Testes",
            "command": "python setup_example.py",
            "description": "Executa configuração e testes do sistema"
        },
        {
            "title": "📦 Instalar Dependências",
            "command": "pip install -r requirements.txt",
            "description": "Instala todas as dependências necessárias"
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print(f"   Comando: {example['command']}")
        print(f"   Descrição: {example['description']}")
        
    print("\n" + "="*60)
    print("🔧 CONFIGURAÇÃO PARA RASPBERRY PI")
    print("="*60)
    
    pi_setup = """
1. Habilitar câmera no Raspberry Pi:
   sudo raspi-config -> Interface Options -> Camera -> Enable

2. Instalar dependências do OpenCV:
   sudo apt update
   sudo apt install python3-opencv python3-pip
   
3. Instalar CMake e dlib:
   sudo apt install cmake build-essential
   pip3 install dlib
   
4. Configurar GPIO para fechadura (futuro):
   sudo apt install python3-rpi.gpio
   
5. Executar sistema:
   python3 src/server.py
"""
    
    print(pi_setup)


def main():
    """Função principal do script de configuração."""
    print("🔐 SISTEMA DE RECONHECIMENTO FACIAL")
    print("Configuração e Testes")
    print("="*50)
    
    # Inicializa configuração
    setup = FaceRecognitionSetup()
    
    print("\n📋 MENU DE OPÇÕES:")
    print("1. 🧪 Executar testes completos")
    print("2. 📹 Testar apenas câmera")
    print("3. 🧠 Testar apenas reconhecimento facial")
    print("4. 📝 Criar estrutura de exemplo")
    print("5. 📚 Mostrar exemplos de uso")
    print("6. 🚪 Sair")
    
    while True:
        try:
            choice = input("\n👆 Escolha uma opção (1-6): ").strip()
            
            if choice == "1":
                setup.run_full_test()
                
            elif choice == "2":
                setup.test_camera()
                
            elif choice == "3":
                setup.test_face_recognition()
                
            elif choice == "4":
                setup.create_sample_data()
                
            elif choice == "5":
                show_usage_examples()
                
            elif choice == "6":
                print("👋 Encerrando...")
                break
                
            else:
                print("❌ Opção inválida")
                
        except KeyboardInterrupt:
            print("\n👋 Encerrando...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")


if __name__ == "__main__":
    main()
