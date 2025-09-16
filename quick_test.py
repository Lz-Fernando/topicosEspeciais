#!/usr/bin/env python3
"""
Teste Rápido do Sistema
Script para testar rapidamente se o sistema está funcionando.
"""

import sys
import os

# Adiciona src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Testa se todas as importações estão funcionando."""
    print("🔍 Testando importações...")
    
    try:
        from src.face_recognition_handler import FaceRecognitionHandler
        print("✅ FaceRecognitionHandler importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar FaceRecognitionHandler: {e}")
        return False
        
    try:
        from src.camera_handler import CameraHandler
        print("✅ CameraHandler importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar CameraHandler: {e}")
        return False
        
    try:
        from src.server import FacialRecognitionServer
        print("✅ FacialRecognitionServer importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar FacialRecognitionServer: {e}")
        return False
        
    try:
        from src.client import FacialRecognitionClient
        print("✅ FacialRecognitionClient importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar FacialRecognitionClient: {e}")
        return False
        
    return True

def test_face_handler():
    """Testa o handler de reconhecimento facial."""
    print("\n🧠 Testando FaceRecognitionHandler...")
    
    try:
        from src.face_recognition_handler import FaceRecognitionHandler
        
        handler = FaceRecognitionHandler()
        
        if handler.load_known_faces():
            print("✅ Sistema de reconhecimento inicializado")
            
            count = handler.get_faces_count()
            print(f"📊 Faces conhecidas: {count}")
            
            if count > 0:
                faces = handler.get_known_faces_list()
                print(f"👥 Lista: {', '.join(faces)}")
                
            return True
        else:
            print("❌ Falha ao inicializar reconhecimento")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste do face handler: {e}")
        return False

def test_camera():
    """Testa o handler da câmera."""
    print("\n📹 Testando CameraHandler...")
    
    try:
        from src.camera_handler import CameraHandler
        
        camera = CameraHandler()
        
        print("🔍 Tentando inicializar câmera...")
        if camera.initialize_camera():
            print("✅ Câmera inicializada com sucesso")
            
            info = camera.get_camera_info()
            print(f"📊 Info da câmera: {info}")
            
            print("📸 Tentando capturar frame...")
            frame = camera.capture_frame()
            
            if frame is not None:
                print(f"✅ Frame capturado: {frame.shape}")
                
                # Testa codificação
                success, encoded = camera.encode_frame(frame)
                if success:
                    print(f"✅ Frame codificado: {len(encoded)} bytes")
                    
                    # Salva imagem de teste
                    with open("captured_images/quick_test.jpg", "wb") as f:
                        f.write(encoded)
                    print("✅ Imagem salva em 'captured_images/quick_test.jpg'")
                else:
                    print("❌ Falha ao codificar frame")
                    
                camera.cleanup()
                return True
            else:
                print("❌ Falha ao capturar frame")
                camera.cleanup()
                return False
        else:
            print("❌ Falha ao inicializar câmera")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste da câmera: {e}")
        return False

def main():
    """Executa todos os testes."""
    print("🧪 TESTE RÁPIDO DO SISTEMA")
    print("="*40)
    
    tests = [
        ("Importações", test_imports),
        ("Face Handler", test_face_handler),
        ("Câmera", test_camera)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append(result)
        
        if result:
            print(f"✅ {test_name}: SUCESSO")
        else:
            print(f"❌ {test_name}: FALHOU")
    
    # Resumo
    print(f"\n{'='*40}")
    print("📊 RESUMO DOS TESTES")
    print(f"{'='*40}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Testes que passaram: {passed}/{total}")
    
    if passed == total:
        print("🎉 Todos os testes passaram!")
        print("\n🚀 PRÓXIMOS PASSOS:")
        print("1. Execute 'python src/server.py' para iniciar o servidor")
        print("2. Em outro terminal, execute 'python src/client.py' para conectar")
        print("3. Use o menu do cliente para testar o reconhecimento facial")
        return True
    else:
        print("⚠️  Alguns testes falharam.")
        print("\n🔧 SOLUÇÕES:")
        if not results[0]:
            print("- Instale as dependências: pip install -r requirements.txt")
        if not results[2]:
            print("- Verifique se a câmera está conectada e funcionando")
            print("- Teste com uma câmera externa se a webcam não funcionar")
        return False

if __name__ == "__main__":
    main()
