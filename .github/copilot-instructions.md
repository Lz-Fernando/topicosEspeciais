# Copilot Instructions

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

Este é um projeto Python de reconhecimento facial com arquitetura cliente-servidor usando sockets TCP.

## Contexto do Projeto:
- **Objetivo**: Sistema de reconhecimento facial para controle de acesso
- **Arquitetura**: Cliente-servidor com sockets TCP
- **Threading**: Uso de ThreadPool para conexões simultâneas
- **Hardware**: Preparado para Raspberry Pi (futuro)
- **Camera**: Captura via webcam do PC

## Padrões de Código:
- Use type hints em todas as funções
- Implemente tratamento de exceções adequado
- Siga PEP 8 para formatação
- Use logging para debug e monitoramento
- Docstrings em português para funções principais

## Bibliotecas Principais:
- `opencv-python` para captura de vídeo
- `face_recognition` para reconhecimento facial
- `concurrent.futures` para ThreadPool
- `socket` para comunicação TCP
- `threading` para sincronização

## Estrutura:
- `src/server.py` - Servidor principal com ThreadPool
- `src/client.py` - Cliente para testes
- `src/face_recognition_handler.py` - Lógica de reconhecimento
- `src/camera_handler.py` - Gerenciamento da câmera
- `models/` - Modelos treinados de faces
- `data/` - Imagens de treinamento
