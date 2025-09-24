# 🔐 Sistema de Reconhecimento Facial com Arquitetura Cliente-Servidor

Sistema avançado de reconhecimento facial desenvolvido em Python com arquitetura cliente-servidor usando sockets TCP e ThreadPool para gerenciamento eficiente de múltiplas conexões simultâneas.

## 🎯 Objetivo

Desenvolver um sistema de controle de acesso baseado em reconhecimento facial, preparado para integração com Raspberry Pi e controle de fechaduras eletrônicas.

## ✨ Características Principais

- **🔧 Arquitetura Cliente-Servidor**: Comunicação via sockets TCP
- **⚡ ThreadPool**: Gerenciamento eficiente de conexões simultâneas
- **🧠 Reconhecimento Facial**: Caminho compatível usando OpenCV (LBPH); usa `face_recognition` se disponível
- **📹 Captura de Vídeo**: Suporte para câmera do PC e webcam
- **🔒 Controle de Acesso**: Janela de votação (maioria) para autorizar acesso
- **🍓 Raspberry Pi Ready**: Preparado para deploy em Raspberry Pi
- **📊 Logging Avançado**: Sistema completo de logs para monitoramento
- **🔄 Conexões Simultâneas**: Suporte para múltiplos clientes conectados

## 📁 Estrutura do Projeto

```
topicosEspeciais/
├── 📂 src/
│   ├── 🖥️  server.py                    # Servidor principal com ThreadPool
│   ├── 👤 client.py                     # Cliente interativo para testes
│   ├── 🧠 face_recognition_handler.py   # Lógica original (usa face_recognition, se disponível)
│   ├── 🧠 face_recognition_handler_compatible.py   # Modo compatível (OpenCV LBPH)
│   └── 📹 camera_handler.py             # Gerenciamento da câmera
├── 📂 models/                           # Modelos treinados de faces conhecidas
├── 📂 data/                             # Dados de treinamento
├── 📂 training_images/                  # Imagens para adicionar faces conhecidas
├── 📂 captured_images/                  # Imagens capturadas pelo sistema
├── 📂 logs/                             # Arquivos de log do sistema
├── 📂 .github/
│   └── 📝 copilot-instructions.md       # Instruções para o GitHub Copilot
├── 📋 requirements.txt                  # Dependências do projeto
├── 🧪 setup_example.py                  # Script de configuração e testes
└── 📖 README.md                         # Este arquivo
```

## 🚀 Instalação e Configuração

### 1. 📦 Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. 🧪 Executar Testes de Configuração

```bash
python setup_example.py
```

### 3. 📹 Testar Câmera

```bash
# Execute o script de configuração e escolha a opção de teste da câmera
python setup_example.py
```

## 🎮 Como Usar

### 🖥️ Executar o Servidor

```bash
cd src
python server.py
```

O servidor será iniciado em `localhost:8888` com as seguintes características:
- **ThreadPool**: 5 workers por padrão
- **Logging**: Logs salvos em `server.log`
- **Conexões Simultâneas**: Suporte para múltiplos clientes

### 👤 Executar o Cliente

```bash
cd src
python client.py
```

O cliente oferece um menu interativo com as seguintes opções:
1. **🔍 Reconhecer Face**: Captura e analisa faces do frame atual
2. **➕ Adicionar Face Conhecida**: Coleta guiada (6 passos × 3 fotos) ou importa de pasta/arquivo
3. **👥 Listar Faces Conhecidas**: Mostra todas as pessoas cadastradas
4. **🏓 Ping**: Testa conectividade com o servidor
5. **🛠️ Treinar modelo (LBPH)**: Re-treina com as imagens em `data/<nome>/`
6. **🤖 Reconhecer e identificar (LBPH)**: Predição com limiar configurável
7. **🧹 Limpar modelo**: Limpa dataset/modelos
8. **🔐 Autorizar acesso (votação)**: Janela de votação com parâmetros configuráveis
9. **🚪 Sair**

Notas:
- A opção “Capturar Imagem” foi removida em favor do fluxo de coleta guiada ou importação por pasta.
- As imagens coletadas ficam em `data/<nome>/` e os modelos em `models/`.

### 🧭 Coleta Guiada de Dataset

- 6 passos com instruções (frente, esquerda, direita, cima, baixo, expressão)
- 3 fotos por passo (total 18), salvas em `data/<nome>/`
- Alternativamente, importe fotos de um diretório com imagens do rosto

### 🤖 Treino e Predição (LBPH)

- Após coletar dados, use “Treinar modelo (LBPH)”
- A predição utiliza um limiar (`LBPH_THRESHOLD`) para decidir se um rosto conhecido é aceito
- As imagens de predição são salvas em `captured_images/`

### 🔐 Autorização por Votação (2/3 por padrão)

- Captura N frames (padrão 3) e exige R acertos (padrão 2) abaixo do limiar para permitir
- Parâmetros configuráveis no cliente: quantidade de frames, votos necessários, limiar
- Útil para reduzir falsos positivos em ambientes variáveis

## 🔧 Funcionalidades Técnicas

### 🔌 Arquitetura Cliente-Servidor

- **Protocolo**: TCP Sockets
- **Formato de Dados**: JSON
- **Encoding**: UTF-8
- **Threading**: ThreadPoolExecutor para conexões simultâneas

### 🧠 Reconhecimento Facial

- Modo compatível com OpenCV (LBPH) ativado por padrão
- Se `face_recognition` estiver instalado, o handler original pode ser usado
- Limiar de decisão LBPH configurável via `LBPH_THRESHOLD`

### 📹 Gerenciamento de Câmera

- **Captura**: OpenCV VideoCapture
- **Threading**: Captura contínua em thread separada
- **Buffer**: Queue para frames com controle de latência
- **Formatos**: Suporte para JPEG e PNG

### 🔄 Gerenciamento de Conexões

```python
# Exemplo de uso do ThreadPool no servidor
self.executor = ThreadPoolExecutor(max_workers=5)
future = self.executor.submit(self.handle_client, client_socket, client_address)
```

## 📋 Protocolo de Comunicação

### 📤 Mensagens do Cliente para Servidor

```json
{
    "type": "recognize_face",
    "timestamp": 1234567890.123
}
```

```json
{
    "type": "add_known_face",
    "name": "João Silva",
    "image_data": "base64_encoded_image",
    "timestamp": 1234567890.123
}
```

### 📥 Respostas do Servidor

```json
{
    "type": "recognition_result",
    "recognized_faces": ["João Silva", "Desconhecido"],
    "confidence_scores": [0.95, 0.0],
    "image_data": "base64_encoded_image",
    "timestamp": 1234567890.123
}
```

## 🍓 Configuração para Raspberry Pi

### 1. 🔧 Habilitar Câmera

```bash
sudo raspi-config
# Interface Options -> Camera -> Enable
```

### 2. 📦 Instalar Dependências

```bash
sudo apt update
sudo apt install python3-opencv python3-pip cmake build-essential
pip3 install -r requirements.txt
```

### 3. 🔌 Configuração GPIO (Futuro - Fechadura)

```python
import RPi.GPIO as GPIO

# Pino para controle da fechadura
LOCK_PIN = 18

def unlock_door():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LOCK_PIN, GPIO.OUT)
    GPIO.output(LOCK_PIN, GPIO.HIGH)
    time.sleep(2)  # Mantém desbloqueado por 2 segundos
    GPIO.output(LOCK_PIN, GPIO.LOW)
    GPIO.cleanup()
```

## 🔧 Configurações Avançadas

### ⚙️ Parâmetros do Servidor

```python
server = FacialRecognitionServer(
    host='0.0.0.0',        # Aceita conexões de qualquer IP
    port=8888,             # Porta do servidor
    max_workers=10         # Número máximo de threads
)
```

### ⚙️ Configuração via .env (opcional)

Crie um arquivo `.env` na raiz com variáveis (todas possuem defaults):

```
SERVER_HOST=localhost
SERVER_PORT=8888
MAX_WORKERS=5

# Câmera
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

# Pastas
MODELS_DIR=models
DATA_DIR=data
LOG_DIR=logs

# LBPH
LBPH_THRESHOLD=65.0
```

Observações:
- LBPH_THRESHOLD menor → mais restritivo (menos falsos positivos, mais falsos negativos)
- Ajuste conforme iluminação/qualidade das imagens

### ⚙️ Parâmetros da Câmera

```python
camera = CameraHandler(
    camera_index=0,                # Índice da câmera
    resolution=(640, 480)          # Resolução da captura
)
```

## 📊 Monitoramento e Logs

O sistema gera logs detalhados em:
- `server.log` - Logs do servidor
- `client.log` - Logs do cliente
- `logs/system.log` - Logs gerais do sistema

### 📈 Estatísticas do Servidor

```python
stats = server.get_server_stats()
print(f"Conexões ativas: {stats['active_connections']}")
print(f"Clientes conectados: {stats['connected_clients']}")
```

## 🧪 Testes e Debugging

### 🔍 Executar Testes Completos

```bash
python setup_example.py
# Escolha opção 1 para testes completos
```

### 🐛 Debug Mode

Para ativar modo debug, modifique o nível de logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Roadmap Futuro

- [ ] 🔐 Integração com fechadura eletrônica
- [ ] 🌐 Interface web para administração
- [ ] 📱 Aplicativo móvel para controle
- [ ] 🔄 Backup automático de dados
- [ ] 📧 Notificações por email/SMS
- [ ] 🎯 Detecção de tentativas de invasão
- [ ] 📊 Dashboard de estatísticas
- [ ] 🔒 Criptografia de comunicação

## 🤝 Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🆘 Suporte

Se você encontrar problemas:

1. **🔍 Verifique os logs** em `logs/` e `*.log`
2. **🧪 Execute os testes** com `python setup_example.py`
3. **📖 Consulte a documentação** neste README
4. **🐛 Abra uma issue** descrevendo o problema

## 📞 Contato

- **Desenvolvedor**: João Pedro
- **Projeto**: Tópicos Especiais em Computação
- **Objetivo**: Sistema de Controle de Acesso com Reconhecimento Facial

---

⭐ **Se este projeto foi útil para você, considere dar uma estrela!** ⭐
