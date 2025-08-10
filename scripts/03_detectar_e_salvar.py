import cv2
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import os
import uuid
import time  # Controle do cooldown

# ==== CONFIGURAÇÕES ====

# Caminho do modelo treinado
CAMINHO_MODELO = "runs/detect/modelo_produtos/weights/best.pt"

# Caminho do banco de dados
CAMINHO_BANCO = "../database/produtos.db"

# Tempo mínimo (em segundos) para registrar novamente o mesmo produto
TEMPO_ESPERA = 10

# =======================

# Armazena o último tempo que cada produto foi registrado
ultimo_registro = {}

# 1️⃣ Conectar/criar o banco de dados e tabela
def criar_banco():
    os.makedirs(os.path.dirname(CAMINHO_BANCO), exist_ok=True)
    conexao = sqlite3.connect(CAMINHO_BANCO)
    cursor = conexao.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS produtos_detectados (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT,
        codigo TEXT,
        data_hora TEXT
    )
    """)
    conexao.commit()
    conexao.close()

# 2️⃣ Função para salvar no banco
def salvar_produto(nome_produto):
    conexao = sqlite3.connect(CAMINHO_BANCO)
    cursor = conexao.cursor()

    codigo_produto = str(uuid.uuid4())[:8]  # gera um código único curto
    data_hora_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO produtos_detectados (nome, codigo, data_hora)
    VALUES (?, ?, ?)
    """, (nome_produto, codigo_produto, data_hora_atual))

    conexao.commit()
    conexao.close()
    print(f"✅ Produto salvo: {nome_produto} | Código: {codigo_produto} | Hora: {data_hora_atual}")

# 3️⃣ Carregar modelo YOLO
def carregar_modelo():
    if not os.path.exists(CAMINHO_MODELO):
        print("❌ Modelo não encontrado. Treine primeiro!")
        exit()
    return YOLO(CAMINHO_MODELO)

# 4️⃣ Função principal de detecção
def detectar_e_registrar():
    criar_banco()
    model = carregar_modelo()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=0.5)  # conf = confiança mínima

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])  # id da classe detectada
                nome_classe = model.names[cls_id]  # nome do produto

                agora = time.time()
                if nome_classe not in ultimo_registro or (agora - ultimo_registro[nome_classe] > TEMPO_ESPERA):
                    salvar_produto(nome_classe)
                    ultimo_registro[nome_classe] = agora

                # Desenha caixa na imagem
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, nome_classe, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Detecção de Produtos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🔹 Iniciar
if __name__ == "__main__":
    detectar_e_registrar()
