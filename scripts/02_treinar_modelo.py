from ultralytics import YOLO

# Caminho para o dataset data.yaml exportado pelo Roboflow
CAMINHO_DATASET = "../yolo_dataset/data.yaml"

# Cria o modelo YOLOv8 com base no yolov8n (nano) pré-treinado
model = YOLO("yolov8n.pt")  # Pode trocar por 'yolov8s.pt' ou 'yolov8m.pt' se quiser mais precisão

# Treina o modelo com os dados fornecidos
model.train(
    data=CAMINHO_DATASET,     # Caminho para o YAML com classes, treino e validação
    epochs=30,                 # Quantas vezes ele verá o dataset completo (pode ajustar)
    imgsz=640,                 # Tamanho da imagem redimensionada (padrão 640x640)
    batch=8,                   # Quantas imagens treina de uma vez (ajuste conforme memória da máquina)
    name="modelo_produtos",   # Nome da pasta onde o modelo treinado será salvo
    device='cpu'                # Use 0 para GPU (se tiver), ou 'cpu' para CPU
)
