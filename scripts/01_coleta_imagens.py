import cv2
import os

# Nome do produto que estamos capturando (altere para cada item)
produto_nome = "borracha"

# Caminho onde vamos salvar as imagens
caminho_saida = f"../dataset/produto_2"

# Se a pasta não existir, criamos
os.makedirs(caminho_saida, exist_ok=True)

# Abrimos a câmera (0 = webcam padrão)
cap = cv2.VideoCapture(0)

# Verificamos se a câmera abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

contador = 0
print("Pressione 's' para salvar uma imagem. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()  # Lê o frame da webcam

    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Mostramos o frame ao usuário
    cv2.imshow("Coleta de imagens - Produto", frame)

    # Espera uma tecla
    tecla = cv2.waitKey(1)

    if tecla == ord("s"):  # Se pressionar 's', salva imagem
        nome_arquivo = os.path.join(caminho_saida, f"{produto_nome}_{contador}.jpg")
        cv2.imwrite(nome_arquivo, frame)
        print(f"Imagem salva: {nome_arquivo}")
        contador += 1

    elif tecla == ord("q"):  # Se pressionar 'q', encerra
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
