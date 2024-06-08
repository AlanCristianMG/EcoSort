import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.simple_cnn import SimpleCNN  # Importar el modelo que definiste
from PIL import Image, UnidentifiedImageError
import os

# Verificar si hay GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paso 1: Definir transformaciones
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionar las imágenes a 128x128
    transforms.ToTensor(),  # Convertir las imágenes a tensores de PyTorch
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizar las imágenes
])

# Paso 2: Cargar los datos
class VerifiedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (IOError, UnidentifiedImageError):
            print(f"Error al cargar la imagen {path}. Será omitida.")
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# ImageFolder es una clase dentro del módulo datasets (perteneciente a torchvision)
# que se utiliza para cargar directorios donde las subcarpetas son las clases
train_data = VerifiedImageFolder(root='dataset/train', transform=transform)
test_data = VerifiedImageFolder(root='dataset/test', transform=transform)

# DataLoader itera sobre el conjunto de datos de entrenamiento. Entrega datos en lotes (batches)
# El tamaño del batch (batch_size) indica el número de imágenes que entregará por grupo durante el entrenamiento
# esto permite que entrenar el modelo en partes más pequeñas y aprovechar mejor la memoria de la GPU o CPU
# El barajeo o shuffle previene que el modelo aprenda patrones específicos mediante la aleatoriedad de las imágenes

train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=60, shuffle=False)

# Paso 3: Definir el modelo
model = SimpleCNN().to(device)

# Paso 4: Definir la función de pérdida y el optimizador
# Define la función de pérdida
# Específicamente nn.CrossEntropyLoss se utiliza para problemas de clasificación multiclase
# calcula la pérdida de entropía cruzada entre la salida del modelo (logits) y las etiquetas reales (verdad fundamental)
# Una pérdida más baja indica mejor rendimiento del modelo, ya que las predicciones están más cerca de las etiquetas verdaderas 
criterion = nn.CrossEntropyLoss()

# Define un optimizador
# optim.Adam se refiere a un algoritmo de optimización llamado Adam (Adaptive Moment Estimation)
# es una opción popular debido a su eficiencia y eficacia en el entrenamiento de modelos de aprendizaje profundo
# model.parameters() recupera los parámetros entrenables (pesos y sesgos) del modelo
# lr=0.001 esto establece la tasa de aprendizaje (lr) en 0.001.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Paso 5: Entrenar el modelo
num_epochs = 2

for epoch in range(num_epochs):
    running_loss = 0.0  # Inicializa la variable para acumular la pérdida durante cada época
    for inputs, labels in train_loader:  # itera sobre el data loader, usando las etiquetas correspondientes
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Poner a cero los gradientes del optimizador
        outputs = model(inputs)  # Hacer forward pass
        loss = criterion(outputs, labels)  # Calcular la pérdida
        loss.backward()  # Hacer backward pass
        optimizer.step()  # Actualizar los parámetros

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print('Finished Training')

# Paso 6: Evaluar el modelo
correct = 0  # se utiliza para contar el número de predicciones correctas con el conjunto de prueba
total = 0  # se usa para contar el número total de ejemplos en el conjunto de prueba

# Esta deshabilitación es importante ya que no necesitamos calcular ni almacenar gradientes
with torch.no_grad():  # Deshabilitar el cálculo de gradientes para la evaluación
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # se usa '_' para ignorar el primer valor devuelto (los valores máximos en sí) ya que no son necesarios
        # torch.max devuelve el valor máximo y su índice a lo largo de la dimensión especificada
        # en este caso '1' que representa las clases
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)

        # Compara las predicciones con las etiquetas reales (labels). Esto produce un tensor booleano donde los elementos
        # son True o False si la predicción es correcta o falsa.
        # sum() cuenta el número de predicciones correctas en el lote
        # item() convierte este conteo en un número escalar que se añade a la variable 'correct'
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')

# Paso 7: Guardar el modelo entrenado
# torch.save se utiliza para guardar el estado del modelo en un archivo
# state_dict() es un diccionario que contiene todos los parámetros del modelo (como tensores)
# y sus valores actuales. Esto incluye pesos y sesgos de todas las capas del modelo
torch.save(model.state_dict(), 'simple_cnn.pth')
