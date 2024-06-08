import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module): 
    def __init__(self): # Definicion
        super(SimpleCNN, self).__init__()
        
        # Primera capa convolucional
        # in_channels: número de canales de entrada (3 para imágenes RGB)
        # out_channels: número de filtros de salida (16 en este caso)
        # kernel_size: tamaño del filtro de convolución (3x3)
        # stride: paso del filtro durante la convolución (1 en este caso)
        # padding: agregar ceros alrededor de la imagen de entrada (1 pixel en cada lado)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Capa de pooling máximo
        # kernel_size: tamaño de la ventana de pooling (2x2)
        # stride: paso del pooling (2 en este caso)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Capa lineal (totalmente conectada)
        # 32 * 32 * 32: número de neuronas de entrada (calculado a partir del tamaño de salida de la capa convolucional anterior)
        # 128: número de neuronas en la capa totalmente conectada
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        
        # Capa de salida
        # 128: número de neuronas de entrada (salida de la capa anterior)
        # 3: número de neuronas de salida (tres categorías: perro, gato, caballo)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x): # FLujo de los datos de entrada
        # Aplicar la primera capa convolucional, seguida de ReLU y pooling máximo
        x = self.pool(F.relu(self.conv1(x)))
        # Aplicar la segunda capa convolucional, seguida de ReLU y pooling máximo
        x = self.pool(F.relu(self.conv2(x)))
        # Aplanar el tensor para pasar a las capas totalmente conectadas
        x = x.view(-1, 32 * 32 * 32)
        # Aplicar la primera capa totalmente conectada con ReLU
        x = F.relu(self.fc1(x))
        # Aplicar la capa de salida
        x = self.fc2(x)
        return x
    