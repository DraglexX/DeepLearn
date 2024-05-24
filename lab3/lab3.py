import random
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder


class Perceptron(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, activation='relu'):
        super(Perceptron, self).__init__()
        self.activation = activation
        self.train_time = 0
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_sizes[0])])
        for k in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]))
        self.output = nn.Linear(hidden_sizes[-1], out_size)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x)) if self.activation == 'relu' else torch.sigmoid(layer(x))
        x = self.output(x)
        return x

    def train_model(self, inputs, labels, num_epochs=100):
        start_time = time()
        for epoch in range(num_epochs):
            self.train()
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.lossFn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Эпоха [{epoch + 1}/{num_epochs}], Потери ({self.activation}): {loss.item():.4f}')
                self.train_time = time() - start_time
                print(f"Время обучения с активацией {self.activation}: {self.train_time:.2f} секунд")


# Подготовка и загрузка данных
df = pd.read_csv('data.csv')
# Преобразование меток и подготовка входных данных
labels = LabelEncoder().fit_transform(df.iloc[:, -1])
inputs = df.drop(df.columns[-1], axis=1).values

# Нормализация данных
inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)

inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Разделение данных на обучающую и тестовую выборки
input_size = inputs.shape[1]
output_size = len(np.unique(labels))
hidden_sizes = [random.randint(10, 50) for _ in range(random.randint(1, 3))]

# Создание и обучение моделей
model_relu = Perceptron(input_size, hidden_sizes, output_size, 'relu')
model_sigmoid = Perceptron(input_size, hidden_sizes, output_size, 'sigmoid')

model_relu.train_model(inputs, labels)
model_sigmoid.train_model(inputs, labels)


# Эпоха [10/100], Потери (relu): 1.0576
# Время обучения с активацией relu: 0.01 секунд
# Эпоха [20/100], Потери (relu): 1.0247
# Время обучения с активацией relu: 0.02 секунд
# Эпоха [30/100], Потери (relu): 0.9927
# Время обучения с активацией relu: 0.03 секунд
# Эпоха [40/100], Потери (relu): 0.9609
# Время обучения с активацией relu: 0.03 секунд
# Эпоха [50/100], Потери (relu): 0.9291
# Время обучения с активацией relu: 0.04 секунд
# Эпоха [60/100], Потери (relu): 0.8971
# Время обучения с активацией relu: 0.05 секунд
# Эпоха [70/100], Потери (relu): 0.8649
# Время обучения с активацией relu: 0.06 секунд
# Эпоха [80/100], Потери (relu): 0.8326
# Время обучения с активацией relu: 0.06 секунд
# Эпоха [90/100], Потери (relu): 0.8004
# Время обучения с активацией relu: 0.07 секунд
# Эпоха [100/100], Потери (relu): 0.7686

# Время обучения с активацией relu: 0.08 секунд
# Эпоха [10/100], Потери (sigmoid): 1.1053
# Время обучения с активацией sigmoid: 0.01 секунд
# Эпоха [20/100], Потери (sigmoid): 1.1033
# Время обучения с активацией sigmoid: 0.02 секунд
# Эпоха [30/100], Потери (sigmoid): 1.1022
# Время обучения с активацией sigmoid: 0.02 секунд
# Эпоха [40/100], Потери (sigmoid): 1.1015
# Время обучения с активацией sigmoid: 0.03 секунд
# Эпоха [50/100], Потери (sigmoid): 1.1010
# Время обучения с активацией sigmoid: 0.04 секунд
# Эпоха [60/100], Потери (sigmoid): 1.1005
# Время обучения с активацией sigmoid: 0.05 секунд
# Эпоха [70/100], Потери (sigmoid): 1.1001
# Время обучения с активацией sigmoid: 0.05 секунд
# Эпоха [80/100], Потери (sigmoid): 1.0996
# Время обучения с активацией sigmoid: 0.06 секунд
# Эпоха [90/100], Потери (sigmoid): 1.0992
# Время обучения с активацией sigmoid: 0.07 секунд
# Эпоха [100/100], Потери (sigmoid): 1.0988
# Время обучения с активацией sigmoid: 0.08 секунд

