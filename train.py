# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from models.mlp_model import MLPModel
from utils.preprocess import load_data


def train_and_evaluate(model_name, data_path, site_id):
    # 加载并预处理数据
    scaler, X, y = load_data(data_path, site_id=site_id)

    # 转换为 Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X_tensor.shape[1]
    hidden_dim = 64
    output_dim = 1

    # 初始化模型
    model_name = model_name.lower()
    if model_name == 'lstm':
        model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif model_name == 'gru':
        model = GRUModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif model_name == 'mlp':
        model = MLPModel(input_dim=input_dim)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # 保存模型参数
    torch.save(model.state_dict(), f"{model_name}_model.pth")
    print(f"Training complete and {model_name} model saved.")

    return model, scaler
