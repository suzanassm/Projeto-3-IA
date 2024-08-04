import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

from train import PlantDataset, SimpleAutoencoder

def load_model(model_path):
    model = SimpleAutoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def calculate_anomaly_score(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def visualize_anomalies(model, dataloader):
    for data in dataloader:
        inputs = data.to(device)
        outputs = model(inputs)
        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().detach().numpy()

        for i in range(len(inputs)):
            original = inputs[i].transpose(1, 2, 0)
            reconstructed = outputs[i].transpose(1, 2, 0)
            anomaly_score = calculate_anomaly_score(original, reconstructed)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(original)
            plt.subplot(1, 2, 2)
            plt.title(f"Reconstructed (Anomaly Score: {anomaly_score:.4f})")
            plt.imshow(reconstructed)
            plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset_healthy = PlantDataset("../data/preprocessed/Healthy_Test50", transform=transform)
    dataset_disease = PlantDataset("../data/preprocessed/Disease_Test100", transform=transform)
    dataloader_healthy = DataLoader(dataset_healthy, batch_size=1, shuffle=False)
    dataloader_disease = DataLoader(dataset_disease, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("../models/anomaly_detection_model.pth").to(device)

    print("Visualizing anomalies in healthy test set...")
    visualize_anomalies(model, dataloader_healthy)

    print("Visualizing anomalies in disease test set...")
    visualize_anomalies(model, dataloader_disease)
