import os
from PIL import Image

def preprocess_image(input_path, output_path, size=(256, 256)):
    image = Image.open(input_path).convert("RGB")
    image = image.resize(size)
    image.save(output_path)

def preprocess_directory(input_dir, output_dir, size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            preprocess_image(input_path, output_path, size)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preprocess_directory(r"C:\Users\Arlene Cristina\Downloads\Projeto 3 IA\pytorch-CycleGAN-and-pix2pix\data\raw\Healthy_Train50", 
                         r"C:\Users\Arlene Cristina\Downloads\Projeto 3 IA\pytorch-CycleGAN-and-pix2pix\data\preprocessed\Healthy_Train50")
    preprocess_directory(r"C:\Users\Arlene Cristina\Downloads\Projeto 3 IA\pytorch-CycleGAN-and-pix2pix\data\raw\Healthy_Test50", 
                         r"C:\Users\Arlene Cristina\Downloads\Projeto 3 IA\pytorch-CycleGAN-and-pix2pix\data\preprocessed\Healthy_Test50")
    preprocess_directory(r"C:\Users\Arlene Cristina\Downloads\Projeto 3 IA\pytorch-CycleGAN-and-pix2pix\data\raw\Disease_Test100", 
                         r"C:\Users\Arlene Cristina\Downloads\Projeto 3 IA\pytorch-CycleGAN-and-pix2pix\data\preprocessed\Disease_Test100")
