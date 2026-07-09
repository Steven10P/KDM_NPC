"""One-off script: download MNIST and GTSRB via torchvision into data/."""
import torchvision

DATA_DIR = r"C:\Users\bspd2\Maestria\Tesis_KDM_NPC\data"

print("Downloading MNIST...")
torchvision.datasets.MNIST(root=f"{DATA_DIR}/mnist", train=True, download=True)
torchvision.datasets.MNIST(root=f"{DATA_DIR}/mnist", train=False, download=True)
print("MNIST done.")

print("Downloading GTSRB (train)...")
torchvision.datasets.GTSRB(root=f"{DATA_DIR}/gtsrb", split="train", download=True)
print("Downloading GTSRB (test)...")
torchvision.datasets.GTSRB(root=f"{DATA_DIR}/gtsrb", split="test", download=True)
print("GTSRB done.")
