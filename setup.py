if __name__ == "__main__":
    print("Performing setup...")

    import gdown
    import torch

    WEIGHTS = 'https://drive.google.com/uc?export=download&id=1x_2boVWkN_fgoY949PqDEciZXNB0brYz'
    FILENAME = "matting/weights.pth"
    gdown.download(WEIGHTS, FILENAME)

    print(f"CUDA DETECTED: {torch.cuda.is_available()}")

    print("Setup Complete.")