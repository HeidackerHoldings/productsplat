def setup_vit():
    from transformers import VitMatteForImageMatting
    VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-distinctions-646")

def setup_rembg():
    from rembg import new_session
    new_session()

def check_cuda():
    import torch
    print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")

if __name__ == "__main__":
    print("Performing setup...")

    """
    import gdown
    import torch

    WEIGHTS = 'https://drive.google.com/uc?export=download&id=1x_2boVWkN_fgoY949PqDEciZXNB0brYz'
    FILENAME = "matting/weights.pth"
    gdown.download(WEIGHTS, FILENAME)
    """
    
    setup_vit()
    setup_rembg()
    check_cuda()

    print("Setup Complete.")
