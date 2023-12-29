import cv2
import numpy as np
from transformers import pipeline
    
# Generate Trimap
def erode_and_dilate(mask, k_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)

    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    trimap = np.full(mask.shape, 128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0

    return trimap

def generate_trimap(mask, threshold=0.05, iterations=3):
    threshold = threshold * 255

    trimap = mask.copy()
    trimap = trimap.astype("uint8")

    # Erode and dilate the mask
    trimap = erode_and_dilate(trimap, k_size=(7, 7), iterations=iterations)

    return trimap
    

if __name__ == "__main__":
    # If more memory is available, use facebook/sam-vit-base
    generator =  pipeline("mask-generation", "facebook/sam-vit-base", revision="d0d9250", device = 0, points_per_batch = 256)
    image_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    mask = generator(image_url, points_per_batch = 256)["masks"][0]
    cv2.imwrite("./mask.png", mask)
    #cv2.imwrite("./mask.png", mask)
    #trimap = generate_trimap(mask)
    #cv2.imwrite("./trimap.png", trimap)