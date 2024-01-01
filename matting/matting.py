from rembg import remove as rmbg, new_session
from PIL import Image
import numpy as np
import cv2
from transformers import VitMatteImageProcessor, VitMatteForImageMatting
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm


def get_masks(images: np.ndarray) -> list[np.ndarray]:
    session = new_session()
    return [rmbg(image, session=session, only_mask=True) for image in tqdm(images, desc="Creating Trimaps")]


def get_crops(masks: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    results = []
    for mask in masks:
        mask = np.array(mask)

        # Find the coordinates of non-empty (True) values in the mask
        non_empty_coords = np.argwhere(mask)

        # Find the minimum and maximum coordinates along each axis
        min_coords = np.min(non_empty_coords, axis=0)
        max_coords = np.max(non_empty_coords, axis=0)
        results.append((min_coords, max_coords))

    return results


def erode_and_dilate(
    mask: np.ndarray, k_size: tuple[int, int], iterations: int
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)

    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    trimap = np.full(mask.shape, 128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0

    return trimap


def get_trimaps(
    masks: list[np.ndarray], threshold=0.05, iterations=3
) -> list[np.ndarray]:
    threshold = threshold * 255

    trimaps = []
    for mask in masks:
        trimap = mask.copy()
        trimap = trimap.astype("uint8")

        # Erode and dilate the mask
        trimap = erode_and_dilate(trimap, k_size=(7, 7), iterations=iterations)

        trimaps.append(trimap)

    return trimaps


def apply_crop(image: np.ndarray, crop: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    min_coords, max_coords = crop
    return image[min_coords[0] : max_coords[0] + 1, min_coords[1] : max_coords[1] + 1]


def vitmatte(
    images: list[np.ndarray], trimaps: list[np.ndarray], lowmem: bool = True, scale: int = 2) -> list[np.ndarray]:
    processor = VitMatteImageProcessor.from_pretrained(
        "hustvl/vitmatte-small-distinctions-646"
    )
    model = VitMatteForImageMatting.from_pretrained(
        "hustvl/vitmatte-small-distinctions-646"
    )
    model.to("cuda")

    images = [Image.fromarray(image) for image in images]
    trimaps = [Image.fromarray(np.uint8(trimap)).convert("L") for trimap in trimaps]

    sizes = [image.size for image in images]
    if lowmem:
        images = [image.resize((dim // scale for dim in image.size)) for image in images]
        trimaps = [
            trimap.resize((dim // scale for dim in trimap.size)) for trimap in trimaps
        ]

    preds = []
    for image, trimap, original_size in tqdm(zip(images, trimaps, sizes), desc="Running ViTMatte", total=len(images)):
        pixels = processor(
            images=image, trimaps=trimap, return_tensors="pt"
        ).pixel_values
        with torch.no_grad():
            outputs = model(pixels.to("cuda"))
        alphas = outputs.alphas.flatten(0, 2)
        prediction = F.to_pil_image(alphas)

        if lowmem:
            prediction = prediction.resize(original_size)

        preds.append(np.array(prediction))

    return preds

def apply_mask_predictions(frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    images = []
    for frame, mask in zip(frames, masks):
        images.append(np.concatenate([frame, mask[..., None]], axis=-1))
    return images


def get_matted_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    # Get trimaps
    masks = get_masks(frames)
    trimaps = get_trimaps(masks)
    crops = get_crops(trimaps)

    # Remove unnecessary portions from images (reduce VRAM usage)
    frames = [apply_crop(frame, crop) for frame, crop in zip(frames, crops)]
    trimaps = [apply_crop(trimap, crop) for trimap, crop in zip(trimaps, crops)]

    # Alpha Matting
    preds = vitmatte(frames, trimaps)
    frames = apply_mask_predictions(frames, preds)

    return frames
