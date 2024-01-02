from rembg import remove as rmbg, new_session
from PIL import Image, ImageOps
import numpy as np
import cv2
from transformers import VitMatteForImageMatting
import torch
from tqdm import tqdm


def get_masks(images: np.ndarray) -> list[np.ndarray]:
    session = new_session()
    return [
        rmbg(image, session=session, only_mask=True)
        for image in tqdm(images, desc="Creating Trimaps")
    ]


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


def pad_image(image: np.ndarray, val: int = 32) -> np.ndarray:
    image = Image.fromarray(image)
    width, height = image.size
    width = ((width // val) + 1) * val
    height = ((height // val) + 1) * val
    return np.array(ImageOps.pad(image, (width, height), centering=(0, 0)))


def vitmatte(
    images: list[np.ndarray],
    trimaps: list[np.ndarray],
    lowmem: bool = False,
    scale: int = 2,
) -> list[np.ndarray]:
    model = VitMatteForImageMatting.from_pretrained(
        "hustvl/vitmatte-small-distinctions-646"
    ).to("cuda")

    matted, alphas = [], []
    for image, trimap in tqdm(
        zip(images, trimaps), total=len(images), desc="Performing ViTMatte"
    ):
        image = pad_image(image, 32 * (scale if lowmem else 1))
        trimap = pad_image(np.uint8(trimap), 32 * (scale if lowmem else 1))

        original_image = image.copy()

        if lowmem:
            image = np.array(
                Image.fromarray(image).resize(
                    (image.shape[1] // scale, image.shape[0] // scale)
                )
            )
            trimap = np.array(
                Image.fromarray(trimap).resize(
                    (trimap.shape[1] // scale, trimap.shape[0] // scale)
                )
            )

        pixels = torch.concatenate(
            [
                torch.tensor(image, device="cuda"),
                torch.tensor(trimap, device="cuda")[..., None],
            ],
            axis=2,
        )
        pixels = pixels.permute(2, 0, 1)[None, ...].float() / 255

        with torch.no_grad():
            alpha = model(pixels).alphas

        alpha = alpha.squeeze(0).permute(1, 2, 0).to("cpu").numpy()
        alpha = np.uint8(alpha * 255)

        if lowmem:
            alpha = np.array(
                Image.fromarray(alpha[..., 0]).resize((alpha.shape[1] * scale, alpha.shape[0] * scale))
            )[..., None]

        image = np.concatenate([original_image, alpha], axis=2)

        matted.append(image)
        alphas.append(alpha)

    return matted, alphas


def get_matted_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    # Get trimaps
    masks = get_masks(frames)
    trimaps = get_trimaps(masks)
    crops = get_crops(trimaps)

    # Remove unnecessary portions from images (reduce VRAM usage)
    frames = [apply_crop(frame, crop) for frame, crop in zip(frames, crops)]
    trimaps = [apply_crop(trimap, crop) for trimap, crop in zip(trimaps, crops)]

    # Alpha Matting
    matted, alphas = vitmatte(frames, trimaps, lowmem=True, scale=3)

    # Cropping round 2
    masks = [alpha > 0 for alpha in alphas]
    crops = get_crops(masks)
    matted = [apply_crop(image, crop) for image, crop in zip(matted, crops)]

    # Padding to same size
    max_width = max(image.shape[1] for image in matted)
    max_height = max(image.shape[0] for image in matted)
    matted = [
        np.array(
            ImageOps.pad(
                Image.fromarray(image), (max_width, max_height), centering=(0, 0)
            )
        )
        for image in matted
    ]

    return matted
