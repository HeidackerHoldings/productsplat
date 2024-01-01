# common functions
from PIL import Image
from pathlib import Path
from torchvision.transforms import functional as F
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import cv2
from rembg import remove as rmbg, new_session
import numpy as np


def get_masks(images: Image.Image) -> list[Image.Image]:
    session =  new_session()
    return [rmbg(image, session=session, only_mask=True) for image in images]


# Generate Trimap
def erode_and_dilate(mask: np.ndarray, k_size: tuple[int, int], iterations: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)

    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    trimap = np.full(mask.shape, 128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0

    return trimap


def generate_trimap(mask: np.ndarray, threshold=0.05, iterations=3) -> np.ndarray:
    threshold = threshold * 255

    trimap = mask.copy()
    trimap = trimap.astype("uint8")

    # Erode and dilate the mask
    trimap = erode_and_dilate(trimap, k_size=(7, 7), iterations=iterations)

    return trimap

def get_trimaps(images: list[Image.Image]) -> list[Image.Image]:
    return [generate_trimap(np.array(mask)) for mask in get_masks(images)]

def infer_one_image(data: dict, model) -> Image.Image:
    """
    Infer the alpha matte of one image.
    Input:
        model: the trained model
        image: the input image
        trimap: the input trimap
    """
    output = model(data)["phas"].flatten(0, 2)
    return F.to_pil_image(output)


def init_model():
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    HERE = Path(__file__).parent.resolve()
    config = str(HERE / "vitconfig.py")
    weights = str(HERE / "weights.pth")
    cfg = LazyConfig.load(config)
    model = instantiate(cfg.model)
    model.to("cuda")
    model.eval()
    DetectionCheckpointer(model).load(weights)
    return model


def get_data(image: Image.Image, trimap: Image.Image):
    """
    Get the data of one image.
    Input:
        image_dir: the directory of the image
        trimap_dir: the directory of the trimap
    """
    image = image.convert("RGB")
    image = F.to_tensor(image).unsqueeze(0)
    trimap = trimap.convert("L")
    trimap = F.to_tensor(trimap).unsqueeze(0)

    return {"image": image, "trimap": trimap}


def cal_foreground(image_dir, alpha_dir):
    """
    Calculate the foreground of the image.
    Input:
        image_dir: the directory of the image
        alpha_dir: the directory of the alpha matte
    Output:
        foreground: the foreground of the image, numpy array
    """
    image = Image.open(image_dir).convert("RGB")
    alpha = Image.open(alpha_dir).convert("L")
    alpha = F.to_tensor(alpha).unsqueeze(0)
    image = F.to_tensor(image).unsqueeze(0)
    foreground = image * alpha + (1 - alpha)
    foreground = foreground.squeeze(0).permute(1, 2, 0).numpy()

    return foreground


def make_image_small(image: Image.Image) -> Image.Image:
    return image.resize((dim // 5 for dim in image.size))

def remove_background(
    images: list[Image.Image], save_images: bool = True, return_images: bool = False
):
    trimaps = [make_image_small(Image.open("data/out/trimaps/3.png"))]
    model = init_model()
    results = []
    for i, (image, trimap) in enumerate(zip(images, trimaps)):
        clean = infer_one_image(get_data(image, trimap), model)
        if return_images:
            results.append(clean)
        if save_images:
            clean.save(f"data/out/matting/{i}.png")

    if return_images:
        return results

if __name__ == "__main__":
    image = Image.open("data/out/input/3.png")
    remove_background([make_image_small(image)])
