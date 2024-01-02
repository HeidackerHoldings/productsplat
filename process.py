import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matting import get_matted_frames
import shutil
import argparse


def process_video(file: Path, n: int) -> np.ndarray:
    streams = [
        cv2.VideoCapture(str(file.joinpath(name)))
        for name in os.listdir(file)
        if name.endswith(".MOV")
    ]

    frames_per_video = [vid.get(cv2.CAP_PROP_FRAME_COUNT) for vid in streams]
    total_frames = int(sum(frames_per_video))

    video_indices = [[] for _ in range(len(streams))]
    current_frames = 0
    current_video_index = 0
    spread = total_frames / n
    initial_index = [int(spread * i) for i in range(n)]
    for index in initial_index:
        adjusted_index = index - current_frames

        while adjusted_index > frames_per_video[current_video_index]:
            current_frames += frames_per_video[current_video_index]
            current_video_index += 1
            adjusted_index = index - current_frames
        video_indices[current_video_index].append(int(adjusted_index))

    frames = []
    with tqdm(total=n, desc="Extracting Frames") as pbar:
        for video, index in zip(streams, video_indices):
            for i in index:
                video.set(cv2.CAP_PROP_POS_FRAMES, i)

                ret, frame = video.read()
                if ret:
                    frames.append(frame)

                pbar.update(1)

            video.release()

    return frames


def save_frames(frames: list[np.ndarray], output: Path):
    output.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(output / f"{i}.png"), frame)    


def reset(path: Path, clean: bool =True):
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)


def run_command(command: str, args: dict[str, str]) -> int:
    for key, value in args:
        if value is None:
            command += f" {key}"
        else:
            command += f" {key} {value}"
    return os.system(command)


def extract_features(database: Path, images: Path) -> None:
    args = {
        "--database_path": database,
        "--image_path": images,
        "--SiftExtraction.use_gpu": "1"
    }
    if run_command("colmap feature_extractor", args) != 0:
        raise RuntimeError("Feature extraction failed")


def exhausive_matching(database: Path):
    args = {
        "--database_path": database,
        "--SiftMatching.use_gpu": "1",
    }
    if run_command("colmap exhaustive_matcher", args) != 0:
        raise RuntimeError("Feature matching failed")


def incremental_mapping(database: Path, images: Path, output: Path):
    args = {
        "--database_path": database,
        "--image_path": images,
        "--output_path": output,
        "--Mapper.ba_global_function_tolerance=0.000001": None
    }
    if run_command("colmap mapper", args) != 0:
        raise RuntimeError("Incremental mapping failed")


def undistort(images: Path, output: Path):
    return
    args = {
        "--image_path": images,
        "--input_path": None
    }

    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + args.source_path + "/input \
        --input_path " + args.source_path + "/distorted/sparse/0 \
        --output_path " + args.source_path + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)


def main():

    DATA = (Path(__file__).parent / "data").resolve()
    INPUT = DATA / "vase4"
    OUTPUT = DATA / "out"

    # Cleanup
    reset(OUTPUT, clean=True)

    # Create training images
    frames = process_video(INPUT, 5)
    frames = get_matted_frames(frames)
    save_frames(frames, OUTPUT / "input")

    # Structure from motion
    DB = OUTPUT / "db.db"
    INPUT = OUTPUT / "input"
    extract_features(DB, INPUT)
    exhausive_matching(DB)
    incremental_mapping(DB, INPUT, OUTPUT)

    # Gaussian Splatting
    ...



if __name__ == "__main__":
    main()
