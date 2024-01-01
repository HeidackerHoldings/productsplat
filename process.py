import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matting import get_matted_frames
import shutil
import argparse

DATA = (Path(__file__).parent / "data").resolve()
INPUT = DATA / "vase4"
OUTPUT = DATA / "out"


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


def save_frames(frames: list[np.ndarray], output: Path = OUTPUT / "input"):
    output.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(output / f"{i}.png"), frame)    


def colmap_pipeline(file: Path):
    ...


def reset(clean: bool =True):
    if clean and OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(exist_ok=True)

def main():
    reset(clean=True)
    frames = process_video(INPUT, 20)
    frames = get_matted_frames(frames)
    save_frames(frames)



if __name__ == "__main__":
    main()
