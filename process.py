import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

remove_bg = pipeline(Tasks.universal_matting, model="damo/cv_unet_universal-matting")


def postprocess_frame(frame: np.ndarray) -> np.ndarray:
    return remove_bg(cv2.flip(frame, 1))[OutputKeys.OUTPUT_IMG]


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
    with tqdm(total=n) as pbar:
        for video, index in zip(streams, video_indices):
            for i in index:
                video.set(cv2.CAP_PROP_POS_FRAMES, i)

                ret, frame = video.read()
                if ret:
                    frame = postprocess_frame(frame)
                    frames.append(frame)

                pbar.update(1)

            video.release()

    return frames


if __name__ == "__main__":
    HERE = Path(__file__).parent
    DATA = HERE.joinpath("data/chawan")
    frames = process_video(DATA, 200)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"out/{i}.png", frame)
