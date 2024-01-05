import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matting import get_matted_frames
import shutil
import argparse
from typing import Optional


def get_frame_indices(
    file: Path, n: int, batch_size: int
) -> tuple[list[cv2.VideoCapture], list[list[tuple[int, int]]]]:
    streams = [
        cv2.VideoCapture(str(file.joinpath(name)))
        for name in os.listdir(file)
        if name.endswith(".MOV")
    ]

    frames_per_video = [vid.get(cv2.CAP_PROP_FRAME_COUNT) for vid in streams]
    total_frames = int(sum(frames_per_video))

    video_indices = []
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
        video_indices.append((int(adjusted_index), current_video_index))

    # batching the video indices
    indices = (slice(i, i + batch_size) for i in range(0, n, batch_size))
    video_indices = [video_indices[i] for i in indices]

    return streams, video_indices


def process_video(
    streams: list[cv2.VideoCapture], indices: list[tuple[int, int]]
) -> np.ndarray:
    frames = []
    with tqdm(total=len(indices), desc="Extracting Frames") as pbar:
        for frame_idx, stream_idx in indices:
            stream = streams[stream_idx]
            stream.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = stream.read()
            if ret:
                frames.append(frame)
            else:
                print(f"failed to read frame {frame_idx} in stream {stream_idx}")

            pbar.update(1)

    return frames


def save_frames(frames: list[np.ndarray], output: Path, start_idx: int):
    output.mkdir(exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving Images")):
        cv2.imwrite(str(output / f"{i + start_idx}.png"), frame)


def reset(path: Path, clean: bool = True):
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)
    (path / "distorted").mkdir(exist_ok=True)


def format_command(command: str, args: dict[str, str]) -> str:
    for key, value in args.items():
        arg = ""

        if value is None:
            arg = f" --{key}"
        elif isinstance(value, dict):
            for option, subvalue in value.items():
                if isinstance(subvalue, bool):
                    subvalue = str(subvalue).lower()
                arg += f" --{key}.{option}={subvalue}"
        else:
            arg = f" --{key} {value}"

        command += arg

    return command


def run_command(
    command: str, args: Optional[dict[str, str]] = None, capture_stdout: bool = False
) -> int:
    args = {} if args is None else args
    command = format_command(command, args)
    if capture_stdout:
        with os.popen(f"{command} 2>&1") as stream:
            return stream.read()
    return os.system(command)


def get_model_stats(path: Path) -> dict[str, int]:
    command = f"colmap model_analyzer --path {path}"
    output = run_command(command, capture_stdout=True)
    lines = [line.split("] ")[1].split(": ") for line in output.split("\n") if line]
    stats = {}
    for key, value in lines:
        key = key.lower().replace(" ", "_")
        if "px" in value:
            value = float(value[:-2])
        elif "." in value:
            value = float(value)
        else:
            value = int(value)
        stats[key] = value

    return stats


def reorganize_models(
    path: Path, key: str = "registered_images"
) -> list[tuple[Path, float]]:
    """
    Reorganizes the models in the path based on the number of registered images
    Model ordering can be changed by passing in new stats key
    See output of get_model_stats for available keys

    model subdirs must be have integer names (standard colmap output)
    """
    models = [model for model in path.iterdir() if model.name.isnumeric()]
    stats = sorted(
        [[model, get_model_stats(model)[key]] for model in models],
        key=lambda x: x[1],
        reverse=True,
    )
    for i, (file, _) in enumerate(stats):
        newfile = file.with_name(f"temp{file.name}")
        shutil.move(file, newfile)
        stats[i][0] = newfile
    for i, (file, _) in enumerate(stats):
        newfile = file.with_name(f"{i}")
        shutil.move(file, newfile)
        stats[i][0] = newfile

    return [(file, val) for file, val in stats]


def merge_models(path: Path) -> Path:
    """
    Merges models in the path into a new subdir "merged"

    model subdirs must be have integer names (standard colmap output)

    Returns the merged model output path
    """
    models = reorganize_models(path)
    merged = path / "merged"
    temp = path / "temp"
    merged.mkdir(exist_ok=True)
    temp.mkdir(exist_ok=True)

    # Overwrite the existing merged model
    shutil.rmtree(merged, ignore_errors=True)
    shutil.copytree(models[0][0], merged, dirs_exist_ok=True)

    command = "colmap model_merger"
    args = {"input_path1": merged, "input_path2": merged, "output_path": temp}

    for model, _ in models[1:]:
        # Clean out temp dir (not sure if colmap merge overwrites)
        shutil.rmtree(temp)
        temp.mkdir()

        args["input_path2"] = model
        print(f"Merging {merged} with {model}...")
        formatted = f"{format_command(command, args)}"
        result = run_command(formatted, capture_stdout=True)
        status = result.split("=> Merge ")[-1].split("\n")[0]
        print(result)

        # Ensure that the merge was successful before copying back into "merged"
        if status == "succeeded":
            print(f"Saving Successful Merge")
            shutil.rmtree(merged)
            shutil.copytree(temp, merged)
        elif status == "failed":
            print("Merge unsuccessful")
        else:
            print("Merge failed for unknown reasons")

    shutil.rmtree(temp)
    return merged


def bundle_adjust(path: Path) -> None:
    args = {
        "input_path": path,
        "output_path": path,
        "BundleAdjustment": {
            "max_num_iterations": 100,
            "max_linear_solver_iterations": 200,
            "function_tolerance": 0,
            "gradient_tolerance": 0,
            "parameter_tolerance": 0,
            "refine_focal_length": 1,
            "refine_principal_point": 0,
            "refine_extra_params": 1,
            "refine_extrinsics": 1,
        },
    }
    if run_command("colmap bundle_adjuster", args) != 0:
        raise RuntimeError("Bundle adjustment failed")


def extract_features(database: Path, images: Path) -> None:
    args = {
        "database_path": database,
        "image_path": images,
        "ImageReader": {
            "single_camera": False,
            "single_camera_per_folder": False,
            "single_camera_per_image": False,
            "existing_camera_id": -1,
            "default_focal_length_factor": 1.2,
            "camera_model": "SIMPLE_RADIAL",
        },
        "SiftExtraction": {
            "use_gpu": True,
            "gpu_index": -1,
            "estimate_affine_shape": False,
            "upright": False,
            "domain_size_pooling": False,
            "num_threads": -1,
            "max_image_size": 3200,
            "max_num_features": 8192,
            "first_octave": -1,
            "num_octaves": 4,
            "octave_resolution": 3,
            "max_num_orientations": 2,
            "dsp_num_scales": 10,
            "peak_threshold": 0.0066666666666666671,
            "edge_threshold": 10,
            "dsp_min_scale": 0.16666666666666666,
            "dsp_max_scale": 3,
        },
    }
    if run_command("colmap feature_extractor", args) != 0:
        raise RuntimeError("Feature extraction failed")


def exhausive_matching(database: Path):
    args = {
        "database_path": database,
        "SiftMatching": {
            "use_gpu": True,
            "gpu_index": -1,
            "cross_check": True,
            "guided_matching": False,
            "num_threads": -1,
            "max_num_matches": 32768,
            "max_ratio": 0.80000000000000004,
            "max_distance": 0.69999999999999996,
        },
        "TwoViewGeometry": {
            "multiple_models": False,
            "min_num_inliers": 15,
            "compute_relative_pose": False,
            "max_error": 4,
            "confidence": 0.999,
            "max_num_trials": 10000,
            "min_inlier_ratio": 0.25,
        },
        "ExhaustiveMatching": {"block_size": 50},
    }
    if run_command("colmap exhaustive_matcher", args) != 0:
        raise RuntimeError("Feature matching failed")


def incremental_mapping(database: Path, images: Path, output: Path):
    n_images = len(list(images.iterdir()))
    args = {
        "database_path": database,
        "image_path": images,
        "output_path": output,
        "Mapper": {
            "ignore_watermarks": False,
            "multiple_models": True,
            "extract_colors": True,
            "ba_refine_focal_length": True,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": True,
            "fix_existing_images": False,
            "tri_ignore_two_view_tracks": True,
            "min_num_matches": 15,
            "max_num_models": 50,
            "max_model_overlap": 20,
            "min_model_size": 10,
            "init_image_id1": -1,
            "init_image_id2": -1,
            "init_num_trials": 200,
            "num_threads": -1,
            "ba_min_num_residuals_for_multi_threading": 50000,
            "ba_local_num_images": 6,
            "ba_local_max_num_iterations": 25,
            "ba_global_images_freq": max(500, 5 * n_images),
            "ba_global_points_freq": max(250000, 2500 * n_images),
            "ba_global_max_num_iterations": 50,
            "ba_global_max_refinements": 5,
            "ba_local_max_refinements": 2,
            "snapshot_images_freq": 0,
            "init_min_num_inliers": 100,
            "init_max_reg_trials": 2,
            "abs_pose_min_num_inliers": 30,
            "max_reg_trials": 3,
            "tri_max_transitivity": 1,
            "tri_complete_max_transitivity": 5,
            "tri_re_max_trials": 1,
            "min_focal_length_ratio": 0.10000000000000001,
            "max_focal_length_ratio": 10,
            "max_extra_param": 1,
            "ba_local_function_tolerance": 0,
            "ba_global_images_ratio": 1.5000000000000001,
            "ba_global_points_ratio": 1.5000000000000001,
            "ba_global_function_tolerance": 0,
            "ba_global_max_refinement_change": 0.00050000000000000001,
            "ba_local_max_refinement_change": 0.001,
            "init_max_error": 4,
            "init_max_forward_motion": 0.94999999999999996,
            "init_min_tri_angle": 16,
            "abs_pose_max_error": 12,
            "abs_pose_min_inlier_ratio": 0.25,
            "filter_max_reproj_error": 4,
            "filter_min_tri_angle": 1.5,
            "local_ba_min_tri_angle": 6,
            "tri_create_max_angle_error": 2,
            "tri_continue_max_angle_error": 2,
            "tri_merge_max_reproj_error": 4,
            "tri_complete_max_reproj_error": 4,
            "tri_re_max_angle_error": 5,
            "tri_re_min_ratio": 0.20000000000000001,
            "tri_min_angle": 1.5,
        },
    }
    if run_command("colmap mapper", args) != 0:
        raise RuntimeError("Incremental mapping failed")


def undistort(images: Path, distorted: Path, output: Path):
    args = {
        "image_path": images,
        "input_path": distorted,
        "output_path": output,
        "output_type": "COLMAP",
    }
    if run_command("colmap image_undistorter", args) != 0:
        raise RuntimeError("Image undistortion failed")

    # For some reason sparse/* needs to be sparse/0/*
    sparse = output / "sparse"
    temp = output / "temp"
    shutil.move(sparse, temp)
    shutil.move(temp, sparse / "0")

    # Move back alpha matted images into images dir
    shutil.rmtree(output / "images")
    shutil.copytree(images, output / "images")


def main(n: int = 1000, batch_size: int = 100, skip_gen: bool = False):
    DATA = (Path(__file__).parent / "data").resolve()
    VIDEOS = DATA / "vase4"
    OUTPUT = DATA / "out"
    INPUT = OUTPUT / "input"
    DISTORTED = OUTPUT / "distorted"
    DB = OUTPUT / "db.db"

    if not skip_gen:
        # Cleanup
        reset(OUTPUT, clean=True)

        # Create training images
        streams, indices = get_frame_indices(VIDEOS, n, batch_size)
        total_saved = 0
        for i, batch_idx in enumerate(indices):
            print(f"\nBatch {i + 1}: {len(batch_idx)} images...")
            save_frames(
                get_matted_frames(process_video(streams, batch_idx)),
                output=OUTPUT / "input",
                start_idx=total_saved,
            )
            total_saved += len(batch_idx)
        for stream in streams:
            stream.release()

    # Structure from motion
    extract_features(DB, INPUT)
    exhausive_matching(DB)
    incremental_mapping(DB, INPUT, DISTORTED)

    MERGED = merge_models(DISTORTED)
    bundle_adjust(MERGED)

    undistort(INPUT, MERGED, OUTPUT)

    # Gaussian Splatting
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--batch_size", "-b", type=int, default=100)
    parser.add_argument("--skip", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    main(args.n, args.batch_size, args.skip)
