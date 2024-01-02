import gdown
from pathlib import Path
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str)
    args = parser.parse_args()

    VALID = {
        "applewatch": "https://drive.google.com/drive/folders/1NSKhZGKkKIn7K-O9Vfp9syY2J6LbJqST?usp=sharing",
        "chawan": "https://drive.google.com/drive/folders/170brEZlT1wke2TQ6aNHwqZSJU1LdMHmA?usp=sharing",
        "metagross": "https://drive.google.com/drive/folders/1fLJ9giT65sV1Kl-koHUqFMrZvhcf_Xju?usp=sharing",
        "ring": "https://drive.google.com/drive/folders/1HQhMfYB8nROwkCMZ9uMEf8Zo92pv8Ykr?usp=sharing",
        "vase1": "https://drive.google.com/drive/folders/1R36kvv8Lr68Q7UvhKuVOPJQ3YKARKNgC?usp=sharing",
        "vase2": "https://drive.google.com/drive/folders/1saJPN6yZnRBXj4lRjvVpY6Y-RTecJ-kV?usp=sharing",
        "vase3": "https://drive.google.com/drive/folders/1g6yEjRctOavZnLHB8QLh91_-4N-Rmq1l?usp=sharing",
        "vase4": "https://drive.google.com/drive/folders/1uYcO769TMZ_d4Y-pV2X_SW-1HvQklCvy?usp=sharing"
    }

    folder_name = None
    if args.name is None:
        print("Folder name not specified. Downloading Vase4")
        folder_name = "vase4"
    elif args.name not in VALID:
        print(f"Invalid folder name {args.name}")
        sys.exit()
    else:
        print(f"Downloading {args.name}")
        folder_name = args.name
        

    DATA = (Path(__file__).parent / f"data").resolve()
    DATA.mkdir(exist_ok=True)
    OUTPUT = DATA / folder_name
    OUTPUT.mkdir(exist_ok=True)

    gdown.download_folder(url=VALID[folder_name], output=str(OUTPUT))