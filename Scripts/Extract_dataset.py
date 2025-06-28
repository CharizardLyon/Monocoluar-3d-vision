import pandas as pd
import os
import cv2
from pathlib import Path
import numpy as np

base_path = Path("")  # Add your sessions root path here
output_path = Path("data")
camera_filenames = ["camA.mp4", "camB.mp4", "camC.mp4", "camD.mp4", "camE.mp4"]
train_ratio = 0.8
FRAME_FOLDER_PATTERN = "frame_{idx:06d}"
FRAME_IMAGE_PATTERN = "cam{cam_num}.jpg"  # Using cam1, cam2, ... for naming

TRAIN_DIR = output_path / "train"
VAL_DIR = output_path / "val"

def process_session(session_dir: Path, train_ratio: float):
    print(f"▶ Processing session {session_dir.name}")

    video_folder = session_dir / "videos-raw"
    csv_folder = session_dir / "pose-3d"

    video_paths = [video_folder / f"cam{c}.mp4" for c in "ABCDE"]
    if not all(p.exists() for p in video_paths):
        print(f"  ⚠️  Missing camera videos in {video_folder} → skipping session")
        return [], []

    csv_files = list(csv_folder.glob("*.csv"))
    if len(csv_files) != 1:
        print(f"  ⚠️  Expected exactly 1 CSV in {csv_folder}, found {len(csv_files)} → skipping session")
        return [], []

    csv_file = csv_files[0]
    session_labels = pd.read_csv(csv_file)
    if session_labels.empty:
        print(f"  ⚠️  CSV file {csv_file.name} is empty → skipping session")
        return [], []

    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    if not all(cap.isOpened() for cap in caps):
        print(f"  ❌ Could not open all videos in session {session_dir.name} → skipping session")
        for cap in caps:
            cap.release()
        return [], []

    num_frames = len(session_labels)
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    samples = min(min(frame_counts), num_frames)
    split_idx = int(samples * train_ratio)

    train_samples = []
    val_samples = []

    for i in range(samples):
        # Determine train or val split
        split = "train" if i < split_idx else "val"
        out_dir = TRAIN_DIR if split == "train" else VAL_DIR

        frame_folder_name = FRAME_FOLDER_PATTERN.format(idx=i)
        frame_folder = out_dir / "frames" / frame_folder_name
        frame_folder.mkdir(parents=True, exist_ok=True)

        frame_images = []
        for cam_idx, cap in enumerate(caps):
            ok, frame = cap.read()
            if not ok:
                print(f"  ⚠️  Could not read frame {i} from cam{cam_idx+1} → stopping session early")
                samples = i
                break

            img_name = FRAME_IMAGE_PATTERN.format(cam_num=cam_idx+1)
            img_path = frame_folder / img_name
            cv2.imwrite(str(img_path), frame)
            frame_images.append(str(img_path.relative_to(output_path)))

        else:
            # Extract 3D joint coordinates columns from CSV (all cols ending with _x, _y, _z)
            row = session_labels.iloc[i]
            xyz_cols = [col for col in row.index if col.endswith(('_x', '_y', '_z'))]
            coords = row[xyz_cols].to_numpy(dtype=np.float32)

            # Reshape into (21, 3) assuming 21 joints * 3 coords (x,y,z)
            joints3d = coords.reshape(-1, 3)

            # Save joints3d as .npy
            joints_dir = out_dir / "joints3d"
            joints_dir.mkdir(parents=True, exist_ok=True)
            joints_path = joints_dir / f"{frame_folder_name}.npy"
            np.save(joints_path, joints3d)

            # Store sample info for optional CSV log (can be omitted)
            sample_info = {
                "session": session_dir.name,
                "frame_folder": str(frame_folder.relative_to(output_path)),
                "split": split,
                "frame_files": ";".join(frame_images),
                "joints_path": str(joints_path.relative_to(output_path))
            }
            if split == "train":
                train_samples.append(sample_info)
            else:
                val_samples.append(sample_info)

            continue
        break

    for cap in caps:
        cap.release()

    print(f"✅ Processed {samples} frames for session {session_dir.name} ({split_idx} train / {samples - split_idx} val)")
    return train_samples, val_samples

def main():
    output_path.mkdir(exist_ok=True)
    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)

    all_train_samples = []
    all_val_samples = []

    for session_folder in sorted(base_path.iterdir()):
        if session_folder.is_dir() and session_folder.name.startswith("session"):
            train_samples, val_samples = process_session(session_folder, train_ratio)
            all_train_samples.extend(train_samples)
            all_val_samples.extend(val_samples)

    # Optional: Save a master CSV log for reference
    log_df = pd.DataFrame(all_train_samples + all_val_samples)
    log_path = output_path / "dataset_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\nDataset log saved to {log_path.resolve()}")

if __name__ == "__main__":
    main()
    print("✅ Dataset processing complete.")
