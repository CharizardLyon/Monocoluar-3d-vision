import pandas as pd
import os
import cv2
from pathlib import Path

base_path = Path("") # add your path here
output_path = Path("dataset")
camera_filenames = ["camA.mp4", "camB.mp4", "camC.mp4", "camD.mp4", "camE.mp4"]
train_ratio = 0.8
FRAME_NAME_PATTERN = "session_sample_{idx:05d}_cam{cam_letter}.png"

TRAIN_DIR = output_path / "train"
TEST_DIR = output_path / "test"
LABELS_PATH = output_path / "labels.csv"

def process_session(session_dir: Path,
                    global_labels: pd.DataFrame,
                    seen_uids: set) -> pd.DataFrame:

    session = session_dir.name
    print(f"▶ Processing {session}")

    # Videos Routes
    video_folder = session_dir / "videos-raw"
    video_paths = [video_folder / f"cam{c}.mp4" for c in "ABCDE"]
    if not all(p.exists() for p in video_paths):
        print(f"  ⚠️  Missing camera videos in {video_folder} → skipping")
        return global_labels

    # CSV Route
    csv_folder = session_dir / "pose-3d"
    csv_files = list(csv_folder.glob("*.csv"))
    if len(csv_files) != 1:
        print(f"  ⚠️  Expected 1 CSV in {csv_folder}, found {len(csv_files)} → skipping")
        return global_labels
    csv_file = csv_files[0]

    # Open Videos
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    if not all(cap.isOpened() for cap in caps):
        print(f"  ❌ Could not open all video files in {session} → skipping")
        return global_labels

    # Read CSV
    session_labels = pd.read_csv(csv_file)
    if session_labels.empty:
        print(f"  ⚠️  CSV {csv_file.name} is empty → skipping")
        return global_labels

    num_rows   = len(session_labels)
    frame_cnts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    samples    = min(min(frame_cnts), num_rows)
    split_idx  = int(samples * train_ratio)

    cam_letters = ["A", "B", "C", "D", "E"]

    for i in range(samples):
        uid = f"{session}_sample_{i:05d}"
        if uid in seen_uids:
            print(f"  🔁 Duplicate UID {uid} found → skipping")
            continue

        split = "train" if i < split_idx else "test"
        out_dir = TRAIN_DIR if split == "train" else TEST_DIR

        frame_files = []
        for cam_idx, cap in enumerate(caps):
            cam_letter = cam_letters[cam_idx]
            ok, frame = cap.read()
            if not ok:
                print(f"  ⚠️  Could not read frame {i} from cam{cam_letter} → stopping session")
                samples = i
                break

            fname = FRAME_NAME_PATTERN.format(session=session, idx=i, cam_letter=cam_letter)
            fpath = out_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fpath), frame)
            frame_files.append(str(fpath.relative_to(output_path)))
        else:
            # Only 3D coordinates
            row_data = session_labels.iloc[i].copy()
            xyz_cols = [col for col in row_data.index if col.endswith(("_x", "_y", "_z"))]
            row_filtered = row_data[xyz_cols].copy()

            row_filtered["sample_uid"]  = uid
            row_filtered["session"]     = session
            row_filtered["split"]       = split
            row_filtered["frame_files"] = ";".join(frame_files)

            global_labels = pd.concat([global_labels, row_filtered.to_frame().T], ignore_index=True)
            continue
        break

    for cap in caps:
        cap.release()

    print(f"✅ Processed {samples} frames for session {session} "
          f"({split_idx} train / {samples - split_idx} test)")
    return global_labels

def main():
    output_path.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)
    
    if LABELS_PATH.exists():
        labels_master = pd.read_csv(LABELS_PATH)
    else:
        labels_master = pd.DataFrame()
        
    seen = set(labels_master.get("sample_uid", []))
    
    for session_folder in sorted(base_path.iterdir()):
        if session_folder.is_dir() and session_folder.name.startswith("session"):
            labels_master = process_session(session_folder, labels_master, seen)
            seen = set(labels_master["sample_uid"])
            
    labels_master.to_csv(LABELS_PATH, index=False)
    print(f"\n labels.csv with {len(labels_master)} total samples saved → "
          f"{LABELS_PATH.resolve()}")
    
if __name__ == "__main__":
    main()
    print("✅ Processing complete.")
            