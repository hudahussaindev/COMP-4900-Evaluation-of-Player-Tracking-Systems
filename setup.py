
"""
Quick Setup
Run this first to check everything is installed
"""

import os
from pathlib import Path
import subprocess
import sys

def setup_directories():
    #Create necessary directories
    print("\nSetting up directories...")
    
    base_dir = Path('./required')
    dirs = [
        base_dir,
        base_dir / 'systems',
        base_dir / 'videos',
        base_dir / 'results',
        base_dir / 'test_clips',
        base_dir / 'scripts'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"{dir_path}")
    
    print("\nDirectories created!")

def setup_tracking_systems():
    "Clone and setup the tracking systems"
    systems_dir = Path(__file__).parent / "required" / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    
    systems = {
        'eagle': 'https://github.com/nreHieW/Eagle.git',
        'darkmyter': 'https://github.com/Darkmyter/Football-Players-Tracking.git',
        'anshchoudhary': 'https://github.com/AnshChoudhary/Football-Tracking.git',
        'tracklab': 'https://github.com/TrackingLaboratory/tracklab.git'
    }
    
    print("\nSetting up tracking systems...")
    for name, repo in systems.items():
        system_path = systems_dir / name
        if not system_path.exists():
            print(f"Cloning {name}...")
            try:
                subprocess.run(['git', 'clone', repo, str(system_path)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone {name}: {e}")
        else:
            print(f"{name} already exists")
    return systems_dir


def install_requirements(system_name, system_path):
    print(f"\nInstalling dependencies for: {system_name}")
    os.chdir(system_path)

    if (system_path / 'requirements.txt').exists():
        print("Found requirements.txt")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=False)

    if (system_path / 'setup.py').exists():
        print("Found setup.py")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=False)

    if system_name == 'eagle':
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'opencv-python', 'numpy', 'scipy', 'torch', 'torchvision'], check=False)

    elif system_name == 'darkmyter':
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics', 'opencv-python', 'numpy'], check=False)
        

    elif system_name == 'anshchoudhary':
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'opencv-python', 'numpy'], check=False)

    elif system_name == 'tracklab':
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'tracklab'], check=False)
    os.chdir(Path(__file__).parent)


def write_darkmyter_bytetrack(system_path):
    bytetrack_code = (
        "import numpy as np\n\n"
        "class Track:\n"
        "    def __init__(self, tlwh, score, track_id):\n"
        "        self.tlwh = np.array(tlwh, dtype=float)\n"
        "        self.score = score\n"
        "        self.track_id = track_id\n\n"
        "class BYTETracker:\n"
        "    def __init__(self, iou_threshold=0.3):\n"
        "        self.iou_threshold = iou_threshold\n"
        "        self.tracks = []\n"
        "        self.next_id = 1\n\n"
        "    def iou(self, boxA, boxB):\n"
        "        xA = max(boxA[0], boxB[0])\n"
        "        yA = max(boxA[1], boxB[1])\n"
        "        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])\n"
        "        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])\n"
        "        inter = max(0, xB - xA) * max(0, yB - yA)\n"
        "        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter\n"
        "        return inter / union if union > 0 else 0.0\n\n"
        "    def update(self, detections):\n"
        "        updated_tracks = []\n"
        "        used = set()\n\n"
        "        # match detections to tracks\n"
        "        for track in self.tracks:\n"
        "            best_iou = 0\n"
        "            best_det = None\n"
        "            best_idx = -1\n"
        "            for i, det in enumerate(detections):\n"
        "                if i in used:\n"
        "                    continue\n"
        "                iou = self.iou(track.tlwh, det[:4])\n"
        "                if iou > best_iou:\n"
        "                    best_iou = iou\n"
        "                    best_det = det\n"
        "                    best_idx = i\n\n"
        "            if best_iou > self.iou_threshold:\n"
        "                track.tlwh = np.array(best_det[:4])\n"
        "                updated_tracks.append(track)\n"
        "                used.add(best_idx)\n\n"
        "        # new tracks for unmatched detections\n"
        "        for i, det in enumerate(detections):\n"
        "            if i not in used:\n"
        "                new_track = Track(det[:4], det[4], self.next_id)\n"
        "                self.next_id += 1\n"
        "                updated_tracks.append(new_track)\n\n"
        "        self.tracks = updated_tracks\n"
        "        return updated_tracks\n"
    )
    with open(system_path / "bytetrack.py", "w") as f:
        f.write(bytetrack_code)
        
def write_darkmyter_inference(system_path):
    code = (
        "from ultralytics import YOLO\n"
        "import argparse, json, cv2\n"
        "from bytetrack import BYTETracker\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--source', required=True)\n"
        "    parser.add_argument('--output', required=True)\n"
        "    args = parser.parse_args()\n\n"
        "    model = YOLO('yolov8n.pt')\n"
        "    tracker = BYTETracker()\n\n"
        "    cap = cv2.VideoCapture(args.source)\n"
        "    results = {}\n"
        "    frame_id = 0\n\n"
        "    while True:\n"
        "        ret, frame = cap.read()\n"
        "        if not ret:\n"
        "            break\n\n"
        "        dets = model(frame)[0].boxes\n"
        "        det_list = []\n"
        "        for d in dets:\n"
        "            x1, y1, x2, y2 = d.xyxy[0].tolist()\n"
        "            det_list.append([x1, y1, x2 - x1, y2 - y1, float(d.conf[0])])\n\n"
        "        tracks = tracker.update(det_list)\n"
        "        frame_results = []\n"
        "        for t in tracks:\n"
        "            x, y, w, h = t.tlwh\n"
        "            frame_results.append({\n"
        "                'id': int(t.track_id),\n"
        "                'x': float(x),\n"
        "                'y': float(y),\n"
        "                'w': float(w),\n"
        "                'h': float(h)\n"
        "            })\n\n"
        "        results[str(frame_id)] = frame_results\n"
        "        frame_id += 1\n\n"
        "    cap.release()\n"
        "    with open(args.output, 'w') as f:\n"
        "        json.dump(results, f)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )    
    with open(system_path / "inference.py", "w") as f:
        f.write(code)


def main():
    base_dir = Path(__file__).parent
    systems_dir = base_dir / 'required' / 'systems'
    
    setup_directories()
    setup_tracking_systems()
    systems = ['eagle', 'darkmyter', 'anshchoudhary', 'tracklab']
    
    for system in systems:
        system_path = systems_dir / system
        if system_path.exists():
            install_requirements(system, system_path)
            if system == 'darkmyter':
                write_darkmyter_bytetrack(system_path)
                write_darkmyter_inference(system_path)

        else:
            print(f" {system} not found at {system_path}")
   

if __name__ == '__main__':
    main()
