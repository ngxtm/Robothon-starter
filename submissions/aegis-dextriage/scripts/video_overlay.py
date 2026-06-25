import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def apply_overlay(video_path: Path, trajectory_path: Path, output_path: Path, fps: int = 10) -> bool:
    try:
        with open(trajectory_path, 'r') as f:
            trajectory = json.load(f)
            
        samples = trajectory.get("trajectory_samples", [])
        if not samples:
            print("No trajectory samples found.")
            return False

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return False
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Find the closest sample for this frame
            time_s = frame_idx / video_fps
            sample = min(samples, key=lambda s: abs(s.get("time_s", 0) - time_s))
            
            # Create a simple overlay text block
            y0, dy = 30, 20
            
            # Display time and grasp state
            cv2.putText(frame, f"Time: {time_s:.2f}s", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            y = y0 + dy
            if "selected_grasp" in sample:
                cv2.putText(frame, f"Grasp: {sample['selected_grasp']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += dy
            
            if "contact_confirmed" in sample:
                cv2.putText(frame, f"Contact: {sample['contact_confirmed']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += dy
                
            forces = sample.get("forces", [])
            if forces:
                force_sum = sum(forces)
                cv2.putText(frame, f"Total Force: {force_sum:.2f} N", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += dy

            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        return True
    except Exception as e:
        print(f"Overlay failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Apply telemetry overlay to video")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--trajectory", type=Path, required=True, help="Input trajectory JSON path")
    parser.add_argument("--output", type=Path, required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=10, help="Minimum update frequency in Hz")
    
    args = parser.parse_args()
    
    if apply_overlay(args.video, args.trajectory, args.output, args.fps):
        print(f"Successfully wrote {args.output}")
        return 0
    return 1

if __name__ == "__main__":
    exit(main())
