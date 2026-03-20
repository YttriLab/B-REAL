import argparse
import numpy as np
from tune_cropping import FindRatBoundingBox, get_background_avg
from tqdm import tqdm
import cv2
import os

def crop_video(video_path, output_dir, num_frames_crop, crop_parameters, desired_framerate=None, cached_bg=None, output_name=None):
    """
    Extract cropped frames from a video, saving each as a flat PNG.

    Output: output_dir/video_name/0.png, 1.png, 2.png, ...
    One file per processed frame — no duplication. Temporal grouping
    (e.g. triplets for TSM) happens at dataset load time.
    """
    vid_name = output_name or os.path.basename(video_path).split('.')[0]
    out_sub_dir = os.path.join(output_dir, vid_name)
    os.makedirs(out_sub_dir, exist_ok=True)

    if cached_bg:
        bg_avg = cv2.imread(cached_bg, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        bg_avg = get_background_avg(vid_path=video_path, num_frames=num_frames_crop)
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    cropper = FindRatBoundingBox(bg_avg, **crop_parameters)

    if fps:
        if desired_framerate is not None:
            if fps % desired_framerate == 0:
                skip_size = int(fps // desired_framerate)
            else:
                raise ValueError("Video framerate not an integer multiple of desired framerate")
        else:
            skip_size = 1
    else:
        raise ValueError("Cannot read framerate from video codec")

    frame_counter = 0
    processed_counter = 0
    pbar = tqdm(total=length // skip_size)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        if frame_counter % skip_size != 0:
            frame_counter += 1
            continue

        *_, frame_cropped = cropper(frame)

        # Save each frame individually (flat structure)
        out_path = os.path.join(out_sub_dir, f'{processed_counter}.png')
        cv2.imwrite(out_path, frame_cropped)

        frame_counter += 1
        processed_counter += 1
        pbar.update(1)

    pbar.close()
    video.release()
    print(f"Saved {processed_counter} frames to {out_sub_dir}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess video into flat cropped frames')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--video', type=str, default=r"C:\Realtime\data\2026-02-23-12-02\video.mp4",
                        help='Path to input video')
    parser.add_argument('--bg', type=str, default=r"backgrounds\bg_20260223_114136.png",
                        help='Path to cached background image')
    parser.add_argument('--framerate', type=int, default=30,
                        help='Desired output framerate (default: 30)')
    parser.add_argument('--name', type=str, default=None,
                        help='Override output subdirectory name (default: derived from video filename)')
    args = parser.parse_args()

    crop_parameters = {
        'box_size': 220,
        'difference_threshold': 5
    }

    out_path = args.output_dir or r"C:\Users\Jimi\Desktop\Brayden\RealtimeLightning\RealtimeBehaviorClassification\processed_frames"

    print(f"Output directory: {out_path}")
    if args.name:
        print(f"Output subdirectory name: {args.name}")

    crop_video(video_path=args.video, output_dir=out_path,
               desired_framerate=args.framerate, num_frames_crop=500,
               crop_parameters=crop_parameters, cached_bg=args.bg, output_name=args.name)

if __name__ == "__main__":
    main()
