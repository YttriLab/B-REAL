import cv2
import numpy as np
from functools import partial


def pop_resize_factor(crop_params: dict):
    """Split a full-frame pre-resize out of crop_params.

    crop_params is otherwise splatted straight into FindRatBoundingBox(**...),
    which would reject an unknown 'resize_factor' kwarg. This returns
    (clean_params, factor) where factor (default 1.0) scales the entire frame
    (and background) before cropping. Values <= 0 are treated as 1.0 (no resize).
    """
    cp = dict(crop_params or {})
    try:
        rf = float(cp.pop("resize_factor", 1.0))
    except (TypeError, ValueError):
        rf = 1.0
    return cp, (rf if rf and rf > 0 else 1.0)


def resize_frame(img, factor):
    """Resize a full frame/background by `factor` (INTER_AREA). No-op at 1.0."""
    if img is None or factor == 1.0:
        return img
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


class FindRatBoundingBox():
    def __init__(self, bg, difference_threshold=5, box_size=100,
                 downsample_factor=1, open1_kernel=4, close_kernel=15, open2_kernel=10,
                 difference_mode="directional", smoothing=0.0, deadband=0.0,
                 min_contour_area=0.0):
        self.difference_threshold = difference_threshold
        # "directional" = subject is darker than background (small - bg < -t);
        # "absolute" = any change from background (|small - bg| > t).
        self.difference_mode = difference_mode
        self.box_size = box_size
        self.downsample_factor = downsample_factor
        self.open1_kernel = open1_kernel
        self.close_kernel = close_kernel
        self.open2_kernel = open2_kernel
        # Temporal stabilization (all in downsampled "small" space):
        #   smoothing  — EMA weight on the previous center, 0..1 (0 = off).
        #   deadband   — don't move the box until the detected center drifts this
        #                many px from the smoothed center (kills micro-jitter).
        #   min_contour_area — ignore blobs smaller than this (drops specks).
        self.smoothing = max(0.0, min(1.0, smoothing))
        self.deadband = deadband
        self.min_contour_area = min_contour_area
        self.smoothed_center = None  # (cx, cy) in small space; persists across frames
        self.h, self.w = bg.shape[:2]
        self.resizer = partial(cv2.resize, dsize=(self.w // downsample_factor, self.h // downsample_factor), interpolation=cv2.INTER_NEAREST)
        self.bg = self.resizer(bg)
        self.prev_bounding = (0, 0, self.w, self.h)

    def resize(self, frame):
        return self.resizer(frame)

    @staticmethod
    def _contour_centroid(cnt):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            return (M["m10"] / M["m00"], M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        return (x + w / 2.0, y + h / 2.0)

    def _select_contour(self, contours):
        """Pick the blob to track: drop specks below min_contour_area; when
        smoothing is on and we have a previous center, prefer the nearest of the
        comparably-large blobs (avoids flipping to a spurious/secondary blob);
        otherwise take the largest."""
        cands = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        if not cands:
            cands = list(contours)
        if self.smoothing > 0 and self.smoothed_center is not None and len(cands) > 1:
            max_area = max(cv2.contourArea(c) for c in cands)
            big = [c for c in cands if cv2.contourArea(c) >= 0.5 * max_area]
            sx, sy = self.smoothed_center
            return min(big, key=lambda c: (lambda cx, cy: (cx - sx) ** 2 + (cy - sy) ** 2)(*self._contour_centroid(c)))
        return max(cands, key=cv2.contourArea)

    def _apply_smoothing(self, raw_cx, raw_cy):
        """EMA + dead-band on the crop center (small space). Returns (cx, cy)."""
        if self.smoothing <= 0 or self.smoothed_center is None:
            self.smoothed_center = (raw_cx, raw_cy)
            return raw_cx, raw_cy
        px, py = self.smoothed_center
        if self.deadband > 0 and ((raw_cx - px) ** 2 + (raw_cy - py) ** 2) ** 0.5 <= self.deadband:
            return px, py  # within dead-band: hold the box still
        a = self.smoothing
        nx = (1 - a) * raw_cx + a * px
        ny = (1 - a) * raw_cy + a * py
        self.smoothed_center = (nx, ny)
        return nx, ny

    def crop(self, frame):
        # Convert RGB to grayscale for processing if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        small = self.resizer(gray)
        diff = small - self.bg
        if self.difference_mode == "absolute":
            mask = np.abs(diff) > self.difference_threshold
        else:  # "directional": subject darker than background
            mask = diff < -self.difference_threshold
        thresholded = (mask * 255).astype(np.uint8)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN,
                                  np.ones((self.open1_kernel, self.open1_kernel), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                                  np.ones((self.close_kernel, self.close_kernel), np.uint8))
        opened_again = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                                        np.ones((self.open2_kernel, self.open2_kernel), np.uint8))
        contours, _ = cv2.findContours(opened_again, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = self._select_contour(contours)
            raw_cx, raw_cy = self._contour_centroid(cnt)
            # Temporal smoothing of the center keeps the crop steady so TSM sees
            # the animal's own motion, not box jitter.
            center_x, center_y = self._apply_smoothing(raw_cx, raw_cy)

            # Create fixed-size box centered on the (smoothed) center
            new_x = max(0, int(round(center_x)) - self.box_size // 2)
            new_y = max(0, int(round(center_y)) - self.box_size // 2)
            bb = (new_x, new_y, self.box_size, self.box_size)
            self.prev_bounding = bb
            
            # Crop from original frame (scale back to original size)
            x_scaled = new_x * self.downsample_factor
            y_scaled = new_y * self.downsample_factor
            w_scaled = self.box_size * self.downsample_factor
            h_scaled = self.box_size * self.downsample_factor
            
            # Ensure crop doesn't go out of bounds
            x_scaled = max(0, min(x_scaled, frame.shape[1] - w_scaled))
            y_scaled = max(0, min(y_scaled, frame.shape[0] - h_scaled))
            x_end = min(x_scaled + w_scaled, frame.shape[1])
            y_end = min(y_scaled + h_scaled, frame.shape[0])
            
            cropped_rgb = frame[y_scaled:y_end, x_scaled:x_end]
            
            return bb, small, opened_again, cropped_rgb
        
        print("No contours detected")
        # Return previous bounding box and empty crop
        x_scaled = self.prev_bounding[0] * self.downsample_factor
        y_scaled = self.prev_bounding[1] * self.downsample_factor
        w_scaled = self.prev_bounding[2] * self.downsample_factor
        h_scaled = self.prev_bounding[3] * self.downsample_factor
        
        x_scaled = max(0, min(x_scaled, frame.shape[1] - w_scaled))
        y_scaled = max(0, min(y_scaled, frame.shape[0] - h_scaled))
        x_end = min(x_scaled + w_scaled, frame.shape[1])
        y_end = min(y_scaled + h_scaled, frame.shape[0])
        
        cropped_rgb = frame[y_scaled:y_end, x_scaled:x_end]
        
        return self.prev_bounding, small, opened_again, cropped_rgb
    
    def __call__(self, frame):
        return self.crop(frame)

def get_background_avg(vid_path, num_frames):
    cap = cv2.VideoCapture(vid_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.zeros((num_frames, height, width))
    idxs = np.random.randint(0, frame_count, num_frames)
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise Exception("Can't read frame")
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = frames.mean(axis=0)
    cap.release()
    return avg

def get_background_avg_live(cam_idx, num_frames):
    cap = cv2.VideoCapture(cam_idx)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.zeros((num_frames, height, width))

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            raise Exception("Can't read frame")
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    avg = frames.mean(axis=0)
    cap.release()

    return avg

class ParameterTuner:
    CTRL_WIN = 'Controls'
    VIZ_WIN  = 'Visualization'
    SCALE    = 0.5   # display scale factor for all panels

    def __init__(self, video_path, num_bg_frames=500, bg_path=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initial parameters
        self.params = {
            'difference_threshold': 5,
            'box_size': 100,
            'downsample_factor': 1,
            'open1_kernel': 4,
            'close_kernel': 15,
            'open2_kernel': 10
        }
        
        # Get background
        print("Computing background...")
        if bg_path is not None:
            self.bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        else:
            self.bg = get_background_avg(video_path, num_bg_frames)
        
        # ── Controls window (trackbars only) ─────────────────────────────────
        cv2.namedWindow(self.CTRL_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.CTRL_WIN, 400, 200)
        cv2.moveWindow(self.CTRL_WIN, 50, 50)
        cv2.imshow(self.CTRL_WIN, np.zeros((1, 400, 3), dtype=np.uint8))

        cv2.createTrackbar('Diff Threshold', self.CTRL_WIN,
                           self.params['difference_threshold'], 50,
                           self.update_diff_threshold)
        cv2.createTrackbar('Box Size', self.CTRL_WIN,
                           self.params['box_size'], 500,
                           self.update_box_size)
        cv2.createTrackbar('Downsample', self.CTRL_WIN,
                           self.params['downsample_factor'], 4,
                           self.update_downsample)
        cv2.createTrackbar('Open1 Kernel', self.CTRL_WIN,
                           self.params['open1_kernel'], 30,
                           self.update_open1_kernel)
        cv2.createTrackbar('Close Kernel', self.CTRL_WIN,
                           self.params['close_kernel'], 30,
                           self.update_close_kernel)
        cv2.createTrackbar('Open2 Kernel', self.CTRL_WIN,
                           self.params['open2_kernel'], 30,
                           self.update_open2_kernel)

        # ── Visualization window ─────────────────────────────────────────────
        # WINDOW_AUTOSIZE sizes the window to exactly the image passed to
        # imshow — no stretching is possible.
        cv2.namedWindow(self.VIZ_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.VIZ_WIN, 500, 50)

        self.update_tracker()
    
    def update_diff_threshold(self, val):
        self.params['difference_threshold'] = max(1, val)
        self.update_tracker()
    
    def update_box_size(self, val):
        self.params['box_size'] = max(10, val)
        self.update_tracker()
    
    def update_downsample(self, val):
        self.params['downsample_factor'] = max(1, val)
        self.update_tracker()
    
    def update_open1_kernel(self, val):
        self.params['open1_kernel'] = max(1, val)
        self.update_tracker()
    
    def update_close_kernel(self, val):
        self.params['close_kernel'] = max(1, val)
        self.update_tracker()
    
    def update_open2_kernel(self, val):
        self.params['open2_kernel'] = max(1, val)
        self.update_tracker()
    
    def toggle_play(self, val):
        self.playing = (val == 1)
    
    def update_tracker(self):
        self.tracker = FindRatBoundingBox(
            self.bg,
            difference_threshold=self.params['difference_threshold'],
            box_size=self.params['box_size'],
            downsample_factor=self.params['downsample_factor'],
            open1_kernel=self.params['open1_kernel'],
            close_kernel=self.params['close_kernel'],
            open2_kernel=self.params['open2_kernel']
        )
    
    def run(self):
        print("\nControls:")
        print("  Q/ESC: Quit")
        print("  S: Save current parameters")
        print("  B: Recapture background")
        print("\nAdjust trackbars in the Controls window.\n")

        # Flush stale frames from the camera buffer
        for _ in range(5):
            self.cap.read()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break

            # frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

            (x, y, w, h), small, mask, cropped_rgb = self.tracker.crop(frame)

            # Scale bounding box back to pre-downsample coordinates
            s = self.params['downsample_factor']
            frame_with_box = frame.copy()
            cv2.rectangle(frame_with_box,
                          (x * s, y * s),
                          ((x + w) * s, (y + h) * s),
                          (0, 255, 0), 2)

            # ── All panels scaled to SCALE before compositing ─────────────────
            SC = self.SCALE

            # p1: live frame with bounding box
            p1 = cv2.resize(frame_with_box, None, fx=SC, fy=SC,
                            interpolation=cv2.INTER_AREA)
            cv2.putText(p1, "Live + BBox", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # p2: mask — resize to exactly match p1
            p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            p2 = cv2.resize(p2, (p1.shape[1], p1.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            cv2.putText(p2, "Mask", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # p3: cropped region at scaled size
            CROP_D = int(300 * SC)
            if cropped_rgb.shape[0] > 0 and cropped_rgb.shape[1] > 0:
                p3 = cv2.resize(cropped_rgb, (CROP_D, CROP_D))
            else:
                p3 = np.zeros((CROP_D, CROP_D, 3), dtype=np.uint8)
            cv2.putText(p3, "Cropped", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Top row: live frame | mask  (side by side, same height)
            top_row = np.hstack([p1, p2])

            # Bottom row: cropped + param text, same total width as top row
            total_w = top_row.shape[1]
            bottom_row = np.zeros((CROP_D, total_w, 3), dtype=np.uint8)
            bottom_row[:CROP_D, :CROP_D] = p3

            tx = CROP_D + 20
            ty = 24
            for label in [
                f"Diff Thr : {self.params['difference_threshold']}",
                f"Box Size : {self.params['box_size']}",
                f"Downsamp : {self.params['downsample_factor']}",
                f"Open1    : {self.params['open1_kernel']}",
                f"Close    : {self.params['close_kernel']}",
                f"Open2    : {self.params['open2_kernel']}",
            ]:
                cv2.putText(bottom_row, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                ty += 24

            display = np.vstack([top_row, bottom_row])

            # WINDOW_AUTOSIZE ensures window matches display pixel-for-pixel
            cv2.imshow(self.VIZ_WIN, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self.save_parameters()
            elif key == ord('i'):
                self.save_background()
            elif key == ord('b'):
                self.recapture_background()

        cv2.destroyAllWindows()
    
    def save_parameters(self):
        print("\n" + "="*60)
        print("CURRENT PARAMETERS:")
        print("="*60)
        print("FindRatBoundingBox(")
        print("    bg,")
        print(f"    difference_threshold={self.params['difference_threshold']},")
        print(f"    box_size={self.params['box_size']},")
        print(f"    downsample_factor={self.params['downsample_factor']},")
        print(f"    open1_kernel={self.params['open1_kernel']},")
        print(f"    close_kernel={self.params['close_kernel']},")
        print(f"    open2_kernel={self.params['open2_kernel']}")
        print(")")
        print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    tuner = ParameterTuner(video_path, bg_path=r"backgrounds\bg_20260223_114136.png")
    tuner.run()