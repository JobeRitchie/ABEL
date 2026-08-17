import sys
from pathlib import Path
import numpy as np
import cv2


def build_clip(video_path: Path, n_frames=16, out_size=(112, 112)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    try:
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_video <= 0:
            raise RuntimeError("video has no frames")
        idx = np.linspace(0, n_video - 1, num=n_frames, dtype=int)
        clip = np.zeros((n_frames, out_size[0], out_size[1], 3), dtype=np.uint8)
        for i, fno in enumerate(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame {fno} from {video_path}")
            h, w = frame.shape[:2]
            side = min(h, w)
            cx = w // 2
            cy = h // 2
            x0 = max(0, cx - side // 2)
            y0 = max(0, cy - side // 2)
            crop = frame[y0 : y0 + side, x0 : x0 + side]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            clip[i] = cv2.resize(rgb, out_size, interpolation=cv2.INTER_AREA)
        return clip[np.newaxis, ...]
    finally:
        cap.release()


def main():
    if len(sys.argv) < 2:
        print("Usage: run_r3d_sample.py PATH_TO_VIDEO")
        sys.exit(2)
    video = Path(sys.argv[1])
    if not video.exists():
        print("Video not found:", video)
        sys.exit(2)
    print("Building clip from", video)
    clip = build_clip(video)
    print("Clip shape:", clip.shape, "dtype:", clip.dtype)

    # Import and run the project's embed function
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from abel.services import r3d_feature_service as r3d

    try:
        emb = r3d.R3DFeatureService._embed(clip)
    except Exception as e:
        print("_embed raised:", e)
        raise
    print("Embedding shape:", emb.shape)
    print("Embedding stats: min={}, max={}, mean={}, std={}".format(
        float(np.nanmin(emb)), float(np.nanmax(emb)), float(np.nanmean(emb)), float(np.nanstd(emb))
    ))


if __name__ == "__main__":
    main()
