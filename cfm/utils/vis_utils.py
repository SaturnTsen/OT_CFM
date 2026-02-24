# Convert mp4 to frames, and crop with some paddings
from PIL import Image, ImageChops
from moviepy import VideoFileClip
import imageio.v3 as iio
import os

def mp4_to_frames(video_path, out_dir, fps=None, ext="png"):
    os.makedirs(out_dir, exist_ok=True)

    clip = VideoFileClip(video_path)
    if fps is not None:
        clip = clip.with_fps(fps)   # ← 注意：with_fps，不是 set_fps

    for i, frame in enumerate(clip.iter_frames()):
        iio.imwrite(f"{out_dir}/frame_{i}.{ext}", frame)


def crop_frames(input_dir, output_dir, threshold=250, pad_ratio=0.05):
    os.makedirs(output_dir, exist_ok=True)

    bboxes = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".png"):
            continue
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        bbox = nonwhite_bbox(img, threshold)
        bboxes.append(bbox)

    common_bbox = intersect_bbox(bboxes)
    expanded_bbox = expand_bbox(common_bbox, img.size, pad_ratio)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".png"):
            continue
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        cropped_img = img.crop(expanded_bbox)
        cropped_img.save(os.path.join(output_dir, filename))
        
def nonwhite_bbox(img, threshold=250):
    img = img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    gray = diff.convert("L")
    bw = gray.point(lambda x: 0 if x < threshold else 255, "1")
    return bw.getbbox()

def intersect_bbox(bboxes):
    left   = max(b[0] for b in bboxes)
    upper  = max(b[1] for b in bboxes)
    right  = min(b[2] for b in bboxes)
    lower  = min(b[3] for b in bboxes)
    return left, upper, right, lower

def expand_bbox(bbox, img_size, pad_ratio=0.05):
    w, h = img_size
    l, u, r, d = bbox

    pad_x = int((r - l) * pad_ratio)
    pad_y = int((d - u) * pad_ratio)

    l = max(0, l - pad_x)
    u = max(0, u - pad_y)
    r = min(w, r + pad_x)
    d = min(h, d + pad_y)

    return l, u, r, d