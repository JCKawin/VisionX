#!/usr/bin/env python3
"""
Integrated object detection + 9-quadrant + approximate distance estimation
(using ncnn + OpenCV). Adapted from your provided code.

Requirements:
 - ncnn Python bindings
 - OpenCV (cv2)
 - numpy

Run:
 python3 integrated_detection.py
"""

import time
import cv2
import numpy as np
from ncnn import ncnn

# === Config ===
MODEL_PARAM = "./model.ncnn.param"
MODEL_BIN   = "./model.ncnn.bin"
INPUT_NAME  = "in0"
OUTPUT_NAME = "out0"
USB_INDEX   = 0                 # camera index (change to 1 if needed)
TARGET_FPS  = 60
CONF_THRESH = 0.20
NMS_THRESH  = 0.45
DISPLAY     = True              # show camera window (set False for headless)
SAVE_OUTPUT = False             # write annotated output to file
OUTPUT_PATH = "out_2fps_ncnn_annotated.mp4"
TARGET_SIZE = (640, 640)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CAN_MOVE = {
    "person","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe",
    "car","motorcycle","airplane","bus","train","truck","boat"
}

CANNOT_MOVE = {
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush","fire hydrant","stop sign","parking meter","bench"
}

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

# Approximate real-world heights (meters) for a few COCO classes.
# These are used to bias the simple size->distance heuristic.
# Add or tweak as needed for better approximations.
REAL_HEIGHTS_M = {
    "person": 1.7,
    "bicycle": 1.2,
    "car": 1.5,        # approx vehicle height (sedan)
    "motorcycle": 1.2,
    "bus": 3.0,
    "truck": 3.0,
    "boat": 2.0,
    "dog": 0.6,
    "cat": 0.25,
    "traffic light": 2.5,
    "train": 3.0,
    "truck": 3.0,
    "elephant": 3.0,
}

# Baseline constant to convert relative size to meters.
# This is heuristic — calibrate or replace with proper focal-length formula for accuracy.
DISTANCE_BASE = 0.5   # when bbox height == image height -> distance ~ DISTANCE_BASE meters

# === NCNN helper functions ===

def init_ncnn(param_path, bin_path, use_vulkan=True, num_threads=4):
    net = ncnn.Net()
    # try to set opts (bindings vary)
    try: net.opt.use_vulkan_compute = not bool(use_vulkan)
    except: pass
    try: net.opt.num_threads = int(num_threads)
    except: pass

    if net.load_param(param_path) != 0:
        raise RuntimeError(f"Failed to load param: {param_path}")
    if net.load_model(bin_path) != 0:
        raise RuntimeError(f"Failed to load model bin: {bin_path}")

    return net

def letterbox_to_ncnn_mat(img_bgr, target_size=(640,640)):
    target_w, target_h = target_size
    h, w = img_bgr.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    mat = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_BGR, target_w, target_h)

    mean_vals = []
    norm_vals = [1/255, 1/255, 1/255]
    mat.substract_mean_normalize(mean_vals, norm_vals)

    return mat, scale, pad_x, pad_y, new_w, new_h

def yolo11_postprocess(mat_out_np, orig_w, orig_h, scale, pad_x, pad_y):
    dets = []
    if mat_out_np is None:
        return dets

    # Expect shape starting with coords then class scores (you used 84 earlier)
    if mat_out_np.shape[0] < 5:
        print("Unexpected output shape:", mat_out_np.shape)
        return dets

    # first 4 rows = cx, cy, w, h (per detection)
    cx = mat_out_np[0]
    cy = mat_out_np[1]
    w  = mat_out_np[2]
    h  = mat_out_np[3]

    class_scores = mat_out_np[4:]
    # class_scores shape: (num_classes, num_detections)
    # if only one detection, dims may differ — ensure consistent axis
    if class_scores.ndim == 1:
        # single-detection case
        class_scores = class_scores.reshape(-1, 1)

    class_ids = np.argmax(class_scores, axis=0)
    scores    = np.max(class_scores, axis=0)

    conf_mask = scores > CONF_THRESH
    if not np.any(conf_mask):
        return dets

    cx = cx[conf_mask]
    cy = cy[conf_mask]
    w  = w[conf_mask]
    h  = h[conf_mask]
    class_ids = class_ids[conf_mask]
    scores    = scores[conf_mask]

    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    # map from letterboxed coordinates back to original image coords
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # clip
    x1 = np.clip(x1, 0, orig_w-1)
    y1 = np.clip(y1, 0, orig_h-1)
    x2 = np.clip(x2, 0, orig_w-1)
    y2 = np.clip(y2, 0, orig_h-1)

    boxes_cv = []
    for a,b,c,d in zip(x1,y1,x2,y2):
        boxes_cv.append([float(a), float(b), float(c-a), float(d-b)])

    # cv2.dnn.NMSBoxes expects list of [x,y,w,h] and scores list
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores.tolist(), CONF_THRESH, NMS_THRESH)
    if len(indices) == 0:
        return dets

    for idx in indices.flatten():
        x,y,w_box,h_box = boxes_cv[idx]
        dets.append((int(x), int(y), int(x+w_box), int(y+h_box),
                     float(scores[idx]), int(class_ids[idx])))

    return dets

# === Quadrant & distance helpers ===

def get_quadrant(x_center, y_center, img_w, img_h):
    """Return quadrant number 1..9 for the center point.
       Layout:
         1 2 3
         4 5 6
         7 8 9
    """
    # clamp centers
    x_center = max(0, min(x_center, img_w - 1))
    y_center = max(0, min(y_center, img_h - 1))

    third_w = img_w / 3.0
    third_h = img_h / 3.0

    col = int(x_center // third_w)
    row = int(y_center // third_h)

    # clip indices to 0..2
    col = min(max(col, 0), 2)
    row = min(max(row, 0), 2)

    quadrant = row * 3 + col + 1
    return quadrant

def estimate_distance_m(box_h_pixels, img_h_pixels, class_name):
    """Estimate distance in meters using a simple heuristic:
       distance ≈ (img_h / box_h) * DISTANCE_BASE * (real_h / person_h)
       This is approximate. Calibrate for your camera for better results.
    """
    if box_h_pixels <= 0:
        return float("inf")

    base = (img_h_pixels / (box_h_pixels + 1e-6)) * DISTANCE_BASE

    real_h = REAL_HEIGHTS_M.get(class_name, REAL_HEIGHTS_M.get("person", 1.7))
    person_h = REAL_HEIGHTS_M.get("person", 1.7)

    # scale distance by object real-world height relative to person
    distance_m = base * (real_h / person_h)

    # clamp reasonable range and round
    if distance_m > 100.0:
        distance_m = 100.0
    return round(distance_m, 2)

def group_label(class_name):
    """Return a high-level group for the class."""
    if class_name == "person":
        return "human"
    if class_name in {"car","truck","bus","train","motorcycle","bicycle"}:
        return "vehicle"
    if class_name in {"traffic light","stop sign"}:
        return "infrastructure"
    return "other"

# === Main run loop ===

def run(usb_index=USB_INDEX, target_fps=TARGET_FPS,
        model_param=MODEL_PARAM, model_bin=MODEL_BIN,
        input_name=INPUT_NAME, output_name=OUTPUT_NAME,
        target_size=TARGET_SIZE,
        display=DISPLAY, save_output=SAVE_OUTPUT, output_path=OUTPUT_PATH):

    cap = cv2.VideoCapture(usb_index)
    if not cap.isOpened():
        raise IOError(f"Could not open camera index {usb_index}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {orig_w}x{orig_h}")

    out_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, target_fps, (orig_w, orig_h))

    net = init_ncnn(model_param, model_bin)

    frame_interval = 1.0 / target_fps
    last_time = 0.0

    print("Starting main loop... press 'q' in window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed.")
                break

            now = time.time()
            if (now - last_time) < frame_interval:
                # small sleep to save CPU
                time.sleep(0.001)
                continue
            last_time = now

            mat_in, scale, pad_x, pad_y, new_w, new_h = letterbox_to_ncnn_mat(frame, target_size)

            ex = net.create_extractor()
            ex.input(input_name, mat_in)
            out = ex.extract(output_name)

            if isinstance(out, tuple):
                _, mat_out = out
            else:
                mat_out = out

            try:
                mat_out_np = np.array(mat_out)
            except Exception as e:
                mat_out_np = None

            detections = yolo11_postprocess(mat_out_np, orig_w, orig_h, scale, pad_x, pad_y)

            # draw 3x3 grid
            if display or out_writer:
                # copy to annotate
                vis = frame.copy()
                # vertical lines
                cv2.line(vis, (orig_w//3, 0), (orig_w//3, orig_h), (0,255,0), 1)
                cv2.line(vis, (2*orig_w//3, 0), (2*orig_w//3, orig_h), (0,255,0), 1)
                # horizontal lines
                cv2.line(vis, (0, orig_h//3), (orig_w, orig_h//3), (0,255,0), 1)
                cv2.line(vis, (0, 2*orig_h//3), (orig_w, 2*orig_h//3), (0,255,0), 1)
            else:
                vis = None

            for (x1, y1, x2, y2, score, cls) in detections:
                # guard class index overflow
                if cls < 0 or cls >= len(COCO_CLASSES):
                    continue
                class_name = COCO_CLASSES[cls]

                # For demonstration, we only print/meter moving objects if desired:
                # if class_name not in CAN_MOVE:
                #     continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                box_h = (y2 - y1)

                quadrant = get_quadrant(cx, cy, orig_w, orig_h)
                distance = estimate_distance_m(box_h, orig_h, class_name)
                grp = group_label(class_name)

                console_msg = f"{class_name} {score:.2f} quad={quadrant} dist~{distance}m group={grp} bbox=({x1},{y1},{x2},{y2})"
                print(console_msg)

                if vis is not None:
                    # draw bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 160, 255), 2)
                    # label text
                    label = f"{class_name} {score:.2f}"
                    info = f"Q{quadrant} {distance}m"
                    # combine for top-left text, ensure it fits
                    text = f"{label} | {info}"
                    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
                    tx = max(0, x1)
                    ty = max(15, y1 - 6)
                    cv2.rectangle(vis, (tx, ty-th-4), (tx+tw+4, ty+4), (0,160,255), -1)
                    cv2.putText(vis, text, (tx+2, ty-2), FONT, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    # draw center point
                    cv2.circle(vis, (int(cx), int(cy)), 3, (255,0,0), -1)

            # show/save annotated frame
            if vis is not None:
                if display:
                    cv2.imshow("Detection 3x3 + distance", vis)
                if out_writer:
                    out_writer.write(vis)

            # handle key press (non-blocking)
            if display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed.")
                    break

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        if display:
            cv2.destroyAllWindows()
        try:
            ncnn.destroy_gpu_instance()
        except Exception:
            pass

if __name__ == "__main__":
    run()
