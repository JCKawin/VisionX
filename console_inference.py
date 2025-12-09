import time
import cv2
import numpy as np
from ncnn import ncnn

MODEL_PARAM = "./model.ncnn.param"
MODEL_BIN   = "./model.ncnn.bin"
INPUT_NAME  = "in0"
OUTPUT_NAME = "out0"
USB_INDEX   = 1
TARGET_FPS  = 2
CONF_THRESH = 0.20
NMS_THRESH  = 0.45
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

def init_ncnn(param_path, bin_path, use_vulkan=True, num_threads=4):
    net = ncnn.Net()
    try: net.opt.use_vulkan_compute = not bool(use_vulkan)
    except: pass
    try: net.opt.num_threads = int(num_threads)
    except: pass

    if net.load_param(param_path) != 0:
        raise RuntimeError(f"Failed param: {param_path}")
    if net.load_model(bin_path) != 0:
        raise RuntimeError(f"Failed bin: {bin_path}")

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

    if mat_out_np.shape[0] != 84:
        print("Unexpected output shape:", mat_out_np.shape)

    cx = mat_out_np[0]
    cy = mat_out_np[1]
    w  = mat_out_np[2]
    h  = mat_out_np[3]

    class_scores = mat_out_np[4:84]

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

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    x1 = np.clip(x1, 0, orig_w-1)
    y1 = np.clip(y1, 0, orig_h-1)
    x2 = np.clip(x2, 0, orig_w-1)
    y2 = np.clip(y2, 0, orig_h-1)

    boxes_cv = []
    for a,b,c,d in zip(x1,y1,x2,y2):
        boxes_cv.append([float(a), float(b), float(c-a), float(d-b)])

    indices = cv2.dnn.NMSBoxes(boxes_cv, scores.tolist(), CONF_THRESH, NMS_THRESH)
    if len(indices) == 0:
        return dets

    for idx in indices.flatten():
        x,y,w,h = boxes_cv[idx]
        dets.append((int(x), int(y), int(x+w), int(y+h),
                     float(scores[idx]), int(class_ids[idx])))

    return dets

def run(usb_index=USB_INDEX, target_fps=TARGET_FPS,
        model_param=MODEL_PARAM, model_bin=MODEL_BIN,
        input_name=INPUT_NAME, output_name=OUTPUT_NAME,
        target_size=(640,640),
        save_output=False, output_path="out_2fps_ncnn.mp4"):

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

    print("Starting main loop (2 FPS, headless)...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed.")
                break

            now = time.time()
            if (now - last_time) >= frame_interval:
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
                except:
                    mat_out_np = None

                detections = yolo11_postprocess(mat_out_np, orig_w, orig_h, scale, pad_x, pad_y)

                for (x1, y1, x2, y2, score, cls) in detections:
                    class_name = COCO_CLASSES[cls]

                    # ONLY print can-move objects
                    if class_name not in CAN_MOVE:
                        continue

                    # NEW PRINT FORMAT
                    print(f"{class_name} {score:.2f} ({x1},{y1},{x2},{y2})")

                if out_writer:
                    out_writer.write(frame)

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        try:
            ncnn.destroy_gpu_instance()
        except:
            pass


if __name__ == "__main__":
    run()
