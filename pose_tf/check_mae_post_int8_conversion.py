import numpy as np, cv2, tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
IMG_DIR = ROOT / "coco" / "val2017"

imgs = sorted(IMG_DIR.glob("*.jpg"))[:5]
if not imgs:
    raise FileNotFoundError(f"No images found in {IMG_DIR}. Place COCO val JPGs there or point IMG_DIR elsewhere.")

def load(path):
    interp = tf.lite.Interpreter(model_path=str(path)); interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()

int8, i8_in, outs8   = load(MODEL_DIR / "pose_int8.tflite")
fp16, f16_in, outs16 = load(MODEL_DIR / "human-pose-estimation-new-single_float16.tflite")

# match outputs by channel count (38 = PAF, 19 = heatmap)
matches = {}
for o8 in outs8:
    c = o8["shape"][-1]
    matches[c] = [o8, next(o16 for o16 in outs16 if o16["shape"][-1]==c)]

def run(interp, inp_detail, img):
    h, w = inp_detail["shape"][1:3]
    img = cv2.resize(img, (w, h)).astype(np.float32)
    img = (img - 128.0) * (1/256.0)
    if inp_detail["dtype"] == np.int8:
        scale, zero = inp_detail["quantization"]
        img = np.clip(img / scale + zero, -128, 127).astype(np.int8)
    interp.set_tensor(inp_detail["index"], img[None, ...])
    interp.invoke()

maes = {c: [] for c in matches}
for p in imgs:
    img = cv2.imread(str(p))
    if img is None: continue
    run(int8, i8_in, img)
    run(fp16, f16_in, img)
    for c,(o8,o16) in matches.items():
        y8  = int8.get_tensor(o8["index"]).astype(np.float32)
        y16 = fp16.get_tensor(o16["index"]).astype(np.float32)
        # dequantize int8 for fair comparison
        s,z = o8["quantization"]
        y8 = (y8 - z) * s
        maes[c].append(np.mean(np.abs(y8 - y16)))

for c,v in maes.items():
    print(f"MAE for {c} channels: {np.mean(v):.6f}")
