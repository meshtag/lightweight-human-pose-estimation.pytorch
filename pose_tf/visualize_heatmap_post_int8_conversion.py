import numpy as np, cv2, tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
IMG_DIR = ROOT / "coco" / "val2017"

imgs = sorted(IMG_DIR.glob("*.jpg"))
if not imgs:
    raise FileNotFoundError(f"No images found in {IMG_DIR}. Place a COCO val image there or point the script to one.")

img_path = imgs[0]
print("Using image:", img_path)

def load(path):
    i = tf.lite.Interpreter(model_path=str(path)); i.allocate_tensors()
    return i, i.get_input_details()[0], i.get_output_details()

int8, i8_in, outs8   = load(MODEL_DIR / "pose_int8.tflite")
fp16, f16_in, outs16 = load(MODEL_DIR / "human-pose-estimation-new-single_float16.tflite")

# Match outputs by channel count
match = lambda c: next(o for o in outs16 if o["shape"][-1] == c)
paf8, paf16 = outs8[0], match(outs8[0]["shape"][-1])      # 38 channels
hm8,  hm16  = outs8[-1], match(outs8[-1]["shape"][-1])    # 19 channels

def run(interp, inp_detail, img):
    h, w = inp_detail["shape"][1:3]
    img = cv2.resize(img, (w, h)).astype(np.float32)
    img = (img - 128.0) * (1/256.0)
    if inp_detail["dtype"] == np.int8:
        s, z = inp_detail["quantization"]
        img = np.clip(img / s + z, -128, 127).astype(np.int8)
    interp.set_tensor(inp_detail["index"], img[None, ...]); interp.invoke()

img = cv2.imread(str(img_path))
run(int8, i8_in, img); run(fp16, f16_in, img)

def dequant(out, detail):
    s, z = detail["quantization"]; return (out.astype(np.float32) - z) * s

paf8_out = dequant(int8.get_tensor(paf8["index"]), paf8)
paf16_out = fp16.get_tensor(paf16["index"]).astype(np.float32)
hm8_out = dequant(int8.get_tensor(hm8["index"]), hm8)
hm16_out = fp16.get_tensor(hm16["index"]).astype(np.float32)

def show(title, a, b, cidx=0):
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(a[0,:,:,cidx], cmap='magma'); ax[0].set_title(f"{title} int8 c{cidx}")
    ax[1].imshow(b[0,:,:,cidx], cmap='magma'); ax[1].set_title(f"{title} fp16 c{cidx}")
    diff = np.abs(a[0,:,:,cidx]-b[0,:,:,cidx])
    ax[2].imshow(diff, cmap='viridis'); ax[2].set_title(f"abs diff (mean {diff.mean():.4f})")
    for a_ in ax: a_.axis("off")
    plt.tight_layout(); plt.show()

show("Heatmap", hm8_out, hm16_out, cidx=0)   # try other cidx values
show("PAF", paf8_out, paf16_out, cidx=0)
