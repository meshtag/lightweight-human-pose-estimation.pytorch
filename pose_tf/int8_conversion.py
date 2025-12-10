import tensorflow as tf, cv2, numpy as np
from pathlib import Path

WIDTH = 456
HEIGHT = 256
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "coco" / "val2017"
SAVED_MODEL_DIR = ROOT

imgs = list(DATA_DIR.glob("*.jpg"))[:100]
if not imgs:
    raise FileNotFoundError(f"No images found in {DATA_DIR}. Add COCO val JPGs for calibration.")


def rep_ds():
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = (img.astype(np.float32) - 128.0) * (1 / 256.0)  # BGR norm
        yield [img[np.newaxis, ...]]


if not (SAVED_MODEL_DIR / "saved_model.pb").exists():
    raise FileNotFoundError(f"SavedModel not found in {SAVED_MODEL_DIR}")

conv = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
open("pose_int8.tflite", "wb").write(conv.convert())
