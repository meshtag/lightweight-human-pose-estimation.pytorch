import torch
from pathlib import Path
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

root = Path(__file__).resolve().parent
ckpt = root / "pre_model" / "checkpoint_iter_370000.pth"
out = root / "pose_tf" / "human-pose-estimation.onnx"

net = PoseEstimationWithMobileNet()
load_state(net, torch.load(ckpt, map_location="cpu"))
net.eval()
dummy = torch.randn(1, 3, 256, 456)
torch.onnx.export(
    net, dummy, str(out),
    input_names=["data"],
    output_names=[
        "stage_0_output_1_heatmaps", "stage_0_output_0_pafs",
        "stage_1_output_1_heatmaps", "stage_1_output_0_pafs",
    ],
    opset_version=18, do_constant_folding=True,
    use_external_data_format=False,
)
print(f"Saved {out}")
