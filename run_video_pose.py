import argparse
import pathlib

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda(non_blocking=True)

    with torch.no_grad():
        stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().numpy(), (1, 2, 0))
    heatmaps = cv2.resize(
        heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC
    )

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().numpy(), (1, 2, 0))
    pafs = cv2.resize(
        pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC
    )

    return heatmaps, pafs, scale, pad


def process_video(
    video_path: pathlib.Path,
    output_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    height_size: int,
    cpu: bool,
    track: bool,
    smooth: bool,
    compile_model: bool,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file '{video_path}' does not exist")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist")

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_state(net, checkpoint)

    net.eval()
    if not cpu:
        net = net.cuda()
    if compile_model and hasattr(torch, "compile"):
        net = torch.compile(net)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []

    while True:
        was_read, frame = cap.read()
        if not was_read:
            break

        orig_img = frame.copy()
        heatmaps, pafs, scale, pad = infer_fast(
            net, frame, height_size, stride, upsample_ratio, cpu
        )

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            heatmap = heatmaps[:, :, kpt_idx]
            total_keypoints_num += extract_keypoints(
                heatmap, all_keypoints_by_type, total_keypoints_num
            )

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                (all_keypoints[kpt_id, 0] * stride / upsample_ratio) - pad[1]
            ) / scale
            all_keypoints[kpt_id, 1] = (
                (all_keypoints[kpt_id, 1] * stride / upsample_ratio) - pad[0]
            ) / scale

        current_poses = []
        for entry in pose_entries:
            if len(entry) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if entry[kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(entry[kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(entry[kpt_id]), 1])
            pose = Pose(pose_keypoints, entry[18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        for pose in current_poses:
            pose.draw(frame)

        frame = cv2.addWeighted(orig_img, 0.6, frame, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(
                frame,
                (pose.bbox[0], pose.bbox[1]),
                (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
                (0, 255, 0),
            )
            if track:
                cv2.putText(
                    frame,
                    f"id: {pose.id}",
                    (pose.bbox[0], pose.bbox[1] - 16),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                )

        writer.write(frame)

    cap.release()
    writer.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lightweight OpenPose on a video and save the visualization."
    )
    script_dir = pathlib.Path(__file__).resolve().parent
    default_checkpoint = script_dir / "pre_model" / "checkpoint_iter_370000.pth"

    parser.add_argument(
        "video_path",
        type=pathlib.Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        default=default_checkpoint,
        help="Path to the model checkpoint to use.",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=pathlib.Path("pose_output.mp4"),
        help="Path where the rendered video will be written.",
    )
    parser.add_argument(
        "--height-size",
        type=int,
        default=256,
        help="Network input height. Higher values improve quality at the cost of speed.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable pose id tracking between frames.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable temporal smoothing for pose tracking.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile (requires PyTorch 2.0+).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        args.video_path,
        args.output_path,
        args.checkpoint_path,
        args.height_size,
        args.cpu,
        track=not args.no_track,
        smooth=not args.no_smooth,
        compile_model=args.compile,
    )
