#!/usr/bin/env python3
"""
Convenience launcher to validate the 2 MB student model with Google Drive paths.

Uses val.py with the provided validation labels, images, and checkpoint:
- Checkpoint:       /content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_2MB/best.pth
- Labels JSON:      /content/drive/MyDrive/MLonMCU/Datasets/annotations/person_keypoints_val2014.json
- Images folder:    /content/drive/MyDrive/MLonMCU/Datasets/val2014

Default device is GPU; add --cpu flag to force CPU inference.
Add --multiscale for multi-scale evaluation (slower but more accurate).
Add --visualize to show keypoints during evaluation.
"""

import sys
import os

# Add necessary paths
POSE_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'lightweight-human-pose-estimation.pytorch')
AI8X_REPO = os.path.dirname(__file__)
sys.path.insert(0, POSE_REPO)
sys.path.insert(0, AI8X_REPO)

import torch
import torch.nn as nn
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.coco import CocoValDataset
from modules.keypoints import extract_keypoints, group_keypoints
import ai8x

ai8x.set_device(device=85, simulate=False, round_avg=False)


class StudentModel2MB(nn.Module):
    """Narrow student sized for ~2 MB FP32 checkpoints"""
    def __init__(self):
        super().__init__()
        # Backbone (stride down to 16x16 for 128x128 inputs)
        self.conv1 = ai8x.FusedConv2dBNReLU(3, 24, 3, stride=2, padding=1, bias=True)
        self.conv2 = ai8x.FusedConv2dBNReLU(24, 24, 3, stride=1, padding=1, bias=True)
        self.conv3 = ai8x.FusedConv2dBNReLU(24, 48, 3, stride=1, padding=1, bias=True)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(48, 48, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv5 = ai8x.FusedConv2dBNReLU(48, 64, 3, stride=1, padding=1, bias=True)
        self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv7 = ai8x.FusedConv2dBNReLU(64, 96, 3, stride=1, padding=1, bias=True)
        self.conv8 = ai8x.FusedConv2dBNReLU(96, 96, 3, stride=1, padding=1, bias=True)
        # CPM head
        self.cpm1 = ai8x.FusedConv2dBNReLU(96, 64, 1, padding=0, bias=True)
        self.cpm2 = ai8x.FusedConv2dBNReLU(64, 64, 3, padding=1, bias=True)
        self.cpm3 = ai8x.FusedConv2dBNReLU(64, 64, 3, padding=1, bias=True)
        # Output heads (same channels as teacher outputs)
        self.heat_conv = ai8x.FusedConv2dBNReLU(64, 48, 1, padding=0, bias=True)
        self.heat_out = ai8x.Conv2d(48, 19, 1, padding=0, bias=True, wide=True)
        self.paf_conv = ai8x.FusedConv2dBNReLU(64, 48, 1, padding=0, bias=True)
        self.paf_out = ai8x.Conv2d(48, 38, 1, padding=0, bias=True, wide=True)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x); x = self.conv6(x)
        x = self.conv7(x); x = self.conv8(x)
        x = self.cpm1(x); x = self.cpm2(x); x = self.cpm3(x)
        h = self.heat_out(self.heat_conv(x))
        p = self.paf_out(self.paf_conv(x))
        return [h, p]


def normalize_student(img):
    """Normalize for student model: (img - 128) / 128"""
    img = np.array(img, dtype=np.float32)
    img = (img - 128.0) / 128.0
    return img


def infer_student(net, img, input_size=128, device=torch.device('cuda')):
    """
    Inference for student models that expect 128x128 input and output 16x16 directly.
    """
    # Resize image to 128x128 maintaining aspect ratio with padding
    h, w, _ = img.shape
    scale = input_size / max(h, w)
    scaled_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # Pad to square
    pad_h = input_size - scaled_img.shape[0]
    pad_w = input_size - scaled_img.shape[1]
    padded_img = cv2.copyMakeBorder(scaled_img, 0, pad_h, 0, pad_w, 
                                     cv2.BORDER_CONSTANT, value=(128, 128, 128))
    
    # Normalize
    normed_img = normalize_student(padded_img)
    
    # Run model
    tensor_img = torch.from_numpy(normed_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = net(tensor_img)
    
    # Outputs are [heatmaps (1,19,16,16), pafs (1,38,16,16)]
    heatmaps = outputs[0].squeeze().cpu().numpy()  # (19, 16, 16)
    pafs = outputs[1].squeeze().cpu().numpy()      # (38, 16, 16)
    
    # Transpose to (H, W, C) and upscale to original image size
    heatmaps = np.transpose(heatmaps, (1, 2, 0))  # (16, 16, 19)
    pafs = np.transpose(pafs, (1, 2, 0))          # (16, 16, 38)
    
    # Upscale to 128x128 first (model's input size)
    heatmaps = cv2.resize(heatmaps, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    
    # Remove padding and scale back to original size
    heatmaps = heatmaps[0:input_size-pad_h, 0:input_size-pad_w, :]
    pafs = pafs[0:input_size-pad_h, 0:input_size-pad_w, :]
    
    heatmaps = cv2.resize(heatmaps, (w, h), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return heatmaps, pafs


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def evaluate_student(labels, output_name, images_folder, net, visualize=False, device=torch.device('cuda')):
    """Evaluate student model with correct 128x128 input handling"""
    net = net.to(device).eval()
    
    dataset = CocoValDataset(labels, images_folder)
    coco_result = []
    
    for sample in dataset:
        file_name = sample['file_name']
        img = sample['img']
        
        heatmaps, pafs = infer_student(net, img, input_size=128, device=device)
        
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        
        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)
        
        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })
        
        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey()
            if key == 27:  # esc
                return
    
    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)
    
    run_coco_eval(labels, output_name)


def build_args():
    drive_model_dir = "/content/drive/MyDrive/MLonMCU/Models/MAX78000/IP_2MB"
    drive_checkpoint = f"{drive_model_dir}/best.pth"
    return [
        "val_pose_drive_2mb-3_9.py",
        "--labels", "/content/drive/MyDrive/MLonMCU/Datasets/annotations/person_keypoints_val2014.json",
        "--images-folder", "/content/drive/MyDrive/MLonMCU/Datasets/val2014",
        "--checkpoint-path", drive_checkpoint,
        "--output-name", f"{drive_model_dir}/detections_val.json",
        # GPU by default; user can add --cpu to override
    ]


def main():
    import argparse
    
    # Build default args
    default_args = build_args()
    
    # Allow user CLI args to override defaults
    sys.argv = default_args[:1] + default_args[1:] + sys.argv[1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    parser.add_argument('--cpu', action='store_true', help='force CPU inference (slower)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    net = StudentModel2MB()
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    # Training scripts save with 'model' key
    if 'model' in checkpoint:
        net.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)

    evaluate_student(args.labels, args.output_name, args.images_folder, net, args.visualize, device)


if __name__ == "__main__":
    main()
