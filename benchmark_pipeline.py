# benchmark_pipeline.py
import time
import cv2
import numpy as np
import torch
from src.detection import CustomYOLODetector, CocoYOLODetector
from src.depth.midas import MiDaSDepthEstimator
from src.segmentation import SegFormerSegmenter

def profile():
    dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    print("Loading models...")
    custom_yolo = CustomYOLODetector()
    coco_yolo = CocoYOLODetector()
    depth_est = MiDaSDepthEstimator()
    seg = SegFormerSegmenter()
    
    print("\nWarming up...")
    _ = custom_yolo.predict(dummy)
    _ = coco_yolo.predict(dummy)
    _ = depth_est.predict(dummy)
    _ = seg.predict(dummy)
    
    print("\nProfiling 5 iterations each:")
    
    # Profile Custom YOLO
    t0 = time.perf_counter()
    for _ in range(5):
        _ = custom_yolo.predict(dummy)
    t_custom = (time.perf_counter() - t0) / 5.0 * 1000
    print(f"Custom YOLO:   {t_custom:.1f} ms / frame")
    
    # Profile COCO YOLO
    t0 = time.perf_counter()
    for _ in range(5):
        _ = coco_yolo.predict(dummy)
    t_coco = (time.perf_counter() - t0) / 5.0 * 1000
    print(f"COCO YOLO:     {t_coco:.1f} ms / frame")
    
    # Profile MiDaS Depth
    t0 = time.perf_counter()
    for _ in range(5):
        _ = depth_est.predict(dummy)
    t_depth = (time.perf_counter() - t0) / 5.0 * 1000
    print(f"MiDaS Depth:   {t_depth:.1f} ms / frame")
    
    # Profile SegFormer
    t0 = time.perf_counter()
    for _ in range(5):
        _ = seg.predict(dummy)
    t_seg = (time.perf_counter() - t0) / 5.0 * 1000
    print(f"SegFormer:     {t_seg:.1f} ms / frame")
    
    total_per_frame = t_custom + t_coco + t_depth + t_seg
    print(f"\nTotal serial CPU latency if all run on 1 frame: {total_per_frame:.1f} ms (~{1000/total_per_frame:.1f} FPS)")

if __name__ == "__main__":
    profile()
