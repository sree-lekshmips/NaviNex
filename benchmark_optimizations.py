# benchmark_optimizations.py
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from config import CUSTOM_YOLO, COCO_YOLO, SEGFORMER_MODEL_DIR

def test_optimizations():
    # Set PyTorch CPU threads for maximum multicore efficiency
    torch.set_num_threads(6)
    
    dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    print("Testing YOLO at imgsz=320...")
    yolo = YOLO(str(COCO_YOLO))
    _ = yolo(dummy, imgsz=320, verbose=False)
    
    t0 = time.perf_counter()
    for _ in range(5):
        _ = yolo(dummy, imgsz=320, verbose=False)
    t_yolo320 = (time.perf_counter() - t0) / 5.0 * 1000
    print(f"YOLO (imgsz=320): {t_yolo320:.1f} ms")
    
    print("\nTesting MiDaS optimizations...")
    midas_repo = Path.home() / ".cache" / "torch" / "hub" / "intel-isl_MiDaS_master"
    midas_model = torch.hub.load(str(midas_repo), "DPT_SwinV2_T_256", source="local")
    midas_model.eval()
    midas_transforms = torch.hub.load(str(midas_repo), "transforms", source="local")
    transform = midas_transforms.swin256_transform
    
    rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb)
    
    # Test without heavy bicubic 720x1280 tensor interpolation
    with torch.inference_mode():
        _ = midas_model(input_tensor)
        
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.inference_mode():
            pred = midas_model(input_tensor)
            pred_np = pred.squeeze().cpu().numpy()
            # Fast cv2 resize instead of slow torch bicubic interpolate
            pred_resized = cv2.resize(pred_np, (640, 360), interpolation=cv2.INTER_LINEAR)
            depth = cv2.normalize(pred_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t_midas_fast = (time.perf_counter() - t0) / 5.0 * 1000
    print(f"MiDaS (inference_mode + cv2 resize): {t_midas_fast:.1f} ms")
    
    print("\nTesting SegFormer optimizations...")
    seg_proc = SegformerImageProcessor.from_pretrained(str(SEGFORMER_MODEL_DIR))
    seg_model = SegformerForSemanticSegmentation.from_pretrained(str(SEGFORMER_MODEL_DIR))
    seg_model.eval()
    
    square_frame = cv2.resize(dummy, (256, 256), interpolation=cv2.INTER_LINEAR)
    rgb_seg = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
    inputs = seg_proc(images=rgb_seg, return_tensors="pt")
    
    with torch.inference_mode():
        _ = seg_model(**inputs)
        
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.inference_mode():
            outputs = seg_model(**inputs)
            logits = outputs.logits
            small_seg = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
    t_seg_fast = (time.perf_counter() - t0) / 5.0 * 1000
    print(f"SegFormer (inference_mode): {t_seg_fast:.1f} ms")

if __name__ == "__main__":
    test_optimizations()
