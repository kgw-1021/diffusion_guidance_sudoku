import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. 데이터셋 로드
# 전체 데이터 중 100만 개만 가져온 뒤 분할하거나, 전체를 가져와도 됩니다.
full_dataset = load_dataset("Ritvik19/Sudoku-Dataset", split="train[:1000000]")

# 2. Train / Validation 분할 (예: 90% 학습, 10% 검증)
# seed=42로 고정하여 언제 실행해도 같은 데이터가 검증셋으로 가도록 합니다.
split_data = full_dataset.train_test_split(test_size=0.1, seed=42)

train_ds = split_data["train"]
val_ds = split_data["test"]

print(f"Train set size: {len(train_ds)}") # 900,000 예상
print(f"Val set size:   {len(val_ds)}")   # 100,000 예상

def preprocess(batch):
    """
    기존 전처리 로직 유지
    """
    solutions = batch['solution'] 
    
    processed_images = []
    for sol in solutions:
        # 문자열 -> 9x9 배열 변환
        grid = np.array([int(c) for c in sol]).reshape(9, 9)
        
        # One-hot Encoding: (9, 9) -> (9, 9, 9)
        # 1~9 값을 0~8 인덱스로 변환
        one_hot = np.eye(9)[grid - 1] 
        
        # (H, W, C) -> (C, H, W) 로 Transpose
        tensor = one_hot.transpose(2, 0, 1)
        processed_images.append(tensor)
    
    # Numpy -> Tensor 변환 및 정규화 [0, 1] -> [-1, 1]
    batch_tensor = torch.tensor(np.array(processed_images), dtype=torch.float32)
    batch_tensor = batch_tensor * 2.0 - 1.0 
    
    return {"pixel_values": batch_tensor}

# 3. 전처리 매핑 (Train, Val 둘 다 적용)
train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)

# 4. DataLoader 생성
# 학습용은 섞어줍니다 (Shuffle=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

# 검증용은 섞을 필요가 없습니다 (Shuffle=False) -> 순서대로 평가
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

# --- 테스트 코드 (실행 시 주석 해제하여 확인 가능) ---
# if __name__ == "__main__":
#     print("Checking Train Loader...")
#     sample = next(iter(train_loader))
#     print("Train Input Shape:", sample['pixel_values'].shape)
    
#     print("Checking Val Loader...")
#     val_sample = next(iter(val_loader))
#     print("Val Input Shape:", val_sample['pixel_values'].shape)