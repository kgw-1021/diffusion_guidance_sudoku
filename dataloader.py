import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. 데이터셋 로드
dataset = load_dataset("Ritvik19/Sudoku-Dataset", split="train[:1000000]") # 테스트용으로 5만개만

def preprocess(batch):
    # 데이터셋의 'solution' 컬럼(정답)을 사용해 Prior를 학습합니다.
    solutions = batch['solution'] 
    
    processed_images = []
    for sol in solutions:
        # 문자열을 9x9 정수 배열로 변환
        grid = np.array([int(c) for c in sol]).reshape(9, 9)
        # One-hot Encoding: (9, 9) -> (9, 9, 9)
        # 숫자 1~9를 인덱스 0~8로 매핑 (val - 1)
        one_hot = np.eye(9)[grid - 1] 
        
        # PyTorch Conv2d는 (Channel, Height, Width) 순서이므로 Transpose
        # 결과: (9, 9, 9)
        tensor = one_hot.transpose(2, 0, 1)
        processed_images.append(tensor)
    
    # [0, 1] 범위를 [-1, 1]로 스케일링 (Diffusion 표준)
    batch_tensor = torch.tensor(np.array(processed_images), dtype=torch.float32)
    batch_tensor = batch_tensor * 2.0 - 1.0 
    
    return {"pixel_values": batch_tensor}

# 맵핑 적용 및 로더 생성
dataset.set_transform(preprocess)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 확인
sample = next(iter(train_loader))
print("Input Shape:", sample['pixel_values'].shape) 
# 예상 출력: torch.Size([64, 9, 9, 9]) -> (Batch, Channel=9, H=9, W=9)