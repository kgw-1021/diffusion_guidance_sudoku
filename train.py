from torch.optim import Adam
import torch
import torch.nn.functional as F
from dataloader import train_loader
from sudokuNet import SudokuNet

# --- 설정 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Beta Schedule 정의 ---
T = 1000 # 총 Time step
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

def get_noisy_image(x_0, t):
    """x_0에 노이즈를 섞어서 x_t를 만듦"""
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    
    epsilon = torch.randn_like(x_0) # Gaussian Noise
    return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * epsilon, epsilon

# --- 학습 시작 ---
model = SudokuNet().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
epochs = 5 # 데이터가 많아서 5 에폭이면 충분할 듯

print("Start Training...")
model.train()

for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        x_0 = batch["pixel_values"].to(device) # (Batch, 9, 9, 9)
        n = x_0.shape[0]
        
        # 1. 랜덤한 시점 t 샘플링
        t = torch.randint(0, T, (n,), device=device).long()
        
        # 2. 노이즈 추가 (Forward Process)
        x_t, noise = get_noisy_image(x_0, t)
        
        # 3. 모델이 노이즈 예측
        noise_pred = model(x_t, t)
        
        # 4. Loss 계산 (MSE: 실제 노이즈 vs 예측 노이즈)
        loss = F.mse_loss(noise_pred, noise)
        
        # 5. 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.5f}")

# 모델 저장
torch.save(model.state_dict(), "sudoku_diffusion.pth")
print("Training Complete!")