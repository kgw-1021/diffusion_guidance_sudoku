from torch.optim import Adam
import torch
import torch.nn.functional as F
from dataloader import train_loader, val_loader # dataloader 임포트
from sudokuNet import SudokuNet
import os

# --- 설정 (함수 밖이나 안이나 상관없지만, 전역 변수로 둡니다) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

def get_noisy_image(x_0, t):
    """x_0에 노이즈를 섞어서 x_t를 만듦"""
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    
    epsilon = torch.randn_like(x_0) 
    return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * epsilon, epsilon

def evaluate(model, loader):
    """검증 함수"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            x_0 = batch["pixel_values"].to(device)
            n = x_0.shape[0]
            t = torch.randint(0, T, (n,), device=device).long()
            x_t, noise = get_noisy_image(x_0, t)
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)

# =================================================================
# [중요] Windows에서는 실행 코드를 반드시 아래 조건문 안에 넣어야 합니다.
# =================================================================
if __name__ == '__main__':
    print("Using device:", device)
    
    # 모델 및 옵티마이저 초기화
    model = SudokuNet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 5
    best_val_loss = float('inf')

    print("Start Training...")
    
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            x_0 = batch["pixel_values"].to(device)
            n = x_0.shape[0]
            
            t = torch.randint(0, T, (n,), device=device).long()
            x_t, noise = get_noisy_image(x_0, t)
            
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 1000 == 0:
                print(f"Epoch {epoch} | Step {step} | Train Loss: {loss.item():.5f}")

        # Validation 및 저장
        val_loss = evaluate(model, val_loader)
        print(f">>> Epoch {epoch} Finished | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "sudoku_diffusion_best.pth")
            print(f"    [Saved Best Model] New Best Loss: {best_val_loss:.5f}")
        
        torch.save(model.state_dict(), "sudoku_diffusion_last.pth")

    print("Training Complete!")