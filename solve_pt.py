import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm 
from sudokuNet import SudokuNet
from collections import defaultdict
from utils import get_difficulty, calculate_metrics 
import matplotlib.pyplot as plt 
import seaborn as sns 

# ==========================================
# 1. 설정 및 도구 함수
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

GUIDANCE_SCALE = 20.0 
NUM_TESTS = 10        
NUM_LANGEVIN_STEPS = 20  
STEP_SIZE = 0.1  

# Beta Schedule
T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

def str_to_tensor(grid_str):
    grid = np.array([int(c) for c in grid_str]).reshape(9, 9)
    mask_np = (grid != 0).astype(np.float32)
    grid_indices = np.clip(grid - 1, 0, 8) 
    one_hot = np.eye(9)[grid_indices] 
    clues_tensor = torch.tensor(one_hot.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return clues_tensor.to(device), mask_tensor.to(device), grid

# ==========================================
# 2. 에너지 함수
# ==========================================
def compute_energy__batched(x, clues, mask):
    """
    기존 compute_energy와 로직은 같으나, 
    모든 배치를 합치지 않고 (Batch,) 형태의 텐서를 반환합니다.
    """
    probs = torch.softmax(x, dim=1)
    b, c, h, w = probs.shape
    
    # A. Sum Constraints
    row_sum = probs.sum(dim=3)
    col_sum = probs.sum(dim=2)
    probs_boxes = probs.view(b, c, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5).reshape(b, c, 9, 9)
    box_sum = probs_boxes.sum(dim=3)
    
    # (Batch, ...) -> (Batch,) 로 차원 축소하되 sum은 하지 않음 (dim=(1,2) 등 활용)
    loss_sum = torch.sum((row_sum - 1.0)**2, dim=(1, 2)) + \
               torch.sum((col_sum - 1.0)**2, dim=(1, 2)) + \
               torch.sum((box_sum - 1.0)**2, dim=(1, 2))

    # B. Orthogonality Constraints
    I = torch.eye(9, device=x.device).unsqueeze(0)
    
    rows_mat = probs.permute(0, 2, 3, 1).reshape(b, 9, 9, 9) # (B, Row, 9, 9)
    # Batch Matmul을 위해 차원 정리
    # (B, 9, 9, 9) -> (B*9, 9, 9) 로 펼쳐서 계산 후 다시 합침
    rows_gram = torch.matmul(rows_mat.reshape(-1, 9, 9), rows_mat.reshape(-1, 9, 9).transpose(1, 2))
    rows_gram = rows_gram.view(b, 9, 9, 9)
    loss_ortho_row = torch.mean((rows_gram - I.unsqueeze(1)) ** 2, dim=(1, 2, 3))

    cols_mat = probs.permute(0, 3, 2, 1).reshape(b, 9, 9, 9)
    cols_gram = torch.matmul(cols_mat.reshape(-1, 9, 9), cols_mat.reshape(-1, 9, 9).transpose(1, 2))
    cols_gram = cols_gram.view(b, 9, 9, 9)
    loss_ortho_col = torch.mean((cols_gram - I.unsqueeze(1)) ** 2, dim=(1, 2, 3))

    boxes_mat = probs_boxes.permute(0, 2, 3, 1).reshape(b, 9, 9, 9)
    boxes_gram = torch.matmul(boxes_mat.reshape(-1, 9, 9), boxes_mat.reshape(-1, 9, 9).transpose(1, 2))
    boxes_gram = boxes_gram.view(b, 9, 9, 9)
    loss_ortho_box = torch.mean((boxes_gram - I.unsqueeze(1)) ** 2, dim=(1, 2, 3))
    
    loss_ortho = loss_ortho_row + loss_ortho_col + loss_ortho_box

    # C. Entropy
    epsilon = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=(1, 2, 3))
    
    
    total_energy = loss_sum + (30.0 * loss_ortho) + (0.01 * entropy)
    
    return total_energy # Shape: (Batch_Size,)

# ==========================================
# 3. Solver (Debug 모드 추가됨)
# ==========================================
@torch.no_grad()
def solve_sudoku_pt(model, clues, mask, 
                    num_replicas=16,     # 동시에 돌릴 체인 개수 (많을수록 좋음)
                    min_temp=0.8,        # 최저 온도 (정밀 탐색)
                    max_temp=5.0,       # 최고 온도 (활발한 탈출)
                    guidance_scale=20.0,
                    langevin_steps=50, 
                    debug=False):
    
    model.eval()
    device = next(model.parameters()).device
    
    # 1. 온도 사다리 생성 (Geometric Spacing)
    # T_1=1, ..., T_N=10
    temps = torch.logspace(np.log10(min_temp), np.log10(max_temp), num_replicas).to(device)
    inverse_temps = 1.0 / temps # Beta
    
    # 2. 데이터 확장 (Batch Expansion)
    # (1, 9, 9, 9) -> (num_replicas, 9, 9, 9)
    clues_batched = clues.repeat(num_replicas, 1, 1, 1)
    mask_batched = mask.repeat(num_replicas, 1, 1, 1)
    x_t = torch.randn_like(clues_batched).to(device)
    
    # Best Solution Tracking
    best_loss = float('inf')
    best_x = x_t[0].clone().unsqueeze(0)
    
    pbar = tqdm(reversed(range(T)), total=T, leave=False, desc="Parallel Tempering")
    
    for t in pbar:
        # A. Start Point
        alpha_bar = alphas_cumprod[t]
        noise = torch.randn_like(clues_batched)
        clues_input = clues_batched * 2.0 - 1.0
        noisy_target = torch.sqrt(alpha_bar) * clues_input + torch.sqrt(1 - alpha_bar) * noise
        
        x_t = mask_batched * noisy_target + (1 - mask_batched) * x_t

        # B. Population-based Langevin Loop
        for k in range(langevin_steps):
            with torch.enable_grad():
                x_t = x_t.detach().requires_grad_(True)
                
                # (1) 배치별 에너지 계산
                energies = compute_energy__batched(x_t, clues, mask)
                
                # (2) Clue Loss 추가 (개별 계산)
                clue_losses = torch.sum(mask_batched * (x_t - noisy_target)**2, dim=(1, 2, 3))
                total_losses = energies + (100.0 * clue_losses)
                
                # Best Tracking
                min_loss_batch, min_idx = torch.min(total_losses, dim=0)
                if min_loss_batch.item() < best_loss:
                    best_loss = min_loss_batch.item()
                    best_x = x_t[min_idx].detach().clone().unsqueeze(0)
                
                # (3) Gradient 계산 (sum()을 해서 backward하면 각각의 grad가 계산됨)
                loss_sum = total_losses.sum()
                grad = torch.autograd.grad(loss_sum, x_t)[0]

            # (4) Gradient Normalization & Noise Injection (Temperature 적용)
            grad_flat = grad.reshape(num_replicas, -1)
            grad_norm = torch.norm(grad_flat, dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            normalized_grad = grad / (grad_norm + 1e-8)
            
            # Step Size는 온도에 비례하지 않음 (이동 거리는 일정하게)
            # 다만, Noise 크기가 온도에 비례함!
            current_sigma = torch.sqrt(1 - alphas_cumprod[t])
            base_step_size = (current_sigma ** 2) * 0.1 # 기본 보폭
            
            # Update Rule:
            # x_new = x - step*grad + sqrt(2 * step * Temp) * noise
            # 온도가 높을수록(Temp 큼) 노이즈가 커짐 -> 탈출 용이
            
            # temps: (K,) -> (K, 1, 1, 1) broadcasting
            temp_view = temps.view(-1, 1, 1, 1)
            noise_scale = torch.sqrt(2 * base_step_size * temp_view)
            
            langevin_noise = torch.randn_like(x_t) * noise_scale
            
            x_t = x_t - (guidance_scale * base_step_size * normalized_grad) + langevin_noise
            x_t = torch.clamp(x_t, -1.0, 1.0)
            
            # (5) Parallel Replica Exchange (Vectorized Swap)
            # 파이썬 루프 없이 텐서 슬라이싱으로 처리
            # Swap은 x_t(상태)를 바꾸는 것임. (온도는 고정)
            
            if k % 5 == 0: 
                num_swap_passes = num_replicas // 2 
                
                for _ in range(num_swap_passes):
                    E = total_losses
                    beta = inverse_temps
                    
                    # --- Phase 1: 짝수 인덱스 스왑 (0<->1, 2<->3, ...) ---
                    # 왼쪽: 0, 2, ... (마지막이 홀수면 포함 안 됨)
                    # 오른쪽: 1, 3, ...
                    
                    # 슬라이싱 범위: 0부터 N-1까지 (오른쪽 짝꿍 확보를 위해)
                    E_even = E[0 : num_replicas-1 : 2]
                    E_odd_1 = E[1 : num_replicas : 2]
                    
                    beta_even = beta[0 : num_replicas-1 : 2]
                    beta_odd_1 = beta[1 : num_replicas : 2]
                    
                    # 개수 확인 (안전장치)
                    min_len = min(len(E_even), len(E_odd_1))
                    E_even = E_even[:min_len]
                    E_odd_1 = E_odd_1[:min_len]
                    beta_even = beta_even[:min_len]
                    beta_odd_1 = beta_odd_1[:min_len]
                    
                    # Metropolis Criterion
                    delta = (E_even - E_odd_1) * (beta_even - beta_odd_1)
                    prob = torch.exp(delta)
                    swap_mask = torch.rand(min_len, device=device) < prob
                    
                    # 인덱스 생성
                    base_indices = torch.arange(0, 2 * min_len, 2, device=device)
                    swap_indices_even = base_indices[swap_mask]
                    swap_indices_odd = swap_indices_even + 1
                    
                    if len(swap_indices_even) > 0:
                        temp_x = x_t[swap_indices_even].clone()
                        x_t[swap_indices_even] = x_t[swap_indices_odd]
                        x_t[swap_indices_odd] = temp_x
                        
                        # 에너지 값도 스왑 (재계산 방지)
                        temp_E = total_losses[swap_indices_even].clone()
                        total_losses[swap_indices_even] = total_losses[swap_indices_odd]
                        total_losses[swap_indices_odd] = temp_E

                    # --- Phase 2: 홀수 인덱스 스왑 (1<->2, 3<->4, ...) ---
                    # [수정된 부분] 슬라이싱 범위 수정으로 에러 해결
                    # 왼쪽: 1, 3, ... (마지막 인덱스 제외)
                    # 오른쪽: 2, 4, ...
                    
                    E_odd = E[1 : num_replicas-1 : 2]
                    E_even_2 = E[2 : num_replicas : 2]
                    
                    beta_odd = beta[1 : num_replicas-1 : 2]
                    beta_even_2 = beta[2 : num_replicas : 2]
                    
                    min_len_2 = min(len(E_odd), len(E_even_2))
                    
                    if min_len_2 > 0:
                        E_odd = E_odd[:min_len_2]
                        E_even_2 = E_even_2[:min_len_2]
                        beta_odd = beta_odd[:min_len_2]
                        beta_even_2 = beta_even_2[:min_len_2]
                        
                        delta2 = (E_odd - E_even_2) * (beta_odd - beta_even_2)
                        prob2 = torch.exp(delta2)
                        swap_mask2 = torch.rand(min_len_2, device=device) < prob2
                        
                        # 인덱스 생성 (1부터 시작)
                        base_indices_2 = torch.arange(1, 1 + 2 * min_len_2, 2, device=device)
                        swap_idx_odd = base_indices_2[swap_mask2]
                        swap_idx_even = swap_idx_odd + 1
                        
                        if len(swap_idx_odd) > 0:
                            temp2 = x_t[swap_idx_odd].clone()
                            x_t[swap_idx_odd] = x_t[swap_idx_even]
                            x_t[swap_idx_even] = temp2
                            
                            temp_E2 = total_losses[swap_idx_odd].clone()
                            total_losses[swap_idx_odd] = total_losses[swap_idx_even]
                            total_losses[swap_idx_even] = temp_E2

            # Loop 내 Hard Replacement
            x_t = mask_batched * noisy_target + (1 - mask_batched) * x_t

        # C. Reverse Step (Batch 처리)
        with torch.no_grad():
            t_tensor = torch.tensor([t], device=device).long().repeat(num_replicas)
            noise_pred = model(x_t, t_tensor)
            
            alpha = alphas[t]
            beta = betas[t]
            z = torch.randn_like(x_t) if t > 0 else 0
            
            coeff1 = 1 / torch.sqrt(alpha)
            coeff2 = (1 - alpha) / (torch.sqrt(1 - alpha_bar))
            x_prev = coeff1 * (x_t - coeff2 * noise_pred) + torch.sqrt(beta) * z
            x_t = x_prev

    print(f"  >> Best Loss (PT): {best_loss:.4f}")
    
    # 최종 반환: 가장 에너지가 낮았던(Best) 샘플 하나만 반환
    final_logits = best_x.clone() # (1, 9, 9, 9)
    
    large_val = 100.0
    hint_logits = clues * large_val + (1 - clues) * (-large_val)
    
    final_output = mask * hint_logits + (1 - mask) * final_logits
    pred_grid = torch.argmax(final_output, dim=1).cpu().numpy().squeeze() + 1
    
    return pred_grid

# ==========================================
# 4. 시각화 함수 (추가됨)
# ==========================================
def visualize_analysis(history):
    # 1. 에너지 궤적 그리기
    plt.figure(figsize=(12, 5))
    plt.plot(history["energy_trace"])
    plt.title("Energy Trajectory (Langevin Dynamics)")
    plt.xlabel("Total Optimization Steps (Time * Langevin)")
    plt.ylabel("Constraint Energy")
    plt.yscale("log") # 로그 스케일로 보면 급격한 감소가 더 잘 보임
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

    # 2. Conflict Map (Heatmap) 그리기
    snapshots = history["conflict_maps"]
    # 너무 많으면 5개만 추림
    if len(snapshots) > 5:
        indices = np.linspace(0, len(snapshots)-1, 5, dtype=int)
        snapshots = [snapshots[i] for i in indices]
    
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    if len(snapshots) == 1: axes = [axes]
    
    for ax, (t, g_map) in zip(axes, snapshots):
        sns.heatmap(g_map, ax=ax, cmap="Reds", cbar=False, square=True)
        ax.set_title(f"Conflict Map at t={t}")
        ax.axis("off")
        
        # 3x3 격자 그리기 (시각적 도움)
        for x in range(1, 9):
            lw = 2 if x % 3 == 0 else 0.5
            ax.axvline(x, color='black', linewidth=lw)
            ax.axhline(x, color='black', linewidth=lw)

    plt.suptitle("Evolution of Logical Conflicts (Red = High Gradient/Energy)", fontsize=16)
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 메인 테스트 루프
# ==========================================
dataset = load_dataset("Ritvik19/Sudoku-Dataset", split=f"train[:{NUM_TESTS}]")

model = SudokuNet().to(device)
try:
    model.load_state_dict(torch.load("sudoku_diffusion_best.pth"))
    print("Model Loaded Successfully.")
except:
    print("Warning: No saved model found. Running with random weights.")

results = defaultdict(lambda: {"total": 0, "success": 0, "blank_acc_sum": 0.0})

print(f"\n===== 테스트 시작 (총 {NUM_TESTS}개 문제) =====")

for i, item in enumerate(dataset):
    quiz_str = item['puzzle']
    sol_str = item['solution']
    
    diff_label, clue_count = get_difficulty(quiz_str)
    
    clues, mask, _ = str_to_tensor(quiz_str)
    
    is_debug = False
    
    output = solve_sudoku_pt(
        model, clues, mask, 
        guidance_scale=GUIDANCE_SCALE,
        langevin_steps=NUM_LANGEVIN_STEPS, 
        debug=is_debug # 디버그 플래그 전달
    )
    
    if is_debug:
        pred_grid, history = output
        print("\n[Analysis] Visualizing solving process for the first problem...")
        visualize_analysis(history) # 시각화 실행
    else:
        pred_grid = output
    
    # 3. 정답 비교 및 저장 (기존과 동일)
    gt_grid = np.array([int(c) for c in sol_str]).reshape(9, 9)
    is_success, blank_acc = calculate_metrics(pred_grid, gt_grid, quiz_str)

    results[diff_label]["total"] += 1
    if is_success:
        results[diff_label]["success"] += 1
    results[diff_label]["blank_acc_sum"] += blank_acc
    
    results["Overall"]["total"] += 1
    if is_success:
        results["Overall"]["success"] += 1
    results["Overall"]["blank_acc_sum"] += blank_acc
    
    if 10 < i and  i < 13:
        print(f"\n[Problem {i+1}]")
        print("Quiz:\n", np.array([int(c) for c in quiz_str]).reshape(9,9))
        print("Pred:\n", pred_grid)
        print("GT  :\n", gt_grid)

    print(f"Difficulty: {diff_label} | Clues: {clue_count} | Success: {is_success} | Blank Acc: {blank_acc*100:.2f}%")


print("\n" + "="*60)
print(f"{'Difficulty':<10} | {'Count':<5} | {'Success Rate':<12} | {'Blank Accuracy':<15}")
print("-" * 60)

order = ["Easy", "Medium", "Hard", "Overall"]

for label in order:
    if label not in results: continue
    
    data = results[label]
    count = data["total"]
    if count == 0: continue
    
    success_rate = (data["success"] / count) * 100
    avg_blank_acc = (data["blank_acc_sum"] / count) * 100
    
    print(f"{label:<10} | {count:<5} | {success_rate:>11.2f}% | {avg_blank_acc:>14.2f}%")

print("="*60 + "\n")
print(f"parameters: GUIDANCE_SCALE={GUIDANCE_SCALE}, NUM_LANGEVIN_STEPS={NUM_LANGEVIN_STEPS}, STEP_SIZE={STEP_SIZE}\n")