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
NUM_TESTS = 100        
NUM_LANGEVIN_STEPS = 50  
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
def compute_energy(x, tau_, clues, mask):
    # 1. Softmax 
    probs = torch.softmax(x/tau_, dim=1)
    b, c, h, w = probs.shape
    
    # A. Sum Constraints
    row_sum = probs.sum(dim=3) 
    col_sum = probs.sum(dim=2) 
    probs_boxes = probs.view(b, c, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5).reshape(b, c, 9, 9)
    box_sum = probs_boxes.sum(dim=3)
    
    loss_sum = torch.sum((row_sum - 1.0)**2) + \
               torch.sum((col_sum - 1.0)**2) + \
               torch.sum((box_sum - 1.0)**2)

    # B. Orthogonality Constraints
    I = torch.eye(9, device=x.device).unsqueeze(0) 
    
    rows_mat = probs.permute(0, 2, 3, 1).reshape(-1, 9, 9)
    rows_gram = torch.matmul(rows_mat, rows_mat.transpose(1, 2)) 
    loss_ortho_row = torch.mean((rows_gram - I) ** 2)

    cols_mat = probs.permute(0, 3, 2, 1).reshape(-1, 9, 9)
    cols_gram = torch.matmul(cols_mat, cols_mat.transpose(1, 2))
    loss_ortho_col = torch.mean((cols_gram - I) ** 2)

    boxes_mat = probs_boxes.permute(0, 2, 3, 1).reshape(-1, 9, 9)
    boxes_gram = torch.matmul(boxes_mat, boxes_mat.transpose(1, 2))
    loss_ortho_box = torch.mean((boxes_gram - I) ** 2)

    loss_ortho = loss_ortho_row + loss_ortho_col + loss_ortho_box

    # C. Entropy
    epsilon = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + epsilon))
    
    # Weighting
    total_energy = loss_sum + (30.0 * loss_ortho) + (0.01 * entropy)
    
    return total_energy

# ==========================================
# 3. Solver (Debug 모드 추가됨)
# ==========================================
@torch.no_grad()
def solve_sudoku(model, clues, mask, 
                 guidance_scale=20.0,    
                 langevin_steps=50,       
                 langevin_lr=0.05, 
                 debug=False):

    model.eval()
    device = next(model.parameters()).device
    x_t = torch.randn_like(clues).to(device)
    
    # -----------------------------------------------------------
    # [추가] 최적의 해를 저장할 변수 초기화
    # -----------------------------------------------------------
    best_loss = float('inf')
    best_x = x_t.clone()
    
    if debug:
        history = {"energy_trace": [], "conflict_maps": []}

    pbar = tqdm(reversed(range(T)), total=T, leave=False, desc="Solving Sudoku")
    
    for t in pbar:
        # A. Start Point
        alpha_bar = alphas_cumprod[t]
        noise = torch.randn_like(clues)
        clues_input = clues * 2.0 - 1.0
        noisy_clues = torch.sqrt(alpha_bar) * clues_input + torch.sqrt(1 - alpha_bar) * noise
        
        # 초기화
        x_t = mask * noisy_clues + (1 - mask) * x_t

        # B. Langevin Dynamics (Inner Loop)
        for k in range(langevin_steps):
            with torch.enable_grad():
                x_t = x_t.detach().requires_grad_(True)
                
                loss_energy = compute_energy(x_t, tau_, clues, mask)
                loss_clue = torch.sum(mask * (x_t - noisy_clues)**2)
                loss = loss_energy + (100.0 * loss_clue) 
                
                current_loss_val = loss.item()
                if current_loss_val < best_loss:
                    best_loss = current_loss_val
                    best_x = x_t.detach().clone() # 현재 상태 복제 저장
                    # (옵션) 진행바에 현재 베스트 갱신 알림
                    # pbar.set_postfix({'MinLoss': f"{best_loss:.2f}"})

                if debug: history["energy_trace"].append(current_loss_val)
                grad = torch.autograd.grad(loss, x_t)[0]

            # 정석 랑주뱅 노이즈
            noise = torch   .randn_like(x_t)
            noise_scale = torch.sqrt(torch.tensor(2 * langevin_lr, device=device))
            if t < 200: noise_scale *= 0.1

            # Update
            x_t = x_t - (guidance_scale * langevin_lr * grad) + (noise_scale * noise)
            
            # Clipping
            x_t = torch.clamp(x_t, -1.0, 1.0)

            # Loop 내부 Hard Replacement
            x_t = mask * noisy_clues + (1 - mask) * x_t

        # C. Reverse Diffusion Step
        with torch.no_grad():
            t_tensor = torch.tensor([t], device=device).long()
            noise_pred = model(x_t, t_tensor)
            
            alpha = alphas[t]
            beta = betas[t]
            if t > 0: z = torch.randn_like(x_t)
            else: z = 0
            
            coeff1 = 1 / torch.sqrt(alpha)
            coeff2 = (1 - alpha) / (torch.sqrt(1 - alpha_bar))
            x_prev = coeff1 * (x_t - coeff2 * noise_pred) + torch.sqrt(beta) * z
            x_t = x_prev


    print(f"  >> Best Loss found during trajectory: {best_loss:.4f}")
    
    # 힌트 부분은 원본(clues)으로 완벽하게 교체 (노이즈 없는 원본)
    # best_x는 확률 분포 상태이므로, 힌트 부분만 강력하게 덮어씀
    final_logits = best_x.clone()
    
    large_val = 100.0
    hint_logits = clues * large_val + (1 - clues) * (-large_val)
    
    # 힌트 위치는 hint_logits, 빈칸 위치는 best_x의 logits 사용
    final_output = mask * hint_logits + (1 - mask) * final_logits
    
    pred_grid = torch.argmax(final_output, dim=1).cpu().numpy().squeeze() + 1

    if debug: 
        return pred_grid, history
    else: 
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
    
    output = solve_sudoku(
        model, clues, mask, 
        guidance_scale=GUIDANCE_SCALE,
        langevin_steps=NUM_LANGEVIN_STEPS, 
        langevin_lr=STEP_SIZE,
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