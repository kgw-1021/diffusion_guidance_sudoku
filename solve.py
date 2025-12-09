import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm 
from sudokuNet import SudokuNet
from collections import defaultdict
from utils import get_difficulty, calculate_metrics 

# ==========================================
# 1. 설정 및 도구 함수
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
GUIDANCE_SCALE = 20.0  # 이 값을 조절하며 실험해보세요 (10 ~ 100)
NUM_TESTS = 10        # 테스트할 문제 개수
NUM_LANGEVIN_STEPS = 20  # 스텝당 랑주뱅 반복 횟수
STEP_SIZE = 0.2  # Langevin step 크기

# 앞서 학습에 사용한 Beta Schedule 가져오기
T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

def str_to_tensor(grid_str):
    """
    스도쿠 문자열("0043...")을 입력받아:
    1. clues_onehot: (1, 9, 9, 9) - 힌트 위치의 One-hot 텐서
    2. mask: (1, 1, 9, 9) - 힌트가 있는 곳은 1, 빈칸은 0
    """
    grid = np.array([int(c) for c in grid_str]).reshape(9, 9)
    
    # 0은 빈칸이므로 마스크 생성
    mask_np = (grid != 0).astype(np.float32)
    
    # One-hot encoding (숫자 1->idx 0, ... 9->idx 8)
    # 빈칸(0)은 일단 0번 인덱스로 들어가지만 mask로 가려지므로 상관없음
    grid_indices = np.clip(grid - 1, 0, 8) 
    one_hot = np.eye(9)[grid_indices] # (9, 9, 9)
    
    # Transpose to (Channel, H, W) -> (1, 9, 9, 9)
    clues_tensor = torch.tensor(one_hot.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return clues_tensor.to(device), mask_tensor.to(device), grid

# ==========================================
# 2. 에너지 함수 (규칙 위반 정도 측정)
# ==========================================
def compute_energy(x, clues, mask):
    """
    Sum Constraint + Entropy + Orthogonality (Gram Matrix)
    """
    # 1. Softmax (Batch, Channel=9, H=9, W=9)
    probs = torch.softmax(x, dim=1) 
    b, c, h, w = probs.shape
    
    # -----------------------------------------------------
    # A. Sum Constraints (기존: 행/열/박스 합은 1이어야 함)
    # -----------------------------------------------------
    row_sum = probs.sum(dim=3) # (B, 9, 9)
    col_sum = probs.sum(dim=2) # (B, 9, 9)
    
    # Box Reshape: (B, 9, 9) where dim 1 is box_idx, dim 2 is cell_idx
    probs_boxes = probs.view(b, c, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5).reshape(b, c, 9, 9)
    box_sum = probs_boxes.sum(dim=3)
    
    loss_sum = torch.sum((row_sum - 1.0)**2) + \
               torch.sum((col_sum - 1.0)**2) + \
               torch.sum((box_sum - 1.0)**2)

    # -----------------------------------------------------
    # B. Orthogonality Constraints (신규 추가: 충돌 방지)
    # Target: A @ A.T should be Identity Matrix
    # -----------------------------------------------------
    I = torch.eye(9, device=x.device).unsqueeze(0) # (1, 9, 9) Identity Matrix
    
    # (1) Rows Orthogonality
    # 변환: (B, C, H, W) -> (B, H, W, C) -> (Batch*Row, Cell, Num)
    # 각 행(Row)에 대해 (9 Cells x 9 Numbers) 행렬을 만듦
    rows_mat = probs.permute(0, 2, 3, 1).reshape(-1, 9, 9)
    rows_gram = torch.matmul(rows_mat, rows_mat.transpose(1, 2)) # (B*9, 9, 9)
    # 모든 행의 Gram Matrix가 단위 행렬(I)과 같아야 함
    loss_ortho_row = torch.mean((rows_gram - I) ** 2)

    # (2) Cols Orthogonality
    # 변환: (B, C, H, W) -> (B, W, H, C) -> (Batch*Col, Cell, Num)
    cols_mat = probs.permute(0, 3, 2, 1).reshape(-1, 9, 9)
    cols_gram = torch.matmul(cols_mat, cols_mat.transpose(1, 2))
    loss_ortho_col = torch.mean((cols_gram - I) ** 2)

    # (3) Boxes Orthogonality
    # 변환: (B, C, Box, Cell) -> (B, Box, Cell, C) -> (Batch*Box, Cell, Num)
    # 앞서 만든 probs_boxes 사용 (B, C, 9, 9)
    boxes_mat = probs_boxes.permute(0, 2, 3, 1).reshape(-1, 9, 9)
    boxes_gram = torch.matmul(boxes_mat, boxes_mat.transpose(1, 2))
    loss_ortho_box = torch.mean((boxes_gram - I) ** 2)

    loss_ortho = loss_ortho_row + loss_ortho_col + loss_ortho_box

    # -----------------------------------------------------
    # C. Entropy (Sharpening)
    # -----------------------------------------------------
    epsilon = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + epsilon))
    
    # === 가중치 조절 (Hyperparameter Tuning) ===
    # Sum은 기본 골격, Ortho는 충돌 방지, Entropy는 마무리 결정
    
    # Ortho는 값이(9x9 sum) 커질 수 있으므로 scale을 좀 낮춰줌
    total_energy = loss_sum + (2.0 * loss_ortho) + (0.01 * entropy)
    
    return total_energy

# ==========================================
# 3. Solver (Guidance + Replacement)
# ==========================================
@torch.no_grad()
def solve_sudoku(model, clues, mask, 
                 guidance_scale=20.0,    # 기본 가이던스 강도      
                 langevin_steps=5,       # 한 스텝당 반복 수정 횟수 (중요!)
                 langevin_lr=0.2):       # 반복 수정 시 이동 보폭
    """
    Improved Solver with Langevin Dynamics & Strict Replacement
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. 초기 노이즈 설정
    x_t = torch.randn_like(clues).to(device)
    
    # 진행바 설정
    pbar = tqdm(reversed(range(T)), total=T, leave=False, desc="Solving Sudoku")
    
    for t in pbar:
        # -----------------------------------------------------------
        # A. Replacement (1차): 힌트 위치 강제 고정
        # -----------------------------------------------------------
        # 현재 시점 t에 맞는 노이즈가 섞인 정답(clues)을 생성
        alpha_bar = alphas_cumprod[t]
        noise = torch.randn_like(clues)
        
        # q_sample: sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*epsilon
        # clues는 One-hot(0 or 1)이지만, 모델 입력은 [-1, 1] 범위이므로 변환
        clues_input = clues * 2.0 - 1.0
        noisy_clues = torch.sqrt(alpha_bar) * clues_input + torch.sqrt(1 - alpha_bar) * noise
        
        # 마스크 위치만 덮어쓰기
        x_t = mask * noisy_clues + (1 - mask) * x_t

        # -----------------------------------------------------------
        # B. Langevin Dynamics (Inner Loop): 반복적 에너지 최소화
        # -----------------------------------------------------------
        # 어려운 문제일수록 여기서 여러 번 두들겨 맞춰야 함
        t_tensor = torch.tensor([t], device=device).long()
        
        for _ in range(langevin_steps):
            with torch.enable_grad():
                x_t = x_t.detach().requires_grad_(True)
                
                # (1) 모델 예측 (x_0 추정을 위해)
                # 여기서는 x_t 자체의 에너지를 줄이는 방향으로 심플하게 구현
                
                # (2) 에너지 계산 (업그레이드된 함수 사용 필수!)
                # 주의: compute_energy_final 내부에서 가중치를 조절하거나,
                # 여기서 ortho_weight를 인자로 넘겨줄 수 있게 함수를 수정하는 것이 좋음.
                # 여기서는 함수 내부 코드를 직접 쓰거나 수정한 함수를 호출한다고 가정.
                
                # *중요* 앞서 만든 compute_energy_final을 호출하되, 
                # 내부에서 loss_ortho에 5.0을 곱했다고 가정하거나 여기서 조절.
                loss = compute_energy(x_t, clues, mask) 
                
                # (3) 그라디언트 계산
                grad = torch.autograd.grad(loss, x_t)[0]
            
            # (4) Update: 에너지가 낮아지는 방향으로 조금 이동
            # scale * lr * grad
            x_t = x_t - (guidance_scale * langevin_lr * grad)
            
            # (5) Replacement (2차): 수정하다가 힌트가 망가지면 안 되므로 다시 고정
            x_t = mask * noisy_clues + (1 - mask) * x_t

        # -----------------------------------------------------------
        # C. Reverse Diffusion Step (Denoising)
        # -----------------------------------------------------------
        # Langevin으로 수정된 x_t를 바탕으로 다음 스텝(t-1)으로 이동
        with torch.no_grad():
            noise_pred = model(x_t, t_tensor)
            
            alpha = alphas[t]
            beta = betas[t]
            
            if t > 0:
                z = torch.randn_like(x_t)
            else:
                z = 0
            
            # DDPM Sampling Equation
            # mean = 1/sqrt(alpha) * (x_t - (beta / sqrt(1-alpha_bar)) * noise_pred)
            coeff1 = 1 / torch.sqrt(alpha)
            coeff2 = (1 - alpha) / (torch.sqrt(1 - alpha_bar))
            
            x_prev = coeff1 * (x_t - coeff2 * noise_pred) + torch.sqrt(beta) * z
            x_t = x_prev

    # -----------------------------------------------------------
    # 4. Final Post-Processing
    # -----------------------------------------------------------
    # 마지막으로 힌트 위치는 100% 원본 힌트로 덮어씌움 (노이즈 없는)
    # clues(One-hot)를 Logit 스케일(큰 양수/음수)로 변환해서 적용
    
    # 생성된 최종 로짓
    final_logits = x_t.clone()
    
    # 힌트가 있는 위치(mask=1)에는, 정답 채널에 아주 큰 값을, 오답 채널에 아주 작은 값을 줌
    # clues가 One-hot (B, 9, 9, 9)라고 가정
    large_val = 100.0
    hint_logits = clues * large_val + (1 - clues) * (-large_val)
    
    final_output = mask * hint_logits + (1 - mask) * final_logits
    
    # 확률 -> 숫자 변환
    pred_grid = torch.argmax(final_output, dim=1).cpu().numpy().squeeze() + 1
    
    return pred_grid

# ==========================================
# 4. 메인 테스트 루프
# ==========================================
# 데이터셋 로드 (Test split이 따로 없으면 train 뒷부분 사용)
dataset = load_dataset("Ritvik19/Sudoku-Dataset", split=f"train[:{NUM_TESTS}]")

# 모델 로드 (학습된 가중치)
model = SudokuNet().to(device)
try:
    model.load_state_dict(torch.load("sudoku_diffusion_best.pth"))
    print("모델 가중치 로드 성공!")
except:
    print("경고: 저장된 모델이 없습니다. 랜덤 가중치로 실행됩니다.")

results = defaultdict(lambda: {"total": 0, "success": 0, "blank_acc_sum": 0.0})

print(f"\n===== 테스트 시작 (총 {NUM_TESTS}개 문제) =====")

for i, item in enumerate(dataset):
    quiz_str = item['puzzle']
    sol_str = item['solution']
    


    diff_label, clue_count = get_difficulty(quiz_str)
    
    # 1. 전처리
    clues, mask, _ = str_to_tensor(quiz_str)
    
    # 2. 풀이 (Diffusion)
    pred_grid = solve_sudoku(model, clues, mask, guidance_scale=GUIDANCE_SCALE,
                             langevin_steps=NUM_LANGEVIN_STEPS, langevin_lr=STEP_SIZE)
    
    # 3. 정답 비교
    gt_grid = np.array([int(c) for c in sol_str]).reshape(9, 9)
    is_success, blank_acc = calculate_metrics(pred_grid, gt_grid, quiz_str)

    # 4. 결과 저장
    results[diff_label]["total"] += 1
    if is_success:
        results[diff_label]["success"] += 1
    results[diff_label]["blank_acc_sum"] += blank_acc
    
    # 전체 합계(Overall)도 저장
    results["Overall"]["total"] += 1
    if is_success:
        results["Overall"]["success"] += 1
    results["Overall"]["blank_acc_sum"] += blank_acc
    
    # 결과 출력 (처음 3개만 자세히)
    if i < 3:
        print(f"\n[Problem {i+1}]")
        print("Quiz:\n", np.array([int(c) for c in quiz_str]).reshape(9,9))
        print("Pred:\n", pred_grid)
        print("GT  :\n", gt_grid)

    print(f"Difficulty: {diff_label} | Clues: {clue_count} | Success: {is_success} | Blank Acc: {blank_acc*100:.2f}%")


print("\n" + "="*60)
print(f"{'Difficulty':<10} | {'Count':<5} | {'Success Rate':<12} | {'Blank Accuracy':<15}")
print("-" * 60)

# 난이도 순서대로 출력 (Easy -> Medium -> Hard -> Overall)
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