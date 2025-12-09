import numpy as np

def get_difficulty(quiz_str):
    """0이 아닌 숫자의 개수(힌트 수)로 난이도 판별"""
    clues = 81 - quiz_str.count('0')
    if clues >= 46:
        return "Easy", clues
    elif clues >= 31:
        return "Medium", clues
    else:
        return "Hard", clues

def calculate_metrics(pred, gt, quiz_str):
    """
    pred: (9,9) 예측 행렬
    gt: (9,9) 정답 행렬
    quiz_str: 문제 문자열 (빈칸 위치 확인용)
    """
    quiz_arr = np.array([int(c) for c in quiz_str]).reshape(9, 9)
    mask_blanks = (quiz_arr == 0) # 빈칸이었던 곳만 True
    
    # 1. 전체 성공 여부 (Perfect Match)
    is_success = np.array_equal(pred, gt)
    
    # 2. 빈칸 정확도 (Blank Accuracy)
    # 빈칸이었던 위치에서만 정답과 비교
    correct_blanks = np.sum((pred == gt) & mask_blanks)
    total_blanks = np.sum(mask_blanks)
    
    blank_acc = correct_blanks / total_blanks if total_blanks > 0 else 1.0
    
    return is_success, blank_acc