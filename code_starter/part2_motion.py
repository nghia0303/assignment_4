import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from part1_codec import shannon_entropy


def load_image_grayscale(filepath):
    """
        Loads an image and converts it to a grayscale numpy array.
    """
    img = Image.open(filepath).convert('L')
    return np.array(img, dtype=np.float32)


def visualize_results(curr_img, predicted_img, residual, mvs, search_range):
    """
        Visualizes the performance of the Inter-frame Motion Compensation.
    
        Args:
            curr_img (np.ndarray): The original target image (Ground Truth).
            predicted_img (np.ndarray): The motion-compensated prediction image.
            residual (np.ndarray): The pixel-wise difference between current and predicted.
            mvs (np.ndarray): The motion vector array of shape (rows, cols, 2).
            search_range (int): The search window parameter 'p' used (for display purposes).
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Prediction
    ax[0].imshow(predicted_img, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title(f"Predicted Frame (p={search_range})")
    ax[0].axis('off')
    
    # 2. Residual
    bpp = shannon_entropy(residual.flatten())
    ax[1].imshow(residual, cmap='gray', vmin=-50, vmax=50)
    ax[1].set_title(f"Residual\nEntropy: {bpp:.3f} bpp")
    ax[1].axis('off')

    # 3. Vectors
    ax[2].imshow(curr_img, cmap='gray', alpha=0.5)
    h, w, _ = mvs.shape
    Y, X = np.mgrid[0:h, 0:w] * 16 + 8
    dy, dx = mvs[:, :, 0], mvs[:, :, 1]
    
    ax[2].quiver(X, Y, dx, dy, color='red', angles='xy', scale_units='xy', scale=0.4, width=0.002)
    ax[2].set_title("Motion Vectors")
    ax[2].axis('off')
    # ax[2].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# =========================================================
# TO DO METHODS
# =========================================================
def motion_estimation(current_frame, ref_frame, block_size=16, search_range=16):
    """
    Step 1: Motion Estimation (Search).
    Identifies the best matching block in ref_frame for every block in current_frame.

    Args:
        current_frame: Target image (I_curr).
        ref_frame: Reference image (I_ref).
        block_size: Macroblock size (default 16).
        search_range: Window size 'p' (+/- p).

    Returns:
        motion_vectors: 3D array (rows, cols, 2) containing (dy, dx) for each block.
    """
    h, w = current_frame.shape
    mb_rows = h // block_size
    mb_cols = w // block_size
    
    motion_vectors = np.zeros((mb_rows, mb_cols, 2), dtype=int)
    
    for r in range(mb_rows):
        for c in range(mb_cols):
            # TODO: Implement Block Matching
            # 1. Define current block and search window (handle boundaries).
            y_curr = r * block_size
            x_curr = c * block_size
            current_block = current_frame[y_curr:y_curr+block_size, x_curr:x_curr+block_size]

            # Xác định vùng tìm kiếm:
            y_min = max(0, y_curr - search_range)
            y_max = min(h - block_size, y_curr + search_range)
            x_min = max(0, x_curr - search_range)
            x_max = min(w - block_size, x_curr + search_range)

            best_sad = float('inf')
            best_dy, best_dx = 0, 0
            # 2. Iterate through all candidates in window (Full Search).
            for y_ref in range(y_min, y_max + 1):
                for x_ref in range(x_min, x_max + 1):
                    candidate_block = ref_frame[y_ref:y_ref + block_size, x_ref:x_ref + block_size]
                    sad = np.sum(np.abs(current_block - candidate_block))

                    # 3. Find candidate with min SAD.
                    if sad < best_sad:
                        best_sad = sad
                        best_dy = y_ref - y_curr
                        best_dx = x_ref - x_curr



            # 4. Store displacement (dy, dx) in motion_vectors[r, c]
            motion_vectors[r, c] = [best_dy, best_dx]
    return motion_vectors

def motion_compensation(ref_frame, motion_vectors, block_size=16):
    """
    Step 2: Motion Compensation (Predict).
    Reconstructs the predicted image using ONLY the reference frame and vectors.

    Args:
        ref_frame: Reference image (I_ref).
        motion_vectors: (dy, dx) displacements from Step 1.
    
    Returns:
        predicted_frame: The Motion Compensated image (P).
    """
    h, w = ref_frame.shape
    mb_rows, mb_cols, _ = motion_vectors.shape
    predicted_frame = np.zeros_like(ref_frame)
    
    for r in range(mb_rows):
        for c in range(mb_cols):
            # TODO: Implement Reconstruction
            # 1. Retrieve (dy, dx) for this block.
            dy, dx = motion_vectors[r, c]

            y_curr = r * block_size
            x_curr = c * block_size

            y_ref = y_curr + dy
            x_ref = x_curr + dx

            # 2. Copy the matching block from ref_frame to predicted_frame.
            predicted_frame[y_curr:y_curr + block_size, x_curr:x_curr + block_size] = \
                ref_frame[y_ref:y_ref + block_size, x_ref:x_ref + block_size]

    return predicted_frame

def calc_residual(current_frame, ref_frame, block_size=16, search_range=16):
    """
    Step 3: Integration (The Codec).
    Combines Estimation and Compensation to generate the Residual.

    Returns:
        motion_vectors: The MVs used.
        predicted_frame: The predicted image (P).
        residual: The difference image (R = I_curr - P).
    """
    # TODO: Implement the full pipeline
    # 1. search to get motion vector
    motion_vectors = motion_estimation(current_frame, ref_frame, block_size, search_range)
    # 2. predict
    predict_frame = motion_compensation(ref_frame, motion_vectors, block_size)
    # 3. Calculate residual
    residual = current_frame - predict_frame

    return motion_vectors, predict_frame, residual


def calculate_bpp(residuals):
    """
    Hàm phụ trợ: Tính Entropy (Bitrate ước tính)
    Viết lại một chút so với Part 1 để không bị giống code cũ.
    """
    data = residuals.flatten()
    if len(data) == 0: return 0

    # Dùng numpy để đếm tần suất
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)

    # Tính entropy: -sum(p * log2(p))
    entropy = (-1) * np.sum(probabilities * np.log2(probabilities))
    return entropy


def question_3(curr, ref):
    """
    Trả lời toàn bộ yêu cầu phân tích:
    1. So sánh năng lượng (Energy Comparison).
    2. Phân tích đánh đổi Thời gian vs Hiệu quả nén (Trade-off Analysis).
    """
    print("\n" + "=" * 50)
    print("      REPORT ANALYSIS (QUESTION 3)      ")
    print("=" * 50)

    # --- PHẦN 1: SO SÁNH NĂNG LƯỢNG (Với p=16 tiêu chuẩn) ---
    print("\n[Part 1] Energy Analysis (p=16):")
    _, _, res_standard = calc_residual(curr, ref, block_size=16, search_range=16)

    # Tính năng lượng
    diff_no_mc = curr - ref
    energy_no_mc = np.sum(diff_no_mc ** 2)
    energy_with_mc = np.sum(res_standard ** 2)

    print(f"  - Energy WITHOUT Motion Comp: {energy_no_mc:15,.0f}")
    print(f"  - Energy WITH Motion Comp:    {energy_with_mc:15,.0f}")

    if energy_with_mc > 0:
        ratio = energy_no_mc / energy_with_mc
        print(f"  => Improvement Ratio: {ratio:.2f}x better compression potential.")

    # --- PHẦN 2: THÍ NGHIỆM ĐÁNH ĐỔI (TRADE-OFF) ---
    print("\n[Part 2] Performance Trade-off Experiment:")
    print(f"{'Range (p)':<10} | {'Time (s)':<10} | {'Bitrate (bpp)':<15}")
    print("-" * 40)

    p_values = [4, 8, 16, 32]  # Có thể thêm 64 nếu máy khỏe
    stats_time = []
    stats_bpp = []

    for p in p_values:
        # Bắt đầu đo giờ bằng perf_counter (chính xác hơn time.time)
        t_start = time.perf_counter()
        _, _, res = calc_residual(curr, ref, search_range=p)
        t_end = time.perf_counter()
        duration = t_end - t_start
        # Tính entropy
        bpp = calculate_bpp(res)
        stats_time.append(duration)
        stats_bpp.append(bpp)

        print(f"{p:<10} | {duration:<10.4f} | {bpp:<15.4f}")

    # --- PHẦN 3: VẼ BIỂU ĐỒ ---
    print("\n[Info] Plotting results...")
    plt.style.use('bmh')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_t = 'tab:purple'
    ax1.set_xlabel('Search Range (p)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (s)', color=color_t, fontsize=12, fontweight='bold')
    line1 = ax1.plot(p_values, stats_time, color=color_t,
                     marker='^', markersize=10, linestyle='-.', linewidth=2,
                     label='Time (s)')
    ax1.tick_params(axis='y', labelcolor=color_t)

    ax2 = ax1.twinx()
    color_b = 'darkgreen'
    ax2.set_ylabel('Entropy / Bitrate (bpp)', color=color_b, fontsize=12, fontweight='bold')
    line2 = ax2.plot(p_values, stats_bpp, color=color_b,
                     marker='s', markersize=8, linestyle='-', linewidth=2, alpha=0.8,
                     label='Entropy (bpp)')

    ax2.tick_params(axis='y', labelcolor=color_b)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.title("Search Range Analysis: Cost vs. Quality", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()


def main():
    ref_path = "../data/part_2/ref_img.png" 
    curr_path = "../data/part_2/curr_img.png"
    
    ref_img = load_image_grayscale(ref_path)
    curr_img = load_image_grayscale(curr_path)

    print("Running Standard Motion Estimation (p=16)...")
    
    # TODO: Call your calc_residual method here
    # mvs, pred, res = calc_residual(curr_img, ref_img, search_range=16)
    #
    # if mvs is not None:
    #     visualize_results(curr_img, pred, res, mvs, 16)
    # else:
    #     print("Not implemented yet.")

    # You may need to add code to able answer the question 3 in the Report Questions subsection.
    question_3(curr_img, ref_img)

if __name__ == "__main__":
    main()