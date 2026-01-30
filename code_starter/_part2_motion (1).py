import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from part1_codec import shannon_entropy
# import cv2


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
        y_base = r * block_size
        for c in range(mb_cols):
            x_base = c * block_size

            # the block to encode
            curr_block = current_frame[y_base : y_base+block_size, x_base : x_base+block_size]

            min_sad = float('inf')
            best_dy, best_dx = 0, 0

            # search window in reference frame
            y_min_search = max(0, y_base - search_range)
            y_max_search = min(h - block_size, y_base + search_range)

            x_min_search = max(0, x_base - search_range)
            x_max_search = min(w - block_size, x_base + search_range)

            for y_ref in range(y_min_search, y_max_search + 1):
                for x_ref in range(x_min_search, x_max_search + 1):

                    candidate = ref_frame[y_ref : y_ref+block_size, x_ref : x_ref+block_size]

                    sad = np.sum(np.abs(curr_block - candidate))

                    if sad < min_sad:
                        min_sad = sad
                        best_dy = y_ref - y_base
                        best_dx = x_ref - x_base

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
    mb_rows, mb_cols, _ = motion_vectors.shape
    predicted_frame = np.zeros_like(ref_frame)

    for r in range(mb_rows):
        y_base = r * block_size
        for c in range(mb_cols):
            x_base = c * block_size

            # retrieve vector
            dy, dx = motion_vectors[r, c]

            # calculate where to copy from
            src_y = y_base + dy
            src_x = x_base + dx

            # copy pixels
            predicted_frame[y_base : y_base+block_size, x_base : x_base+block_size] = ref_frame[src_y : src_y+block_size, src_x : src_x+block_size]

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
    motion_vectors = motion_estimation(current_frame, ref_frame, block_size, search_range)
    predicted_frame = motion_compensation(ref_frame, motion_vectors, block_size)
    residual = current_frame - predicted_frame

    return motion_vectors, predicted_frame, residual


def main():
    ref_path = "data/part_2/ref_img.png"
    curr_path = "data/part_2/curr_img.png"

    # ref_path = "data/part_2/000285.png"
    # curr_path = "data/part_2/000286.png"

    ref_img = load_image_grayscale(ref_path)
    curr_img = load_image_grayscale(curr_path)

    search_ranges = [4, 8, 16, 32, 64]
    # search_ranges = [16]

    times = []
    entropies = []

    print(f"{'p':<5} | {'time (s)':<10} | {'entropy (bpp)':<15}")
    print("-" * 35)

    for p in search_ranges:
        start_t = time.time()

        mvs, pred, res = calc_residual(curr_img, ref_img, search_range=p)

        end_t = time.time()
        elapsed = end_t - start_t
        bpp = shannon_entropy(res.flatten())

        times.append(elapsed)
        entropies.append(bpp)

        print(f"{p:<5} | {elapsed:<10.2f} | {bpp:<15.4f}")

        if p == 16:
            visualize_results(curr_img, pred, res, mvs, p)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Search Range (p)')
    ax1.set_ylabel('Execution Time (s)', color=color)
    ax1.plot(search_ranges, times, color=color, marker='o', label='Time')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Residual Entropy (bpp)', color=color)
    ax2.plot(search_ranges, entropies, color=color, marker='s', linestyle='--', label='Entropy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Trade-off: Search Range vs Time vs Compression")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
