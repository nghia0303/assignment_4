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
            # 2. Copy the matching block from ref_frame to predicted_frame.
            pass 

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
    # 2. predict
    # 3. Calculate residual
    return None, None, None


def main():
    ref_path = "../data/part_2/ref_img.png" 
    curr_path = "../data/part_2/curr_img.png"
    
    ref_img = load_image_grayscale(ref_path)
    curr_img = load_image_grayscale(curr_path)

    print("Running Standard Motion Estimation (p=16)...")
    
    # TODO: Call your calc_residual method here
    mvs, pred, res = calc_residual(curr_img, ref_img, search_range=16)
    
    if mvs is not None:
        visualize_results(curr_img, pred, res, mvs, 16)
    else:
        print("Not implemented yet.")

    # You may need to add code to able answer the question 3 in the Report Questions subsection.
    

if __name__ == "__main__":
    main()