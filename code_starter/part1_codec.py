# part1_codec.py
import numpy as np
import cv2
import matplotlib.pyplot as plt


def shannon_entropy(values):
    """
    Computes the Shannon Entropy (H) of a 1D array of values.
    Formula:
      H(X) = - sum(P(x) * log2(P(x)))
    Args:
        values (np.ndarray): 1D array of discrete symbols (e.g., quantized residuals).

    Returns:
        float: The entropy in bits per symbol.
    """
    # TODO: implement bbp

    if len(values) == 0:
        return 0.0

    values = np.asarray(values)
    unique_values, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    log_probs = np.log2(probs)

    entropy = (-1) * np.sum(probs * log_probs)

    return entropy





class MiniCodec:
    def __init__(self, block_size=16):
        self.block_size = block_size

    def get_prediction(self, recon_image, x, y, mode):
        """
        Generates a prediction block based on the mode.
        recon_image (I_rec): The full image (so far reconstructed)
        x, y: Top-left coordinates of the current block
        mode: 'vertical' or 'horizontal' in Intra-Prediction Modes you need to implement.
        """
        # h, w = recon_image.shape
        bs = self.block_size
        prediction = np.zeros((bs, bs), dtype=np.float32)

        # Carefully process the edge cases noted in the document.
        # TODO: Implement Vertical Prediction

        if mode == 'vertical':
            if y > 0: # Tránh hàng trên cùng
                # Lấy hàng pixel ngay phía trên block hiện tại (y-1)
                top_row = recon_image[y - 1, x : x + bs]
                prediction[:, :top_row.shape[0]] = top_row
            else:
                prediction.fill(128.0) # Nếu không có hàng trên, điền giá trị trung tính


        # TODO: Implement Horizontal Prediction
        elif mode == 'horizontal':
            if x > 0: # Tránh cột bên trái cùng
                # Lấy cột pixel ngay bên trái block hiện tại (x-1)
                left_col = recon_image[y : y + bs, x - 1]
                prediction[:left_col.shape[0], :] = left_col.reshape(-1, 1)
            else:
                prediction.fill(128.0) # Nếu không có cột bên trái, điền giá trị trung tính
            
        return prediction

    def encode_image(self, img, Q):
        """
        Iterates over the image in raster scan order, performing the DPCM cycle for each block:

        1. Predict (P): Generate prediction using best mode (SAD) from *reconstructed* neighbors. (use the get_prediction method()) and store the which mode preferred.
        2. Residual (R): Compute error relative to source: R = I_src - P
        3. Quantize (R_q): Apply lossy quantization: R_q = round(R / Q) * Q
        4. Reconstruct (I_rec): Simulate decoder view: I_rec = P + R_q
        5. Update: Overwrite current block in `recon_img` with I_rec for future predictions.

        Args:
            img: Source image (I_src).
            Q: Quantization step size.

        Returns:
            recon_img: Full reconstructed image (I_rec).
            residuals: 1D array of quantized residuals (R_q) for entropy calculation.
            modes_stats:  for statistic the number of blocks preferred Vertical Mode vs. Horizontal Mode
        """
        h, w = img.shape
        recon_img = np.zeros_like(img, dtype=np.float32)
        residuals = []
        modes_stats = {'v': 0, 'h': 0}
        
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                # TODO: Implement Intra-Frame prediction for each block
                current_block = img[y:y + self.block_size, x:x + self.block_size]
                h_block, w_block = current_block.shape

                pred_v = self.get_prediction(recon_img, x, y, 'vertical')[:h_block, :w_block]
                pred_h = self.get_prediction(recon_img, x, y, 'horizontal')[:h_block, :w_block]
                # : là cắt cho khớp kich thước current_block (trường hợp biên)

                sad_v = np.sum(np.abs(current_block - pred_v))
                sad_h = np.sum(np.abs(current_block - pred_h))

                if sad_v < sad_h:
                    best_pred = pred_v
                    modes_stats['v'] += 1
                else:
                    best_pred = pred_h
                    modes_stats['h'] += 1

                # TODO: Compute Residual
                residual = current_block - best_pred

                # TODO: Quantize
                if Q != 0:
                    quantized_residual = np.round(residual / Q) * Q
                else:
                    quantized_residual = residual
                residuals.append(quantized_residual.flatten())

                # TODO: Reconstruct
                reconstructed_block = np.clip(best_pred + quantized_residual, 0, 255)

                # update state
                recon_img[y:y + h_block, x:x + w_block] = reconstructed_block


        return recon_img, np.concatenate(residuals), modes_stats

    def calculate_metrics(self, original, reconstructed, residuals):
        """
        Computes codec performance metrics: distortion (PSNR) and bitrate (BPP). Estimates the theoretical bitrate by calling the `shannon_entropy` function

        Args:
            original (np.ndarray): The raw source image (I_src), acting as the ground truth.
            reconstructed (np.ndarray): The decoded output (I_rec), used to measure distortion.
            residuals (np.ndarray): Flattened array of quantized prediction errors (R_q),
                                    analyzed statistically to estimate the bitrate.


        Returns:
            psnr (float): Peak Signal-to-Noise Ratio (dB).
            bpp (float): Theoretical Bits Per Pixel.
        """
        mse = np.mean((original - reconstructed) ** 2)
        # TODO: Calculate PSNR
        # psnr = 0 # Implement PSNR formula
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 10 * np.log10((255 ** 2) / float(mse))


        # TODO: Calculate Entropy of residuals
        # bpp = 0
        bpp = shannon_entropy(residuals)
        
        return psnr, bpp

# Question 1 in report
def report_mode_statistics(codec, img, Q=10):
    """
    Yêu cầu 1: Thống kê tỷ lệ Vertical vs Horizontal Mode.
    """
    print(f"\n--- Report 1: Mode Statistics (Q={Q}) ---")
    _, _, stats = codec.encode_image(img, Q=Q)

    total_blocks = stats['v'] + stats['h']
    if total_blocks == 0:
        print("Error: No blocks processed.")
        return

    perc_v = (stats['v'] / total_blocks) * 100
    perc_h = (stats['h'] / total_blocks) * 100

    print(f"Total Blocks: {total_blocks}")
    print(f"Vertical Mode:   {stats['v']} ({perc_v:.2f}%)")
    print(f"Horizontal Mode: {stats['h']} ({perc_h:.2f}%)")

def report_rate_distortion(codec, img):
    """
    Yêu cầu 2: Vẽ biểu đồ đánh đổi giữa PSNR và Bitrate với các giá trị Q khác nhau.
    """
    print("\n--- Report 2: Generating Rate-Distortion Curve ---")
    qs = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    psnrs = []
    bpps = []

    for q in qs:
        rec, res, _ = codec.encode_image(img, Q=q)
        psnr, bpp = codec.calculate_metrics(img, rec, res)

        psnrs.append(psnr)
        bpps.append(bpp)
        print(f"Q={q:2d} | PSNR={psnr:.2f} dB | BPP={bpp:.2f}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(bpps, psnrs, 'bo-', linewidth=2, markersize=8)

    # Trang trí biểu đồ
    plt.title("Rate-Distortion Curve (Q-Sweep)")
    plt.xlabel("Estimated Bitrate (bpp) [Lower is better]")
    plt.ylabel("PSNR (dB) [Higher is better]")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Gắn nhãn Q lên từng điểm để dễ nhìn
    for i, q in enumerate(qs):
        plt.annotate(f"Q={q}", (bpps[i], psnrs[i]), textcoords="offset points", xytext=(5, 5), ha='left')

    plt.show()


def report_artifacts(codec, img, Q=50):
    """
    Yêu cầu 3: Hiển thị ảnh tái tạo tại Q cao để soi 'Blocking Artifacts'.
    """
    print(f"\n--- Report 3: Artifact Analysis (Q={Q}) ---")
    rec, res, _ = codec.encode_image(img, Q=Q)

    plt.figure(figsize=(12, 6))

    # Ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Ảnh tái tạo (bị vỡ hạt)
    plt.subplot(1, 2, 2)
    plt.imshow(rec, cmap='gray')
    plt.title(f"Reconstructed (Q={Q})\nNotice blocking artifacts at edges")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = '../data/part_1/curr_img.png'
    
    img = cv2.imread(img_path, 0).astype(np.float32)

    codec = MiniCodec()
    
    # Example run
    # May change to answer the question about Quantization–Distortion with Estimated Rate

    report_mode_statistics(codec, img, Q=10)
    report_rate_distortion(codec, img)
    report_artifacts(codec, img, Q=50)

    # rec, res, stat = codec.encode_image(img, Q=50)
    # plt.imshow(rec, cmap='gray')
    # plt.title("Reconstructed (Q=50)")
    # plt.show()

