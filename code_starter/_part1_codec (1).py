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
    if not len(values):
        return 0

    _, counts = np.unique(values, return_counts=True)
    prob = counts / len(values)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


class MiniCodec:
    def __init__(self, block_size=16):
        self.block_size = block_size

    def get_prediction(self, recon_image, x, y, mode):
        """
        Generates a prediction block based on the mode.
        recon_image (I_rec): The full image (so far reconstructed)
        x, y: Top-left coordinates of the current block
        mode: 'vertical' or 'horizontal' in Intra-Prediction Modes you need to implement
        """
        bs = self.block_size
        prediction = np.full((bs, bs), 128.0, dtype=np.float32)

        if mode == 'vertical' and y > 0:  # copy the top row (y-1) downwards
            top_row = recon_image[y - 1, x : x + bs]
            prediction[:, :top_row.shape[0]] = top_row  # row might be shorter than bs at the far right

        elif mode == 'horizontal' and x > 0:  # copy the left column (x-1) rightwards
            left_col = recon_image[y : y + bs, x - 1]
            prediction[:left_col.shape[0], :] = left_col.reshape(-1, 1)  # column might be shorter than bs at the bottom

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
                cur_block = img[y:y + self.block_size, x:x + self.block_size]
                bh, bw = cur_block.shape

                # predict
                v_pred = self.get_prediction(recon_img, x, y, 'vertical')[:bh, :bw]
                h_pred = self.get_prediction(recon_img, x, y, 'horizontal')[:bh, :bw]

                # calculate SAD
                v_sad = np.sum(np.abs(cur_block - v_pred))
                h_sad = np.sum(np.abs(cur_block - h_pred))

                if v_sad < h_sad:
                    best_pred = v_pred
                    modes_stats['v'] += 1
                else:
                    best_pred = h_pred
                    modes_stats['h'] += 1

                # residual
                res = cur_block - best_pred

                # quantize
                res_q = np.round(res / Q) * Q if Q else res
                residuals.append(res_q.flatten())

                # reconstruct
                rec_block = np.clip(best_pred + res_q, 0, 255)

                # update
                recon_img[y:y + bh, x:x + bw] = rec_block

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
        # higher psnr = higher quality
        if not mse:
            psnr = 100.0
        else:
            psnr = 10 * np.log10((255 ** 2) / mse)

        # lower bpp = higher compression
        bpp = shannon_entropy(residuals)

        return psnr, bpp


if __name__ == "__main__":
    img_path = 'data/part_1/curr_img.png'

    img = cv2.imread(img_path, 0)
    assert img is not None, "Failed to load image"
    img = img.astype(np.float32)

    codec = MiniCodec()

    Qs = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    results_bpp = []
    results_psnr = []

    print(f"{'Q':<5} | {'PSNR (dB)':<10} | {'Bitrate (bpp)':<15} | {'V/H'}")
    print("-" * 50)

    for Q in Qs:
        rec, res, stat = codec.encode_image(img, Q)
        psnr, bpp = codec.calculate_metrics(img, rec, res)

        results_bpp.append(bpp)
        results_psnr.append(psnr)

        print(f"{Q:<5} | {psnr:<10.2f} | {bpp:<15.2f} | {stat['v']}/{stat['h']}")

        if Q == 50:
            cv2.imwrite('figures/part_1/rec_50.png', rec)

    plt.figure(figsize=(8, 6))
    plt.plot(results_bpp, results_psnr, marker='o', linestyle='-')
    plt.xlabel("Estimated Bitrate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.show()
