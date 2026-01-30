# part3_flow.py
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as TF
from PIL import Image
# import numpy as np
import matplotlib.pyplot as plt
from part1_codec import shannon_entropy
# import cv2


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DeepCodec:
    def __init__(self):
        """
        Initializes the DeepCodec with a pre-trained Optical Flow model.
        """
        self.model = self.load_model()
        self.transforms = Raft_Small_Weights.DEFAULT.transforms()

    def load_model(self):
        """
        Loads the RAFT_SMALL model with default pre-trained weights.
        """
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(DEVICE)
        model.eval()

        return model

    def warp_flow(self, img, flow):
        """
        Warps an image based on the optical flow field using Differentiable Bilinear Interpolation.

        Args:
            img (Tensor): Reference image (1, 3, H, W). Normalized 0-1.
            flow (Tensor): Optical flow field (1, 2, H, W) -> (dx, dy).

        Returns:
            Tensor: Warped image (1, 3, H, W).
        """
        B, C, H, W = img.shape

        # (x, y) coordinates base meshgrid (B, H, W, 2)
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W, 1).repeat(B, 1, 1, 1).to(DEVICE)
        yy = yy.view(1, H, W, 1).repeat(B, 1, 1, 1).to(DEVICE)
        grid = torch.cat((xx, yy), -1).float()

        # add the flow to the grid
        grid = grid + flow.permute(0, 2, 3, 1)

        # normalize grid to [-1,1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(H - 1, 1) - 1.0

        # sample
        warped_img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)

        return warped_img

    def predict(self, ref_img_path, curr_img_path):
        """
        Performs the full Inter-frame prediction pipeline:
        1. Load Images
        2. Estimate Flow (Inference)
        3. Warp Reference (Motion Compensation)
        4. Compute Residual

        Args:
            ref_img_path: Path to reference frame.
            curr_img_path: Path to current frame.

        Returns:
            prediction (Tensor): The warped reference image.
            residual (Tensor): The difference (Curr - Pred).
            flow (Tensor): The estimated dense motion field.
        """
        # Load Data
        img1 = Image.open(ref_img_path).convert('RGB')
        img2 = Image.open(curr_img_path).convert('RGB')

        # Convert to Tensor (0-1)
        img1_t = TF.to_tensor(img1).unsqueeze(0).to(DEVICE)
        img2_t = TF.to_tensor(img2).unsqueeze(0).to(DEVICE)

        # Apply RAFT specific transforms (resizing/normalization)
        img1_in, img2_in = self.transforms(img1_t, img2_t)

        # original dim for resizing
        w_orig, h_orig = img1.size

        # 2. Inference
        print("Computing Optical Flow...")
        predicted_flow = None
        residual = None
        prediction = None

        with torch.no_grad():
            # RAFT returns a list of flows (iterations)
            # the last one is the final result
            predicted_flow = self.model(img1_in, img2_in)[-1]

            # flow might have different resolution due to transforms
            # resize back to original resolution to match 'img1_t'
            if predicted_flow.shape[-2:] != (h_orig, w_orig):
                predicted_flow = F.interpolate(predicted_flow, size=(h_orig, w_orig), mode='bilinear', align_corners=False)

            # warp the original reference image using the flow
            prediction = self.warp_flow(img1_t, predicted_flow)

            # calculate residual
            # input images are [0, 1] while residual should be in 0-255 scale for entropy calc
            residual = (img2_t - prediction) * 255.0

        return prediction, residual, predicted_flow


def visualize_results(prediction, residual, flow):
    """
    Visualizes Prediction, Residual, and Optical Flow Field.
    """
    res_flat = residual.cpu().numpy().flatten().astype(int)
    entropy = shannon_entropy(res_flat)

    pred_np = prediction.squeeze().permute(1, 2, 0).cpu().numpy().clip(0, 1)
    pred_np_gray = 0.299 * pred_np[..., 0] + 0.587 * pred_np[..., 1] + 0.114 * pred_np[..., 2]

    res_np = residual.squeeze().cpu().numpy()
    res_gray = 0.299 * res_np[0] + 0.587 * res_np[1] + 0.114 * res_np[2]

    flow_img = flow_to_image(flow).squeeze().permute(1, 2, 0).cpu().numpy() / 255.0

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # cv2.imwrite("figures/part_3/3_predicted.png", (pred_np_gray * 255).astype(np.uint8))
    # cv2.imwrite("figures/part_3/3_residual.png", (res_gray + 128).astype(np.uint8))

    ax[0].imshow(pred_np_gray)
    ax[0].set_title("Warped Reference (Prediction)")
    ax[0].axis('off')

    ax[1].imshow(res_gray, cmap='gray', vmin=-50, vmax=50)
    ax[1].set_title(f"Residual\nEntropy: {entropy:.3f} bpp")
    ax[1].axis('off')

    ax[2].imshow(flow_img)
    ax[2].set_title("Dense Optical Flow Field\n(Color=Direction, Intensity=Magnitude)")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ref_path = "data/part_3/ref_img.png"
    curr_path = "data/part_3/curr_img.png"

    codec = DeepCodec()

    # Run Pipeline
    pred, res, flow = codec.predict(ref_path, curr_path)

    # Visualize
    visualize_results(pred, res, flow)
