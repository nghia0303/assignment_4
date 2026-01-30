# part3_flow.py
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from part1_codec import shannon_entropy


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
        # TODO: Load the RAFT Small model using torchvision.models.optical_flow, move to DEVICE, and consider use .eval() mode to avoid unncessary computation.
        model = None 
        
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
        
        # =================================================================
        # TODO: Implement Differentiable Warping via Grid Sample. Hint: May use F.grid_sample() for samling
        # =================================================================
        
        warped_img = img.clone() 
        
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
        # 1. Load Data
        img1 = Image.open(ref_img_path).convert('RGB')
        img2 = Image.open(curr_img_path).convert('RGB')
        
        # Convert to Tensor (0-1)
        img1_t = TF.to_tensor(img1).unsqueeze(0).to(DEVICE)
        img2_t = TF.to_tensor(img2).unsqueeze(0).to(DEVICE)

        # Apply RAFT specific transforms (resizing/normalization)
        img1_in, img2_in = self.transforms(img1_t, img2_t)

        # 2. Inference
        print("Computing Optical Flow...")
        predicted_flow = None
        residual = None
        prediction = None

        with torch.no_grad():
            # TODO: Run the model and get the result into predicted_flow, wraping to get prediction, and calculate residual. 
            # RAFT transforms might resize images to be divisible by 8. We need flow at original res.
            pass        
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
    ref_path = "../data/part_3/ref_img.png" 
    curr_path = "../data/part_3/curr_img.png"
   
    codec = DeepCodec()
    
    # Run Pipeline
    pred, res, flow = codec.predict(ref_path, curr_path)
    
    # Visualize
    visualize_results(pred, res, flow)