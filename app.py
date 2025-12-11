import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
import os
import time

# ==========================================
# 1. MODEL DEFINITION (Must match training)
# ==========================================
class StabGenerator(nn.Module):
    def __init__(self):
        super(StabGenerator, self).__init__()
        self.clip_len = 32
        self.encoder = nn.Sequential(
            nn.Conv3d(5, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.AdaptiveAvgPool3d((self.clip_len, 1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * self.clip_len, 512), nn.ReLU(),
            nn.Linear(512, self.clip_len * 6)
        )

    def forward(self, x_5ch, x_3ch_raw):
        b, _, t, h, w = x_5ch.shape
        features = self.encoder(x_5ch)
        theta = self.regressor(features).view(-1, 2, 3)
        identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x_5ch.device)
        identity = identity.view(1, 2, 3).repeat(theta.shape[0], 1, 1)
        final_theta = identity + (theta * 0.1)
        x_flat = x_3ch_raw.permute(0, 2, 1, 3, 4).reshape(-1, 3, h, w)
        grid = F.affine_grid(final_theta, x_flat.size(), align_corners=True)
        stabilized = F.grid_sample(x_flat, grid, align_corners=True)
        return stabilized.view(b, t, 3, h, w).permute(0, 2, 1, 3, 4)

# ==========================================
# 2. HELPER: LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StabGenerator().to(DEVICE)
    MODEL_PATH = "best_stab_gan.pth"
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        return model, DEVICE
    else:
        return None, DEVICE

# ==========================================
# 3. HELPER: PROCESS VIDEO
# ==========================================
def process_video(input_path, output_path, model, device):
    CHUNK_SIZE = 32
    RESIZE = (192, 192)
    
    cap = cv2.VideoCapture(input_path)
    
    # Get Video Properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define Codec (mp4v is widely compatible for download, though standard browsers prefer h264)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # We will save SIDE-BY-SIDE video (384x192)
    out = cv2.VideoWriter(output_path, fourcc, fps, (RESIZE[0]*2, RESIZE[1]))
    
    frames_buffer = []
    
    # Streamlit Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize for Model
        frame_resized = cv2.resize(frame, RESIZE)
        frames_buffer.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        
        # Process when buffer is full
        if len(frames_buffer) == CHUNK_SIZE:
            _stabilize_chunk(frames_buffer, model, device, out)
            frames_buffer = [] # Clear buffer
            
            # Update Progress
            frame_idx += CHUNK_SIZE
            progress = min(frame_idx / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx}/{total_frames}...")

    # Process remaining frames (padding)
    if len(frames_buffer) > 0:
        original_len = len(frames_buffer)
        padding = [frames_buffer[-1]] * (CHUNK_SIZE - original_len)
        frames_buffer.extend(padding)
        _stabilize_chunk(frames_buffer, model, device, out, valid_len=original_len)
        
    cap.release()
    out.release()
    progress_bar.progress(1.0)
    status_text.text("✅ Processing Complete!")

def _stabilize_chunk(chunk, model, device, writer, valid_len=32):
    chunk_np = np.array(chunk)
    
    # Optical Flow
    flows = [np.zeros((192, 192, 2), dtype=np.float32)]
    prev = cv2.cvtColor(chunk_np[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(chunk_np)):
        curr = cv2.cvtColor(chunk_np[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        prev = curr
        
    # Tensor Prep
    u_rgb = torch.from_numpy(chunk_np).permute(3, 0, 1, 2).float() / 255.0
    u_flow = torch.from_numpy(np.array(flows)).permute(3, 0, 1, 2).float()
    
    in_5ch = torch.cat([u_rgb, u_flow], dim=0).unsqueeze(0).to(device)
    in_raw = u_rgb.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        stab_tensor = model(in_5ch, in_raw)
        
    # Convert Output
    stab_chunk = stab_tensor[0].permute(1, 2, 3, 0).cpu().numpy() * 255.0
    stab_chunk = np.clip(stab_chunk, 0, 255).astype(np.uint8)
    
    # Write to Video
    for k in range(valid_len):
        left = cv2.cvtColor(chunk_np[k], cv2.COLOR_RGB2BGR) # Unstable
        right = cv2.cvtColor(stab_chunk[k], cv2.COLOR_RGB2BGR) # Stable
        combined = np.hstack((left, right))
        writer.write(combined)

# ==========================================
# 4. STREAMLIT UI LAYOUT
# ==========================================
def main():
    st.set_page_config(page_title="AI Video Stabilizer", layout="centered")
    
    st.title("🎥 Deep Video Stabilizer")
    st.markdown("""
    Upload a shaky video, and the AI will stabilize it using your trained **CNN+GAN** model.
    """)
    
    # Load Model
    model, device = load_model()
    
    if model is None:
        st.error("❌ Model weights (`best_stab_gan.pth`) not found! Please place the file in the same directory.")
        return

    st.sidebar.success(f"Model Loaded on: {device}")

    # File Uploader
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        input_path = tfile.name
        
        st.video(input_path)
        st.info(f"Filename: {uploaded_file.name}")

        if st.button("✨ Stabilize Video"):
            output_path = os.path.join(tempfile.gettempdir(), "stabilized_output.mp4")
            
            try:
                with st.spinner('AI is processing video frame by frame...'):
                    process_video(input_path, output_path, model, device)
                
                # Show Result
                st.success("Video Stabilized Successfully!")
                
                # Display Output (Note: Browser support for .mp4v varies, so we provide download)
                st.subheader("Result (Left: Original | Right: Stabilized)")
                st.video(output_path)
                
                # Download Button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Stabilized Video",
                        data=file,
                        file_name="stabilized_result.mp4",
                        mime="video/mp4"
                    )
            
            except Exception as e:
                st.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
