import cv2
import os

def extract_sequence(start_idx=0, num_frames=-1):
    video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    cap = cv2.VideoCapture(video_url)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames == -1:
        num_frames = total_frames - start_idx

    print(f"Extracting {num_frames} frames starting from {start_idx}...")
    
    # Jump to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    os.makedirs("frames", exist_ok=True)

    count = 0
    while count < num_frames:
        success, frame = cap.read()
        if not success:
            break
            
        filename = f"frames/{count:06d}.png"
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        count += 1

    print(f"Done! Saved {count} frames to folder 'frames/'.")
    cap.release()

if __name__ == "__main__":
    s_in = input("Start Index (default 0): ")
    n_in = input("Number of Frames (-1 for all): ")
    
    start = int(s_in) if s_in else 0
    num = int(n_in) if n_in else -1
    
    extract_sequence(start, num)