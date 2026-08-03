import numpy as np
from lerobot.datasets import LeRobotDataset

# --- Configuration ---
# Place the repo IDs of the datasets that need the color fix
BROKEN_DATASETS = [
    "CelerySticks/tracingLineData_20260730_153120",
    "CelerySticks/tracingLineData_20260730_144408"
]

# Place the repo IDs of the datasets that are already correct
GOOD_DATASETS = [
    "CelerySticks/tracingLineData_20260729_141014",
    "CelerySticks/tracingLineData_20260729_151917"
]

# The name of your brand new, combined dataset
TARGET_DATASET_ID = "CelerySticks/merged_uninverted_varied_heights"

# Update this if you have multiple cameras (e.g., ["observation.images.laptop", "observation.images.wrist"])
CAMERA_KEYS = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]  # Update with your actual camera keys
def process_and_merge():
    # 1. Initialize the target dataset using the schema of the first dataset
    print(f"Initializing target dataset {TARGET_DATASET_ID}...")
    reference_dataset = LeRobotDataset(BROKEN_DATASETS[0])
    
    merged_dataset = LeRobotDataset.create(
        repo_id=TARGET_DATASET_ID,
        fps=reference_dataset.fps,
        robot_type=reference_dataset.meta.robot_type,
        features=reference_dataset.features,
    )

    total_episodes_saved = 0

    # Helper function to process a single dataset
    def ingest_dataset(repo_id, needs_fixing):
        nonlocal total_episodes_saved
        print(f"\nProcessing dataset: {repo_id} (Needs fixing: {needs_fixing})")
        
        source_dataset = LeRobotDataset(repo_id)
        current_source_ep = 0

        for i in range(len(source_dataset)):
            frame_data = source_dataset[i]
            
            # Check for episode boundary in the source dataset
            ep_index = frame_data["episode_index"].item() if hasattr(frame_data["episode_index"], "item") else frame_data["episode_index"]
            if ep_index != current_source_ep:
                merged_dataset.save_episode()
                total_episodes_saved += 1
                print(f"  Saved merged episode {total_episodes_saved} (Finished source episode {current_source_ep})")
                current_source_ep = ep_index

            # Extract task string safely (handles both string and byte formats)
            task_val = frame_data.get("task", "Draw a line")
            if isinstance(task_val, bytes):
                task_val = task_val.decode("utf-8")
            elif hasattr(task_val, "item"):
                task_val = task_val.item()

            # Extract standard features including the required task field
            new_frame = {
                "observation.state": frame_data["observation.state"].numpy(),
                "action": frame_data["action"].numpy(),
                "task": str(task_val),
            }
            
            # Extract pressure if it was recorded in your dataset schema
            if "observation.pressure" in frame_data:
                new_frame["observation.pressure"] = frame_data["observation.pressure"].numpy()
            elif "pressure" in frame_data:
                new_frame["pressure"] = frame_data["pressure"].numpy()

            # Process cameras
            for cam in CAMERA_KEYS:
                img_tensor = frame_data[cam]
                img_array = img_tensor.numpy()
                
                # If the shape is channel-first (3, H, W), transpose to (H, W, 3)
                if img_array.ndim == 3 and img_array.shape[0] == 3:
                    img_array = np.transpose(img_array, (1, 2, 0))

                if needs_fixing:
                    # Apply the winning fix from your diagnostic test (e.g., astype(np.uint8))
                    img_array = img_array.astype(np.uint8) 
                    
                new_frame[cam] = img_array

            # Add to the new dataset
            merged_dataset.add_frame(new_frame)

        # Save the final episode of this source dataset
        merged_dataset.save_episode()
        total_episodes_saved += 1
        print(f"  Saved merged episode {total_episodes_saved} (Finished final source episode {current_source_ep})")

    # 2. Ingest all broken datasets (applying the fix)
    for repo_id in BROKEN_DATASETS:
        ingest_dataset(repo_id, needs_fixing=True)

    # 3. Ingest all good datasets (passing through as-is)
    for repo_id in GOOD_DATASETS:
        ingest_dataset(repo_id, needs_fixing=False)

    # 4. Push the final combined dataset to the Hub
    print(f"\nPushing merged dataset with {total_episodes_saved} total episodes to Hugging Face Hub...")
    merged_dataset.push_to_hub()
    print("Done!")

if __name__ == "__main__":
    process_and_merge()