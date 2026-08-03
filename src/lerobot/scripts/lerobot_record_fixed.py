import numpy as np
from lerobot.scripts.lerobot_record import record
from lerobot.processor import RobotProcessorPipeline

# --- YOUR CUSTOM PROCESSOR ---
class FixFloatToUint8Processor(RobotProcessorPipeline):
    def forward(self, observation):
        """Cast float images (0-255) to standard uint8 (0-255) for saving."""
        obs = dict(observation)
        for key, val in obs.items():
            # Identify camera images
            if "image" in key or "camera" in key:
                # Only cast if it's a float dtype
                if hasattr(val, 'dtype') and val.dtype.kind == 'f':
                    obs[key] = val.astype(np.uint8)
        return obs

if __name__ == "__main__":
    print("Starting recording with uint8 fix...")
    
    # Magic happens here: The @parser.wrap() decorator inside lerobot_record.py
    # automatically reads your command-line arguments and injects the 'cfg' object.
    # We just supply our custom processor via keyword argument.
    record(robot_observation_processor=FixFloatToUint8Processor())