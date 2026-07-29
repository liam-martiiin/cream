#!/usr/bin/env python
import os
import sys
import time
import math
import select

# ==========================================
# 1. FORCE PRIORITY PATH PATCH
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.insert(0, src_dir)

# ==========================================
# 2. HARDWARE IMPORTS (Lab Fork Layout)
# ==========================================
from lerobot.robots.so_follower.so_simplified_follower_h import SOSimplifiedFollowerH
from lerobot.robots.so_follower.config_so_follower import SOSimplifiedFollowerHConfig

# (Update this import path if your leader class is named or located differently)
from lerobot.teleoperators.so_leader.so_simplified_leader_h import SOSimplifiedLeaderH
from lerobot.teleoperators.so_leader.config_so_leader import SOSimplifiedLeaderHConfig

# ==========================================
# 3. ROBOT GEOMETRY & CALIBRATION
# ==========================================
# Standard link lengths in meters
L1 = 0.0139  # Base floor to shoulder joint axis
L2 = 0.0111  # Shoulder axis to elbow axis
L3 = 0.0133  # Elbow axis to wrist axis

# Tool Center Point (TCP) Offset - MEASURE THESE!
Z_OFF = 0.0128  # Vertical distance from wrist pin straight down to marker tip
R_OFF = 0.007  # Horizontal distance from wrist pin forward to marker tip

# The absolute orientation constraint (Sum of Lift + Elbow + Wrist when pen is vertical)
GAMMA_VERTICAL = -1.21  # UPDATE THIS with the value from your test_radians.py script!

# Pre-calculated structural geometry for the FK free-tracking mode
L_TOOL = math.sqrt(Z_OFF**2 + R_OFF**2)
PSI = math.atan2(-Z_OFF, R_OFF)

# ==========================================
# 4. KINEMATICS ENGINE
# ==========================================
def forward_kinematics(t1, t2, t3, t4):
    """Calculates marker tip (X, Y, Z) when the arm is moving freely."""
    total_tool_angle = t2 + t3 + t4 + PSI
    
    Rw = L2 * math.cos(t2) + L3 * math.cos(t2 + t3)
    Zw = L1 + L2 * math.sin(t2) + L3 * math.sin(t2 + t3)
    
    R = Rw + L_TOOL * math.cos(total_tool_angle)
    Z = Zw + L_TOOL * math.sin(total_tool_angle)
    
    X = R * math.cos(t1)
    Y = R * math.sin(t1)
    return X, Y, Z

def inverse_kinematics(X, Y, Z_target):
    """Calculates joint angles to reach (X, Y) while holding Z locked and marker vertical."""
    t1 = math.atan2(Y, X)
    R = math.sqrt(X**2 + Y**2)
    
    # Shift target to find where the WRIST joint needs to go
    Rw = R - R_OFF
    Zw = Z_target + Z_OFF
    
    # Translate relative to shoulder pivot
    R_rel = Rw
    Z_rel = Zw - L1
    
    D_sq = R_rel**2 + Z_rel**2
    
    # Security check to prevent math domain errors if point is out of physical reach
    cos_t3 = (D_sq - L2**2 - L3**2) / (2.0 * L2 * L3)
    if not (-1.0 <= cos_t3 <= 1.0):
        raise ValueError("Target position out of physical reach!")
        
    t3 = math.acos(cos_t3)
    t2 = math.atan2(Z_rel, R_rel) - math.atan2(L3 * math.sin(t3), L2 + L3 * math.cos(t3))
    
    # The magical remainder calculation that forces the marker to stay vertical
    t4 = GAMMA_VERTICAL - t2 - t3
    
    return t1, t2, t3, t4

def check_keyboard_toggle():
    """Non-blocking check for terminal key presses."""
    dr, dw, de = select.select([sys.stdin], [], [], 0.0)
    if dr:
        return sys.stdin.readline().strip()
    return None

# ==========================================
# 5. MAIN CONTROL LOOP
# ==========================================
def main():
    print("Initializing configurations...")
    
    # Configure LEADER (Using Lab Fork ID Registry)
    lead_config = SOSimplifiedLeaderHConfig()
    lead_config.id = "right_leader"  # Make sure you registered the leader with this name!
    lead_config.cameras = {}
    leader = SOSimplifiedLeaderH(lead_config)
    
    # Configure FOLLOWER (Using Lab Fork ID Registry)
    follow_config = SOSimplifiedFollowerHConfig()
    follow_config.id = "right_follower"
    follow_config.cameras = {}
    follower = SOSimplifiedFollowerH(follow_config)
    
    print("Connecting to Leader...")
    leader.connect()
    leader.bus.disable_torque()  # MUST disable torque so you can move it by hand!
    
    print("Connecting to Follower...")
    follower.connect()
    # DO NOT disable torque on the follower. It needs power to drive to the IK coordinates!
    
    height_lock_enabled = False
    locked_Z = 0.0
    
    print("\n=============================================")
    print("          TELEOPERATION ACTIVE               ")
    print("---------------------------------------------")
    print(" Press 'l' and ENTER to Lock/Unlock Height.  ")
    print(" Press Ctrl+C to Exit.                       ")
    print("=============================================\n")
    
    try:
        while True:
            # 1. Read Leader State
            leader_obs = leader.get_observation_radians()
            t1 = leader_obs["shoulder_pan.pos"]
            t2 = leader_obs["shoulder_lift.pos"]
            t3 = leader_obs["elbow_flex.pos"]
            t4 = leader_obs["wrist_flex.pos"]
            
            print(follower.get_observation())
            
            # 2. Check for Lock Toggle
            key = check_keyboard_toggle()
            if key == 'l':
                height_lock_enabled = not height_lock_enabled
                if height_lock_enabled:
                    _, _, current_Z = forward_kinematics(t1, t2, t3, t4)
                    locked_Z = current_Z
                    print(f"[LOCK ON] Marker pinned at Z: {locked_Z:.4f} meters")
                else:
                    print("[LOCK OFF] Returning to free tracking mode.")
            
            # 3. Calculate Targets
            if height_lock_enabled:
                # Calculate where human wants to be on X/Y, but enforce the locked Z
                current_X, current_Y, _ = forward_kinematics(t1, t2, t3, t4)
                try:
                    f1, f2, f3, f4 = inverse_kinematics(current_X, current_Y, locked_Z)
                    action = {
                        "shoulder_pan.pos": f1,
                        "shoulder_lift.pos": f2,
                        "elbow_flex.pos": f3,
                        "wrist_flex.pos": f4,
                    }
                except ValueError:
                    # Fallback to direct joint space mirror if operator pulls out of reach
                    action = {"shoulder_pan.pos": t1, "shoulder_lift.pos": t2, "elbow_flex.pos": t3, "wrist_flex.pos": t4}
            else:
                # Mirror tracking directly in joint space
                action = {"shoulder_pan.pos": t1, "shoulder_lift.pos": t2, "elbow_flex.pos": t3, "wrist_flex.pos": t4}
                
            # 4. Command the Follower
            follower.send_action_radians(action)
            time.sleep(0.01)  # ~100Hz frequency loop
            
    except KeyboardInterrupt:
        print("\nShutting down control loops safely...")
    finally:
        leader.disconnect()
        follower.disconnect()
        print("Hardware disconnected. Goodbye!")

if __name__ == "__main__":
    main()
