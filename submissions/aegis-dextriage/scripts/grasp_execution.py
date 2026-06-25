import sys
import mujoco
from pathlib import Path
from controller import DexTriageController, set_freejoint_pose
from planner import DexTriagePlanner

def execute_grasp(model_path: Path, item_name: str, duration_s: float = 10.0) -> bool:
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        planner = DexTriagePlanner()
        controller = DexTriageController(model, data, planner)
        controller.reset_scene()
        
        # We need to simulate the grasp action over the specified duration
        # using the PhaseState and step logic.
        fps = 30
        steps = int(duration_s * fps)
        
        for step in range(steps):
            state = controller.step(step, steps)
            
            # According to requirement 2.1, we need 3-5 contact points for a valid grasp
            if state["phase"] in {"grasp", "lift"} and 3 <= state["contact_count"] <= 5:
                # Successfully grasped with 3-5 contact points
                return True
                
        return False
        
    except Exception as e:
        print(f"Error executing grasp: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
        item = sys.argv[2] if len(sys.argv) > 2 else "red_emergency_box"
        success = execute_grasp(model_path, item)
        print(f"Grasp Execution {'Succeeded' if success else 'Failed'}")
    else:
        print("Usage: python grasp_execution.py <model_path> [item_name]")
