"""cup-task: offline drink-task scoring from multi-camera video.

Pipeline (offline for now; inference is real-time-capable):
    clips -> cup detection (existing model)        cup_task.cup_detect   [TODO]
          -> body/hand keypoints (YOLO-pose)       cup_task.pose_keypoints
          -> 3D triangulation (existing calib)     cup_task.triangulate  [TODO]
          -> phase segmentation                    cup_task.segment      [TODO]
          -> iMOVE metrics / scoring               cup_task.score        [TODO]

Metric definitions come from iMOVE (Docker on another machine) and are not yet
wired in — scoring is a placeholder until that spec is available.
"""
__version__ = "0.0.1"
