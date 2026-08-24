"""cup-task: offline drink-task scoring from multi-camera video.

Pipeline (offline for now; inference is real-time-capable):
    clips -> cup detection (existing model)        pipeline.cup_detect   [TODO]
          -> body/hand keypoints (YOLO-pose)       pipeline.pose_keypoints
          -> 3D triangulation (existing calib)     pipeline.triangulate  [TODO]
          -> phase segmentation                    pipeline.segment      [TODO]
          -> iMOVE metrics / scoring               pipeline.score        [TODO]

Metric definitions come from iMOVE (Docker on another machine) and are not yet
wired in — scoring is a placeholder until that spec is available.
"""
__version__ = "0.0.1"
