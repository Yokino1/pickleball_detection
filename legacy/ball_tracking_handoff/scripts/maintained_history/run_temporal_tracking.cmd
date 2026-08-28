@echo off
setlocal

call "D:\anacondaa\Scripts\activate.bat" "D:\anacondaa\envs\torch-cu128"
cd /d "D:\ball\ball_tracking_handoff\ball_tracking_handoff"

set "YOLO_CONFIG_DIR=D:\ball\.cache\ultralytics"
set "TORCH_HOME=D:\ball\.cache\torch"
set "PIP_CACHE_DIR=D:\ball\.cache\pip"

python apps\track_video.py ^
  --config legacy\ball_tracking_handoff\configs\maintained_history\tracking_temporal_r1.yaml ^
  --input data\sideview_raw ^
  --output-dir outputs\experiments\history\temporal_r1 ^
  --skip-existing

endlocal
