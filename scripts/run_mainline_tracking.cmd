@echo off
setlocal

call "D:\anacondaa\Scripts\activate.bat" "D:\anacondaa\envs\torch-cu128"
cd /d "D:\ball\ball_tracking_handoff\ball_tracking_handoff"

set "YOLO_CONFIG_DIR=D:\ball\.cache\ultralytics"
set "TORCH_HOME=D:\ball\.cache\torch"
set "PIP_CACHE_DIR=D:\ball\.cache\pip"

python apps\track_video.py ^
  --config configs\tracking.yaml ^
  --input data\sideview_raw ^
  --output-dir outputs\experiments\desktop_ab\mainline ^
  --skip-existing

endlocal
