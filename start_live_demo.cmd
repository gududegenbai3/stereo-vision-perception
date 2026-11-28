@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
if /i "%~1"=="--native-window" goto native_window

if not defined STEREO_PYTHON for /f "delims=" %%P in ('where.exe python.exe') do if not defined STEREO_PYTHON set "STEREO_PYTHON=%%P"
if not defined STEREO_PYTHON (
  echo Activate the project Python environment, then run this launcher.
  pause
  exit /b 1
)
set "STEREO_RESULTS=%~dp0artifacts\live_demo"
set "STEREO_LOG=%STEREO_RESULTS%\live-startup.log"
if /i "%~1"=="--saved-replay" set "STEREO_LOG=%STEREO_RESULTS%\replay-startup.log"
if not exist "%STEREO_RESULTS%" mkdir "%STEREO_RESULTS%" || exit /b 1
if not exist "%~dp0src\live_demo.py" (
  echo Run this launcher from the project checkout.
  pause
  exit /b 1
)
if not exist "%STEREO_PYTHON%" (
  > "%STEREO_LOG%" echo Local Python not found: "%STEREO_PYTHON%"
  type "%STEREO_LOG%"
  pause
  exit /b 1
)
pushd "%~dp0" || exit /b 1
if /i "%~1"=="--saved-replay" goto replay
if /i "%~1"=="--direct-camera" goto direct_camera

rem Ask the existing app to print its capture configuration and a new run path.
call :prepare_native > "%STEREO_LOG%" 2>&1
if errorlevel 1 goto show_error
echo Starting StereoDemo...
start "" /b "%ComSpec%" /d /c call "%~f0" --native-window
if errorlevel 1 goto show_error
set "STEREO_WAIT=0"

:wait_for_receiver
if exist "%STEREO_RUN_DIR%\window-exit.code" goto native_exit
if exist "%STEREO_RUN_DIR%\receiver.json" goto start_camera
set /a STEREO_WAIT+=1
if %STEREO_WAIT% geq 30 (
  >> "%STEREO_LOG%" echo Camera receiver did not start within 30 seconds.
  goto show_error
)
call :delay
goto wait_for_receiver

:start_camera
for /f %%I in ('call "%STEREO_PYTHON%" -E -s -B -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "%STEREO_RUN_DIR%\receiver.json" port') do set "STEREO_TCP_PORT=%%I"
if not defined STEREO_TCP_PORT goto show_error

:camera_retry
if exist "%STEREO_RUN_DIR%\window-exit.code" goto native_exit
>> "%STEREO_RUN_DIR%\camera.stderr.log" echo [%time%] Native FFmpeg opening %STEREO_FRAME_WIDTH%x%STEREO_FRAME_HEIGHT%
"%STEREO_FFMPEG%" -hide_banner -nostdin -loglevel warning -f dshow -rtbufsize 128M -video_size %STEREO_FRAME_WIDTH%x%STEREO_FRAME_HEIGHT% -framerate %STEREO_FRAME_RATE% -vcodec mjpeg -i "video=%STEREO_DEVICE%" -an -vf "settb=expr=1/%STEREO_FRAME_RATE%,setpts=N" -enc_time_base 1:%STEREO_FRAME_RATE% -fps_mode passthrough -pix_fmt bgr24 -f rawvideo "tcp://127.0.0.1:%STEREO_TCP_PORT%" >> "%STEREO_RUN_DIR%\camera.stderr.log" 2>&1
call :delay
goto camera_retry

:native_exit
set /p STEREO_EXIT=<"%STEREO_RUN_DIR%\window-exit.code"
if not "%STEREO_EXIT%"=="0" goto show_error
popd
exit /b 0

:replay
set "STEREO_MODE=--saved-replay"
goto run_managed

:direct_camera
set "STEREO_MODE=--direct-camera"

:run_managed
echo Starting StereoDemo...
"%STEREO_PYTHON%" -E -s -B -X utf8 -m src.live_demo --output-root "%STEREO_RESULTS%" %STEREO_MODE% > "%STEREO_LOG%" 2>&1
set "STEREO_EXIT=%ERRORLEVEL%"
if not "%STEREO_EXIT%"=="0" goto show_error
popd
exit /b 0

:prepare_native
set "STEREO_RUN_DIR="
for /f "tokens=1,* delims==" %%K in ('call "%STEREO_PYTHON%" -E -s -B -X utf8 -m src.live_demo --output-root "%STEREO_RESULTS%" --native-launch-settings') do set "%%K=%%L"
if not defined STEREO_RUN_DIR exit /b 1
echo Run folder "%STEREO_RUN_DIR%"
exit /b 0

:delay
"%STEREO_PYTHON%" -E -s -B -c "from time import sleep; sleep(1)"
exit /b 0

:native_window
"%STEREO_PYTHON%" -E -s -B -X utf8 -m src.live_demo --run-dir "%STEREO_RUN_DIR%" >> "%STEREO_LOG%" 2>&1
> "%STEREO_RUN_DIR%\window-exit.tmp" echo %ERRORLEVEL%
move /y "%STEREO_RUN_DIR%\window-exit.tmp" "%STEREO_RUN_DIR%\window-exit.code" >nul
exit /b 0

:show_error
popd
type "%STEREO_LOG%"
echo Startup log: "%STEREO_LOG%"
pause
exit /b 1
