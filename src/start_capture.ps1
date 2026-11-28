param([string]$Config = 'config/capture_ui.local.json')
$ErrorActionPreference = 'Stop'
$captureRepo = Split-Path $PSScriptRoot -Parent
$captureConfigPath = (Resolve-Path -LiteralPath $Config).Path
$captureConfigRoot = Split-Path $captureConfigPath -Parent
$captureConfig = Get-Content -LiteralPath $captureConfigPath -Raw | ConvertFrom-Json

function Resolve-CaptureExecutable([string]$Executable) {
    if ([System.IO.Path]::IsPathRooted($Executable)) {
        return (Resolve-Path -LiteralPath $Executable).Path
    }
    if ($Executable.Contains('/') -or $Executable.Contains('\')) {
        return (Resolve-Path -LiteralPath (Join-Path $captureConfigRoot $Executable)).Path
    }
    return (Get-Command -Name $Executable -CommandType Application -ErrorAction Stop).Source
}

$capturePython = Resolve-CaptureExecutable $captureConfig.pythonw_path
$captureFfmpeg = Resolve-CaptureExecutable $captureConfig.ffmpeg_path
$captureReadyPath = [System.IO.Path]::GetFullPath((Join-Path $captureConfigRoot $captureConfig.ready_file))
$captureLogRoot = Join-Path $captureRepo 'artifacts/hardware'
New-Item -ItemType Directory -Path $captureLogRoot -Force | Out-Null
$captureLogStem = [System.IO.Path]::GetFileNameWithoutExtension($captureConfigPath)
$capturePythonArgs = @('-s','-m','src.capture_ui','--config',('"{0}"' -f $captureConfigPath))
$captureUi = Start-Process -FilePath $capturePython -ArgumentList $capturePythonArgs -WorkingDirectory $captureRepo -WindowStyle Normal -RedirectStandardOutput (Join-Path $captureLogRoot ($captureLogStem + '.stdout.log')) -RedirectStandardError (Join-Path $captureLogRoot ($captureLogStem + '.stderr.log')) -PassThru
$captureReady = $null
for ($captureAttempt = 0; $captureAttempt -lt 40; $captureAttempt++) {
    Start-Sleep -Milliseconds 250
    if (Test-Path -LiteralPath $captureReadyPath) {
        try {
            $captureCandidate = Get-Content -LiteralPath $captureReadyPath -Raw | ConvertFrom-Json
            if ($captureCandidate.pid -eq $captureUi.Id) { $captureReady = $captureCandidate; break }
        } catch { }
    }
    if ($captureUi.HasExited) { throw ('Capture UI exited; see ' + (Join-Path $captureLogRoot ($captureLogStem + '.stderr.log'))) }
}
if ($null -eq $captureReady) { throw 'Capture UI did not open its local frame receiver' }
$captureSize = '{0}x{1}' -f $captureConfig.frame_wh[0],$captureConfig.frame_wh[1]
$captureNativeArgs = @('-hide_banner','-nostdin','-loglevel','warning','-f','dshow','-rtbufsize','128M','-video_size',$captureSize,'-framerate',[string]$captureConfig.requested_fps,'-vcodec','mjpeg','-i',('"video={0}"' -f $captureConfig.device_name),'-an','-fps_mode','passthrough','-pix_fmt','bgr24','-f','rawvideo',('tcp://127.0.0.1:{0}' -f $captureReady.port))
$captureSource = Start-Process -FilePath $captureFfmpeg -ArgumentList $captureNativeArgs -WorkingDirectory $captureRepo -WindowStyle Hidden -RedirectStandardError (Join-Path $captureReady.session_root 'native_source.log') -PassThru
@{ui_pid=$captureUi.Id; source_pid=$captureSource.Id; session_root=$captureReady.session_root; port=$captureReady.port} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $captureReady.session_root 'processes.json') -Encoding utf8
Write-Output ('Capture window ready. Session: ' + $captureReady.session_root)
