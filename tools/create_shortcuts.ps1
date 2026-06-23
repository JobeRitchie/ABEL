<#
.SYNOPSIS
    Create ABEL shortcuts (Start Menu + Desktop) that launch run_abel.bat with
    the ABEL icon.

.DESCRIPTION
    A .bat file cannot carry its own icon, so this creates Windows .lnk
    shortcuts pointing at run_abel.bat with IconLocation set to abel.ico.
    Paths are resolved from this script's own location, so it works wherever the
    repo lives. Re-running it simply refreshes the shortcuts (idempotent).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\create_shortcuts.ps1
#>

$ErrorActionPreference = 'Stop'

# Repo root = parent of the tools\ folder this script lives in.
$repoRoot = Split-Path -Parent $PSScriptRoot
$target   = Join-Path $repoRoot 'run_abel.bat'
$icon     = Join-Path $repoRoot 'abel\ui\assets\abel.ico'

if (-not (Test-Path $target)) { throw "Launcher not found: $target" }
if (-not (Test-Path $icon))   { throw "Icon not found: $icon. Run tools\make_icon.py first." }

$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\ABEL.lnk'
$desktop   = Join-Path ([Environment]::GetFolderPath('Desktop')) 'ABEL.lnk'

$shell = New-Object -ComObject WScript.Shell
foreach ($linkPath in @($startMenu, $desktop)) {
    $lnk = $shell.CreateShortcut($linkPath)
    $lnk.TargetPath       = $target
    $lnk.WorkingDirectory = $repoRoot
    $lnk.IconLocation     = "$icon,0"
    $lnk.Description       = 'ABEL - Active-learning Behavior Estimation and Labeling'
    $lnk.Save()
    Write-Host "Created shortcut: $linkPath"
}

Write-Host "Done. ABEL is now in the Start Menu and on the Desktop."
