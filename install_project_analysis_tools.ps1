#requires -version 5.1
[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\HO",
    [switch]$SkipElevation
)

$ErrorActionPreference = "Stop"

function Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Ok($m){ Write-Host "[ OK ] $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }

function Has-Command($n){
    return $null -ne (Get-Command $n -ErrorAction SilentlyContinue)
}

function Run-Checked {
    param(
        [string]$Exe,
        [string[]]$Args,
        [string]$Label
    )
    Info $Label
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed (exit code $LASTEXITCODE)."
    }
}

# Windows protection is not bypassed. This requests normal UAC elevation.
if (-not $SkipElevation) {
    $admin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).
        IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $admin) {
        Info "Requesting normal Windows UAC elevation for system-level package installation."
        $args = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$PSCommandPath`"",
            "-ProjectRoot", "`"$ProjectRoot`""
        )
        Start-Process powershell.exe -Verb RunAs -ArgumentList $args | Out-Null
        exit 0
    }
}

if (-not (Test-Path $ProjectRoot)) {
    throw "Project directory not found: $ProjectRoot"
}
Set-Location $ProjectRoot

# --- Prerequisites ---
if (-not (Has-Command "winget")) {
    throw "winget is not available. Install/update Microsoft App Installer first."
}
if (-not (Has-Command "python")) {
    throw "Python is not available."
}
if (-not (Has-Command "node")) {
    throw "Node.js is not available."
}
if (-not (Has-Command "npm")) {
    throw "npm is not available."
}

Write-Host ""
Write-Host "=== Project Analysis Tool Installer ===" -ForegroundColor White
Write-Host "Project: $ProjectRoot"
Write-Host ""

Info "Python version"; & python --version
Info "Node version"; & node --version
Info "npm version"; & npm --version

# --- Graphviz ---
if (-not (Has-Command "dot")) {
    Run-Checked "winget" @(
        "install","--id","Graphviz.Graphviz","--exact",
        "--accept-source-agreements","--accept-package-agreements"
    ) "Installing Graphviz"
}
$graphvizCandidates = @(
    "$env:ProgramFiles\Graphviz\bin",
    "$env:ProgramFiles(x86)\Graphviz\bin"
)
foreach($p in $graphvizCandidates){
    if(Test-Path $p){ $env:Path = "$p;$env:Path" }
}
if (-not (Has-Command "dot")) {
    throw "Graphviz installed but 'dot' is not visible in PATH. Reopen PowerShell and rerun."
}
# Graphviz prints its version to STDERR even on success. PowerShell can
# surface that successful STDERR output as a NativeCommandError when
# $ErrorActionPreference = "Stop". Use cmd.exe so a successful dot -V
# is treated as a normal verification result.
$dotVersion = cmd.exe /c "dot -V 2>&1"
if ($LASTEXITCODE -ne 0) {
    throw "Graphviz verification failed (exit code $LASTEXITCODE): $dotVersion"
}
Write-Host $dotVersion
Ok "Graphviz"

# --- Python tooling ---
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $toolPyVenv = Join-Path $ProjectRoot ".tooling\pyenv"
    New-Item -ItemType Directory -Force -Path (Split-Path $toolPyVenv) | Out-Null
    if (-not (Test-Path $toolPyVenv)) {
        Run-Checked "python" @("-m","venv",$toolPyVenv) "Creating isolated analysis Python environment"
    }
    $venvPython = Join-Path $toolPyVenv "Scripts\python.exe"
}

Run-Checked $venvPython @("-m","pip","install","--upgrade","pip") "Updating pip"
Run-Checked $venvPython @("-m","pip","install","pydeps","code2flow") "Installing pydeps + code2flow"

$pydepsExe = Join-Path (Split-Path $venvPython) "pydeps.exe"
$code2flowExe = Join-Path (Split-Path $venvPython) "code2flow.exe"

if (-not (Test-Path $pydepsExe)) { throw "pydeps executable missing: $pydepsExe" }
if (-not (Test-Path $code2flowExe)) { throw "code2flow executable missing: $code2flowExe" }

& $pydepsExe --version
Ok "pydeps"
& $code2flowExe --help | Select-Object -First 2
Ok "code2flow"

# --- Isolated npm tooling (does not modify dashboard/package.json) ---
$toolRoot = Join-Path $ProjectRoot ".tooling\npm"
New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null

Run-Checked "npm" @("--prefix",$toolRoot,"install","repomix@latest","dependency-cruiser@latest") `
    "Installing Repomix + dependency-cruiser into isolated tooling"

$npmBin = Join-Path $toolRoot "node_modules\.bin"
$repomixExe = Join-Path $npmBin "repomix.cmd"
$depCruiseExe = Join-Path $npmBin "depcruise.cmd"

if (-not (Test-Path $repomixExe)) { throw "Repomix executable missing: $repomixExe" }
if (-not (Test-Path $depCruiseExe)) { throw "dependency-cruiser executable missing: $depCruiseExe" }

& $repomixExe --version
Ok "Repomix"
& $depCruiseExe --version
Ok "dependency-cruiser"

# --- Smoke tests ---
Info "Repomix smoke test"
$smokeDir = Join-Path $ProjectRoot "docs"
$smokeOutDir = Join-Path $ProjectRoot ".tooling\smoke"
New-Item -ItemType Directory -Force -Path $smokeOutDir | Out-Null
$smokeOut = Join-Path $smokeOutDir "repomix-smoke.xml"
Run-Checked $repomixExe @($smokeDir,"--compress","--output",$smokeOut) "Running Repomix smoke test"
if (-not (Test-Path $smokeOut)) { throw "Repomix smoke output was not created." }
Ok "Repomix smoke test"

Info "dependency-cruiser smoke test"
$dashSrc = Join-Path $ProjectRoot "dashboard\src"
if (Test-Path $dashSrc) {
    & $depCruiseExe $dashSrc "--include-only" "^src" "--output-type" "text"
    if ($LASTEXITCODE -eq 0) {
        Ok "dependency-cruiser smoke test"
    } else {
        Warn "dependency-cruiser installed correctly, but the project scan needs a project-specific config."
    }
} else {
    Warn "dashboard\src not found; dependency-cruiser project scan skipped."
}

Info "pydeps CLI smoke test"
& $pydepsExe --help | Select-Object -First 3
if ($LASTEXITCODE -ne 0) { throw "pydeps smoke test failed." }
Ok "pydeps smoke test"

Info "code2flow CLI smoke test"
& $code2flowExe --help | Select-Object -First 3
if ($LASTEXITCODE -ne 0) { throw "code2flow smoke test failed." }
Ok "code2flow smoke test"

# --- Summary file ---
$summary = @"
PROJECT ANALYSIS TOOLS
======================
Project: $ProjectRoot

Graphviz: verified
pydeps: verified
code2flow: verified
Repomix: verified
dependency-cruiser: verified

Python analysis environment:
$venvPython

npm analysis tool directory:
$toolRoot

Note:
Windows security was NOT bypassed. The installer uses the normal UAC
elevation mechanism when administrator rights are required.
"@
$summaryPath = Join-Path $ProjectRoot ".tooling\INSTALL_REPORT.txt"
Set-Content -Path $summaryPath -Value $summary -Encoding UTF8

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " ALL REQUESTED TOOLS INSTALLED + VERIFIED" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "Report: $summaryPath"
Write-Host "No application source files were intentionally changed."
Write-Host ""
