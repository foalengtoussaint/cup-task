# Build the paper on Windows (no `make` here -- see Makefile for the Linux flow).
#
#   .\paper\build.ps1            compile main.tex -> main.pdf
#   .\paper\build.ps1 -Tables    regenerate tables/*.tex from the CSVs first
#
# Toolchain: tectonic (single binary in ~/bin, self-fetches TeX packages) and
# the Anaconda python, which is the only interpreter here with pandas.
param([switch]$Tables)

$ErrorActionPreference = "Stop"
$paper = $PSScriptRoot
$repo = Split-Path $paper -Parent
$tectonic = Join-Path $env:USERPROFILE "bin\tectonic.exe"
$python = Join-Path $env:USERPROFILE "anaconda3\python.exe"

if (-not (Test-Path $tectonic)) { throw "tectonic not found at $tectonic" }

if ($Tables) {
    Write-Host "== regenerating tables/*.tex ==" -ForegroundColor Cyan
    & $python (Join-Path $paper "scripts\make_tables_tex.py")
    if (-not $?) { throw "make_tables_tex.py failed" }
}

Write-Host "== compiling main.tex ==" -ForegroundColor Cyan
Push-Location $paper
try { & $tectonic -X compile main.tex } finally { Pop-Location }

# Surface the drafting markers -- nothing in \val may survive into a submission.
$tex = Get-Content (Join-Path $paper "main.tex") -Raw
$todo = ([regex]::Matches($tex, '\\todo\{')).Count
$val = ([regex]::Matches($tex, '\\val\{')).Count
Write-Host "`nmain.pdf written. drafting markers left: $todo \todo, $val \val" -ForegroundColor Yellow
