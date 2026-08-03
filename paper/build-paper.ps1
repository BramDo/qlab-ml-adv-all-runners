$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

New-Item -ItemType Directory -Force -Path "build" | Out-Null

$stageRoot = Join-Path $env:TEMP "ml-adv-paper-build"
if (Test-Path $stageRoot) {
    Remove-Item -Recurse -Force $stageRoot
}
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stageRoot "build") | Out-Null

Copy-Item "main.tex" $stageRoot -Force
Copy-Item "references.bib" $stageRoot -Force
Get-ChildItem -File | Where-Object { $_.Extension -in @(".png", ".pdf") } | Copy-Item -Destination $stageRoot -Force

Push-Location $stageRoot
try {
    pdflatex -interaction=nonstopmode -output-directory=build main.tex
    bibtex build/main
    pdflatex -interaction=nonstopmode -output-directory=build main.tex
    pdflatex -interaction=nonstopmode -output-directory=build main.tex
    Copy-Item (Join-Path $stageRoot "build\\*") (Join-Path $here "build") -Force
}
finally {
    Pop-Location
}

Write-Host "Built PDF: $here\\build\\main.pdf"

