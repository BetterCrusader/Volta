param(
    [string]$OutDir = "packaging/windows/assets/generated",
    [string]$TokensPath = "packaging/windows/design-tokens.json"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
$outPath = Join-Path $repoRoot $OutDir
New-Item -ItemType Directory -Force -Path $outPath | Out-Null

$tokensFile = Join-Path $repoRoot $TokensPath
$tokens = $null
if (Test-Path $tokensFile) {
    try {
        $tokens = Get-Content -Raw -Path $tokensFile | ConvertFrom-Json
    }
    catch {
        throw "Failed to parse design tokens at $tokensFile. $_"
    }
}

function Resolve-Token {
    param(
        [string]$TokenValue,
        [string]$Fallback
    )
    if ([string]::IsNullOrWhiteSpace($TokenValue)) {
        return $Fallback
    }
    return $TokenValue
}

$bgStart = Resolve-Token -TokenValue $tokens.palette.background_start -Fallback "#060912"
$bgEnd = Resolve-Token -TokenValue $tokens.palette.background_end -Fallback "#0A1530"
$gridGlow = Resolve-Token -TokenValue $tokens.palette.grid_glow -Fallback "#00A0FF"
$titleColor = Resolve-Token -TokenValue $tokens.palette.title -Fallback "#E8F4FF"
$subtitleColor = Resolve-Token -TokenValue $tokens.palette.subtitle -Fallback "#78B9E1"
$brandTitle = Resolve-Token -TokenValue $tokens.branding.title -Fallback "VOLTA"
$brandSubtitle = Resolve-Token -TokenValue $tokens.branding.subtitle -Fallback "DETERMINISTIC INSTALLER"

function New-VoltaBitmap {
    param(
        [int]$Width,
        [int]$Height,
        [string]$FilePath,
        [switch]$DenseGrid
    )

    $bmp = New-Object System.Drawing.Bitmap($Width, $Height)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

    $rect = New-Object System.Drawing.Rectangle(0, 0, $Width, $Height)
    $c1 = [System.Drawing.ColorTranslator]::FromHtml($bgStart)
    $c2 = [System.Drawing.ColorTranslator]::FromHtml($bgEnd)
    $bg = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, $c1, $c2, 45.0)
    $g.FillRectangle($bg, $rect)

    $gridBase = [System.Drawing.ColorTranslator]::FromHtml($gridGlow)
    $gridColor = [System.Drawing.Color]::FromArgb(22, $gridBase.R, $gridBase.G, $gridBase.B)
    $gridPen = New-Object System.Drawing.Pen($gridColor, 1)
    $step = if ($DenseGrid) { 18 } else { 24 }
    for ($x = 0; $x -le $Width; $x += $step) {
        $g.DrawLine($gridPen, $x, 0, $x, $Height)
    }
    for ($y = 0; $y -le $Height; $y += $step) {
        $g.DrawLine($gridPen, 0, $y, $Width, $y)
    }

    $glowBrush = New-Object System.Drawing.Drawing2D.GraphicsPath
    $glowBrush.AddEllipse([int]($Width * 0.2), [int]($Height * 0.1), [int]($Width * 0.6), [int]($Height * 0.75))
    $pathBrush = New-Object System.Drawing.Drawing2D.PathGradientBrush($glowBrush)
    $pathBrush.CenterColor = [System.Drawing.Color]::FromArgb(90, 0, 200, 255)
    $pathBrush.SurroundColors = @([System.Drawing.Color]::FromArgb(0, 0, 0, 0))
    $g.FillEllipse($pathBrush, [int]($Width * 0.18), [int]($Height * 0.08), [int]($Width * 0.64), [int]($Height * 0.8))

    $boltPenOuter = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(120, 0, 150, 255), 7)
    $boltPenOuter.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $boltPenOuter.EndCap = [System.Drawing.Drawing2D.LineCap]::Round

    $boltPenInner = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(220, 255, 255, 255), 2)
    $boltPenInner.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $boltPenInner.EndCap = [System.Drawing.Drawing2D.LineCap]::Round

    [System.Drawing.Point[]]$points = @(
        (New-Object System.Drawing.Point([int]($Width * 0.58), [int]($Height * 0.08))),
        (New-Object System.Drawing.Point([int]($Width * 0.44), [int]($Height * 0.42))),
        (New-Object System.Drawing.Point([int]($Width * 0.58), [int]($Height * 0.42))),
        (New-Object System.Drawing.Point([int]($Width * 0.36), [int]($Height * 0.88)))
    )

    $g.DrawLines($boltPenOuter, $points)
    $g.DrawLines($boltPenInner, $points)

    $titleFont = New-Object System.Drawing.Font("Segoe UI", [float]([math]::Max(10, $Width / 16)), [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Consolas", [float]([math]::Max(7, $Width / 30)), [System.Drawing.FontStyle]::Regular)

    $titleBase = [System.Drawing.ColorTranslator]::FromHtml($titleColor)
    $subtitleBase = [System.Drawing.ColorTranslator]::FromHtml($subtitleColor)
    $titleBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(230, $titleBase.R, $titleBase.G, $titleBase.B))
    $subBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(210, $subtitleBase.R, $subtitleBase.G, $subtitleBase.B))

    $g.DrawString($brandTitle, $titleFont, $titleBrush, [float]($Width * 0.08), [float]($Height * 0.72))
    $g.DrawString($brandSubtitle, $subFont, $subBrush, [float]($Width * 0.08), [float]($Height * 0.85))

    $bmp.Save($FilePath, [System.Drawing.Imaging.ImageFormat]::Bmp)

    $titleFont.Dispose()
    $subFont.Dispose()
    $titleBrush.Dispose()
    $subBrush.Dispose()
    $boltPenOuter.Dispose()
    $boltPenInner.Dispose()
    $gridPen.Dispose()
    $pathBrush.Dispose()
    $glowBrush.Dispose()
    $bg.Dispose()
    $g.Dispose()
    $bmp.Dispose()
}

$header = Join-Path $outPath "header.bmp"
$welcome = Join-Path $outPath "welcome.bmp"

New-VoltaBitmap -Width 150 -Height 57 -FilePath $header
New-VoltaBitmap -Width 164 -Height 314 -FilePath $welcome -DenseGrid

Write-Host "[windows-assets] generated: $header"
Write-Host "[windows-assets] generated: $welcome"
