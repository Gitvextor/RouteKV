param(
    [string]$PackageRoot = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $PackageRoot 'PACKAGE_SHA256.csv'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing package manifest: $manifestPath"
}

$rows = Import-Csv -LiteralPath $manifestPath
$errors = [System.Collections.Generic.List[string]]::new()
foreach ($row in $rows) {
    $path = Join-Path $PackageRoot $row.RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $errors.Add("MISSING $($row.RelativePath)")
        continue
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $row.SHA256.ToLowerInvariant()) {
        $errors.Add("HASH_MISMATCH $($row.RelativePath)")
    }
}

if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    throw "Package verification failed with $($errors.Count) error(s)."
}

Write-Host "PACKAGE_VERIFIED files=$($rows.Count)" -ForegroundColor Green
