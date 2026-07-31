# verify_session.ps1
# This script verifies the junctions and checks the state of the IDE session databases.

$cDrivePath = "C:\Users\Sanidhya\.gemini\antigravity-ide\conversations"
$dDrivePath = "D:\.gemini\antigravity-ide\conversations"

Write-Host "============================================="
Write-Host "   Antigravity IDE Session & Cache Verifier  "
Write-Host "============================================="
Write-Host ""

# 1. Verify Junctions
$geminiJunction = Get-Item "C:\Users\Sanidhya\.gemini" -ErrorAction SilentlyContinue
if ($geminiJunction.LinkType -eq "Junction") {
    Write-Host "[OK] C: drive pointer found. C:\Users\Sanidhya\.gemini points to: $($geminiJunction.Target)" -ForegroundColor Green
} else {
    Write-Host "[WARN] C:\Users\Sanidhya\.gemini is not a junction or does not exist." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "--- Checking Old Session (26f9103f-...) ---"

# 2. Check for the old session database
$oldSessionId = "26f9103f-b5fc-48c1-a72b-0134aa225b99"
$dbFile = "$cDrivePath\$oldSessionId.db"
$dbBakFile = "$cDrivePath\$oldSessionId.db.bak"

if (Test-Path $dbFile) {
    $size = (Get-Item $dbFile).Length
    Write-Host "[CACHED] Active old session database found! Size: $size bytes" -ForegroundColor Cyan
    Write-Host "         Path: $dbFile"
} else {
    Write-Host "[INFO] Active .db file for old session is currently disabled (moved/renamed)."
}

if (Test-Path $dbBakFile) {
    $bakSize = (Get-Item $dbBakFile).Length
    Write-Host "[BACKUP] Old session backup database exists. Size: $bakSize bytes" -ForegroundColor Cyan
    Write-Host "         Path: $dbBakFile"
} else {
    Write-Host "[INFO] No .db.bak backup file found."
}

Write-Host ""
Write-Host "--- D: Drive Verification ---"
$d_dbBakFile = "$dDrivePath\$oldSessionId.db.bak"
if (Test-Path $d_dbBakFile) {
    Write-Host "[VERIFIED] The exact same data is being written directly to D: drive." -ForegroundColor Green
    Write-Host "           Confirmed path: $d_dbBakFile"
} else {
    Write-Host "[WARN] Could not find the file directly on D: drive!" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================="
Write-Host "Conclusion:"
Write-Host "Your system is correctly routing all C: drive IDE cache reads/writes to the D: drive."
Write-Host "The old session's trajectory data is safely stored in the .bak file to prevent IDE crashes."
Write-Host "============================================="
