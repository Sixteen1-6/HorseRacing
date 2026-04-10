Stop-Process -Name chrome -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Use a copy of the default profile to avoid lock conflicts
$debugProfile = "$env:TEMP\chrome-debug-profile"

# Copy cookies from real profile if debug profile doesn't exist yet
if (-not (Test-Path $debugProfile)) {
    Write-Output "Creating debug profile (first time only)..."
    New-Item -ItemType Directory -Path $debugProfile -Force | Out-Null
    $defaultProfile = "$env:LOCALAPPDATA\Google\Chrome\User Data\Default"
    if (Test-Path "$defaultProfile\Cookies") {
        Copy-Item "$defaultProfile\Cookies" "$debugProfile\" -ErrorAction SilentlyContinue
    }
}

Write-Output "Starting Chrome with debug port and separate profile..."
$chromeExe = "C:\Program Files\Google\Chrome\Application\chrome.exe"
Start-Process -FilePath $chromeExe -ArgumentList "--remote-debugging-port=9222", "--user-data-dir=$debugProfile", "--no-first-run"
Start-Sleep -Seconds 8

Write-Output "Checking port 9222..."
$portCheck = netstat -ano | Select-String ":9222"
if ($portCheck) {
    Write-Output "Port found:"
    Write-Output $portCheck
} else {
    Write-Output "Port 9222 NOT found"

    # Check ALL Chrome processes and their ports
    Write-Output "`nAll Chrome processes:"
    Get-Process chrome -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Output "  PID: $($_.Id)"
    }

    Write-Output "`nAll LISTENING ports from chrome processes:"
    $chromePids = (Get-Process chrome -ErrorAction SilentlyContinue).Id
    foreach ($pid in $chromePids) {
        $lines = netstat -ano | Select-String "$pid" | Select-String "LISTENING"
        if ($lines) {
            Write-Output "  PID ${pid}:"
            Write-Output $lines
        }
    }
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:9222/json/version" -UseBasicParsing -TimeoutSec 5
    Write-Output "`nSUCCESS: $($response.Content)"
} catch {
    Write-Output "`nFAILED to connect: $($_.Exception.Message)"
}
