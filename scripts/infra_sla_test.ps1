param(
    [Parameter(Mandatory=$true)][string]$Base,
    [Parameter(Mandatory=$true)][string]$Key,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$headers = @{ "X-API-KEY" = $Key }

Write-Host "=== SLA status ===" -ForegroundColor Cyan
Invoke-RestMethod -Method GET -Uri "$Base/infra/sla/status" -Headers $headers |
    ConvertTo-Json -Depth 30

Write-Host "`n=== SLA dry-run tick ===" -ForegroundColor Cyan
$dry = Invoke-RestMethod `
    -Method POST `
    -Uri "$Base/infra/sla/tick" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body (@{ dry_run = $true } | ConvertTo-Json)
$dry | ConvertTo-Json -Depth 30

if (-not $Execute) {
    Write-Host "`nDry-run complete. Add -Execute for a real SLA tick." -ForegroundColor Green
    exit 0
}

Write-Host "`n=== REAL SLA tick ===" -ForegroundColor Yellow
$accepted = Invoke-RestMethod `
    -Method POST `
    -Uri "$Base/infra/sla/tick" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body (@{ dry_run = $false } | ConvertTo-Json)
$accepted | ConvertTo-Json -Depth 20

$jobId = $accepted.job.id
if (-not $jobId) { throw "No SLA job id returned" }

while ($true) {
    Start-Sleep -Seconds 5
    $job = Invoke-RestMethod -Method GET -Uri "$Base/infra/jobs/$jobId" -Headers $headers
    Write-Host "status=$($job.status) updated=$($job.updated_at)"
    if ($job.status -eq "succeeded") {
        $job | ConvertTo-Json -Depth 40
        break
    }
    if ($job.status -eq "failed") {
        $job | ConvertTo-Json -Depth 40
        throw "SLA tick failed"
    }
}
