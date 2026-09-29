param(
    [Parameter(Mandatory=$true)][string]$Base,
    [Parameter(Mandatory=$true)][string]$Key,
    [switch]$ExecuteFailover,
    [switch]$SyncB2,
    [switch]$PruneB2Extra
)

$ErrorActionPreference = "Stop"
$headers = @{ "X-API-KEY" = $Key }

function Invoke-InfraGet([string]$Path) {
    Invoke-RestMethod -Method GET -Uri "$Base$Path" -Headers $headers
}

function Invoke-InfraPost([string]$Path, $Body) {
    Invoke-RestMethod -Method POST -Uri "$Base$Path" -Headers $headers -ContentType "application/json" -Body ($Body | ConvertTo-Json -Depth 20)
}

Write-Host "=== control plane health ===" -ForegroundColor Cyan
$health = Invoke-InfraGet "/infra/health"
$health | ConvertTo-Json -Depth 20

Write-Host "`n=== status ===" -ForegroundColor Cyan
$status = Invoke-InfraGet "/infra/status"
$status | ConvertTo-Json -Depth 20

Write-Host "`n=== postgres health ===" -ForegroundColor Cyan
foreach ($slot in $status.available.postgres) {
    $r = Invoke-InfraGet "/infra/postgres/$slot/health"
    $r | ConvertTo-Json -Depth 10
    if (-not $r.healthy) { throw "Postgres health failed: $slot" }
}

Write-Host "`n=== paired Render slots ===" -ForegroundColor Cyan
foreach ($slot in $status.available.render) {
    foreach ($role in @("n8n", "sh01")) {
        $service = Invoke-InfraGet "/infra/render/$slot/$role/status"
        $service | ConvertTo-Json -Depth 10
        $r = Invoke-InfraGet "/infra/render/$slot/$role/health"
        $r | ConvertTo-Json -Depth 10
        if (-not $r.healthy) { Write-Warning "$slot/$role is not currently healthy (may be intentionally suspended)" }
    }
}

if ($status.available.b2.Count -gt 0) {
    Write-Host "`n=== B2 health ===" -ForegroundColor Cyan
    foreach ($slot in $status.available.b2) {
        $r = Invoke-InfraGet "/infra/b2/$slot/health"
        $r | ConvertTo-Json -Depth 10
        if (-not $r.healthy) { throw "B2 health failed: $slot" }
    }
}

foreach ($role in @("n8n", "sh01")) {
    Write-Host "`n=== $role Cloudflare router ===" -ForegroundColor Cyan
    $router = Invoke-InfraGet "/infra/router/$role/status"
    $router | ConvertTo-Json -Depth 10
    if (-not $router.persistence_available) { throw "$role Worker ROUTER_STATE KV binding is not available" }
    if ($router.maintenance) { throw "$role Worker is unexpectedly in maintenance mode" }
}

Write-Host "`n=== failover dry run ===" -ForegroundColor Cyan
$dryBody = @{
    sync_b2 = [bool]$SyncB2
    prune_b2_extra = [bool]$PruneB2Extra
    switch_router = $true
    quiesce_source = $true
    dry_run = $true
}
$dry = Invoke-InfraPost "/infra/failover" $dryBody
$dry | ConvertTo-Json -Depth 20

if (-not $ExecuteFailover) {
    Write-Host "`nDry-run suite completed. Add -ExecuteFailover only after reviewing the plan." -ForegroundColor Green
    exit 0
}

Write-Host "`n=== EXECUTING FULL PAIRED RENDER FAILOVER ===" -ForegroundColor Yellow
$liveBody = @{
    sync_b2 = [bool]$SyncB2
    prune_b2_extra = [bool]$PruneB2Extra
    switch_router = $true
    quiesce_source = $true
    dry_run = $false
}
$accepted = Invoke-InfraPost "/infra/failover" $liveBody
$accepted | ConvertTo-Json -Depth 20
$jobId = $accepted.job.id
if (-not $jobId) { throw "No failover job id returned" }

while ($true) {
    Start-Sleep -Seconds 5
    $job = Invoke-InfraGet "/infra/jobs/$jobId"
    Write-Host "status=$($job.status) updated=$($job.updated_at)"
    if ($job.status -eq "succeeded") {
        $job | ConvertTo-Json -Depth 30
        break
    }
    if ($job.status -eq "failed") {
        $job | ConvertTo-Json -Depth 30
        throw "Failover job failed"
    }
}

Write-Host "`n=== final status ===" -ForegroundColor Cyan
$final = Invoke-InfraGet "/infra/status"
$final | ConvertTo-Json -Depth 20

foreach ($role in @("n8n", "sh01")) {
    $finalRouter = Invoke-InfraGet "/infra/router/$role/status"
    if ($finalRouter.maintenance) { throw "$role router was left in maintenance mode" }
}

Write-Host "`nFull paired Render failover completed successfully." -ForegroundColor Green
