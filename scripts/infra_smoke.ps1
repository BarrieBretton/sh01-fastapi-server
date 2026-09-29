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
    Invoke-RestMethod `
        -Method POST `
        -Uri "$Base$Path" `
        -Headers $headers `
        -ContentType "application/json" `
        -Body ($Body | ConvertTo-Json -Depth 20)
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

Write-Host "`n=== n8n Render ring ===" -ForegroundColor Cyan
foreach ($slot in $status.available.render.n8n) {
    $service = Invoke-InfraGet "/infra/render/$slot/status"
    $service | ConvertTo-Json -Depth 10
    $r = Invoke-InfraGet "/infra/render/$slot/health"
    $r | ConvertTo-Json -Depth 10
    if (-not $r.healthy) { Write-Warning "n8n Render slot is not currently healthy (may be intentionally suspended): $slot" }
}

Write-Host "`n=== SH01 Render ring ===" -ForegroundColor Cyan
foreach ($slot in $status.available.render.sh01) {
    $service = Invoke-InfraGet "/infra/render/$slot/status"
    $service | ConvertTo-Json -Depth 10
    $r = Invoke-InfraGet "/infra/render/$slot/health"
    $r | ConvertTo-Json -Depth 10
    if (-not $r.healthy) { Write-Warning "SH01 Render slot is not currently healthy (may be intentionally suspended): $slot" }
}

if ($status.available.b2.Count -gt 0) {
    Write-Host "`n=== B2 health ===" -ForegroundColor Cyan
    foreach ($slot in $status.available.b2) {
        $r = Invoke-InfraGet "/infra/b2/$slot/health"
        $r | ConvertTo-Json -Depth 10
        if (-not $r.healthy) { throw "B2 health failed: $slot" }
    }
}

Write-Host "`n=== n8n Cloudflare router ===" -ForegroundColor Cyan
$n8nRouter = Invoke-InfraGet "/infra/router/n8n/status"
$n8nRouter | ConvertTo-Json -Depth 10
if (-not $n8nRouter.persistence_available) {
    throw "n8n Worker ROUTER_STATE KV binding is not available"
}
if ($n8nRouter.maintenance) {
    throw "n8n Worker is unexpectedly in maintenance mode"
}

Write-Host "`n=== SH01 Cloudflare router ===" -ForegroundColor Cyan
try {
    $sh01Router = Invoke-InfraGet "/infra/router/sh01/status"
    $sh01Router | ConvertTo-Json -Depth 10
    if (-not $sh01Router.persistence_available) {
        Write-Warning "SH01 Worker ROUTER_STATE KV binding is not available"
    }
} catch {
    Write-Warning "SH01 router control is not configured yet: $($_.Exception.Message)"
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

Write-Host "`n=== EXECUTING FULL N8N STACK FAILOVER ===" -ForegroundColor Yellow
Write-Host "The n8n Worker will enter maintenance mode while the active writer is suspended and Postgres is copied." -ForegroundColor Yellow
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

Write-Host "Polling job $jobId ..." -ForegroundColor Cyan
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

Write-Host "`n=== final n8n router ===" -ForegroundColor Cyan
$finalRouter = Invoke-InfraGet "/infra/router/n8n/status"
$finalRouter | ConvertTo-Json -Depth 10
if ($finalRouter.maintenance) { throw "n8n router was left in maintenance mode" }

Write-Host "`nFull n8n stack failover completed successfully." -ForegroundColor Green
