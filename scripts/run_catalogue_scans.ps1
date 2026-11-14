# Windows PowerShell 5.1+. No installations, environment changes or deletion actions.
param([ValidateRange(1,10)][int]$Rounds = 1)
$ErrorActionPreference = 'Stop'
$api = 'https://sh01.vivojaymail.workers.dev'
$progressFile = "$env:USERPROFILE\sh01-scanner-progress.json"
$progress = @{}
$headers = @{}
function Invoke-ScannerRequest {
    param([string]$Path, [string]$Method = 'Get', $Payload = $null)
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $response = $null
        $failure = $null
        $httpRetry = 0
        try {
            $requestArgs = @{Uri="$api$Path"; Method=$Method; Headers=$headers; TimeoutSec=240}
            if ($null -ne $Payload) {
                $requestArgs.ContentType = 'application/json'
                $requestArgs.Body = ($Payload | ConvertTo-Json -Depth 10)
            }
            $response = Invoke-RestMethod @args
            if ($response.PSObject.Properties.Name -notcontains 'ok' -or $response.ok -eq $true) {
                return $response
            }
            $failure = $response
        }
        catch {
            if ($_.Exception.Response) {
                $rawRetry = $_.Exception.Response.Headers['Retry-After']
                [double]$numericRetry = 0
                if ([double]::TryParse([string]$rawRetry, [ref]$numericRetry)) { $httpRetry = $numericRetry }
            }
            if ($_.ErrorDetails.Message) {
                try { $failure = $_.ErrorDetails.Message | ConvertFrom-Json } catch { }
            }
            if ($null -eq $failure) { throw 'Request failed without a readable scanner response. Progress retained; no automatic replay of an ambiguous timeout.' }
        }
        $items = @()
        if ($failure.errors) { $items += @($failure.errors) }
        if ($failure.detail) { $items += @($failure.detail) }
        $retry = $items.Count -gt 0
        $wait = [Math]::Max(75, $httpRetry)
        foreach ($item in $items) {
            $d = $item
            if ($item.diagnostic) { $d = $item.diagnostic }
            $limited = ($d.dashboard_code -eq 'store_rate_limited' -or $item.code -eq 'scanner_storage_cooldown')
            if (-not $limited) { $retry = $false }
            if ($d.retry_after_seconds) { $wait = [Math]::Max($wait, [double]$d.retry_after_seconds) }
        }
        if (-not $retry -or $attempt -eq 3 -or $wait -gt 300) {
            [ordered]@{errors=$failure.errors; detail=$failure.detail; counts=$failure.counts; coverage=$failure.coverage} | ConvertTo-Json -Depth 12 | Write-Host
            throw 'Stopped without advancing this batch. Review the diagnostics above.'
        }
        $wait = [int][Math]::Ceiling($wait) + (Get-Random -Minimum 0 -Maximum 6)
        Write-Host "Storage cooldown: waiting $wait seconds, then retrying the same batch ($attempt/3)."
        Start-Sleep -Seconds $wait
    }
}
try {
    if (Test-Path $progressFile) {
        $saved = Get-Content $progressFile -Raw | ConvertFrom-Json
        foreach ($property in $saved.PSObject.Properties) { $progress[$property.Name] = $property.Value }
    }
    $secret = Read-Host 'Existing MEDIA_CLIP_API_KEY' -AsSecureString
    $credential = [System.Management.Automation.PSCredential]::new('api', $secret)
    $headers = @{'X-API-KEY'=$credential.GetNetworkCredential().Password}
    $catalogue = Invoke-ScannerRequest '/social/inventory/accounts'
    if ($null -eq $catalogue.rows) { throw 'Invalid account catalogue.' }
    $ranBatch = $false
    for ($round = 1; $round -le $Rounds; $round++) {
        $unfinished = 0
        foreach ($account in $catalogue.rows) {
            if ($account.stored_posts -le 0) { continue }
            $id = "$($account.platform):$($account.account_id)"
            $previous = $progress[$id]
            if ($previous.complete -eq $true) { continue }
            $unfinished++
            if ($ranBatch) {
                Write-Host 'Pacing: waiting 75 seconds before the next account batch...'
                Start-Sleep -Seconds 75
            }
            $payload = @{platform=$account.platform; account_id=$account.account_id; max_posts=2}
            if ($previous.cursor) { $payload.cursor = $previous.cursor }
            Write-Host "Round $round - $($account.platform): $($account.handle)"
            $result = Invoke-ScannerRequest '/social/inventory/scan' 'Post' $payload
            if ($null -eq $result.coverage -or $result.PSObject.Properties.Name -notcontains 'resume_cursor') { throw 'Invalid scan response; progress retained.' }
            [ordered]@{account=$account.handle; ok=$result.ok; counts=$result.counts; coverage=$result.coverage; errors=$result.errors} | ConvertTo-Json -Depth 12 | Write-Host
            $complete = $result.coverage.catalogue_exhausted -eq $true
            if (-not $complete -and -not $result.resume_cursor) { throw 'Missing continuation cursor; progress retained.' }
            $progress[$id] = @{cursor=$result.resume_cursor; complete=$complete}
            $progress | ConvertTo-Json -Depth 10 | Set-Content "$progressFile.tmp" -Encoding UTF8
            Move-Item "$progressFile.tmp" $progressFile -Force
            $ranBatch = $true
        }
        if ($unfinished -eq 0) { break }
    }
    Write-Host 'Requested rounds finished. Progress saved. Completion covers stored catalogue posts, not verified full provider history.'
}
catch { Write-Host $_.Exception.Message; throw }
finally {
    $headers.Clear()
    Remove-Variable secret, credential, headers -ErrorAction SilentlyContinue
}
