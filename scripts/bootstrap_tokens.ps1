$Base = "https://sh01.vivojaymail.workers.dev"
$ApiKey = Read-Host "SH01 X-API-Key"
$accounts = @("erika.devereux", "vlvt.ave", "cyootstuff")

foreach ($account in $accounts) {
    $token = Read-Host "Long-lived Threads token for @$account"
    $body = @{
        account = $account
        access_token = $token
        expires_in = 5184000
    } | ConvertTo-Json -Compress

    Invoke-RestMethod `
        -Method POST `
        -Uri "$Base/threads/tokens/bootstrap" `
        -Headers @{ "X-API-Key" = $ApiKey } `
        -ContentType "application/json" `
        -Body $body

    Write-Host "Bootstrapped @$account"
}
