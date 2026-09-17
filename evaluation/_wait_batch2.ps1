$ErrorActionPreference = "SilentlyContinue"
$log = "D:/HO/evaluation/reports/_live_batch2.log"
$deadline = (Get-Date).AddSeconds(25)
while ((Get-Date) -lt $deadline) {
    $done = Select-String -Path $log -Pattern "Report written" -Quiet
    if ($done) { break }
    Start-Sleep -Seconds 5
}
$preds = Select-String -Path $log -Pattern "pred=" | ForEach-Object { $_.Line }
$report = Select-String -Path $log -Pattern "Report written" | ForEach-Object { $_.Line }
$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "run_frozen_eval" }
Write-Output "pred-lines: $($preds.Count)"
Write-Output "report: $($report)"
Write-Output "eval-running: $(if ($proc) { 'yes' } else { 'no' })"
Write-Output "latest-pred: $($preds[-1])"