param([int]$JobId = 44, [int]$Seconds = 25)
$ErrorActionPreference = "SilentlyContinue"
$py = "D:/HO/.venv/Scripts/python.exe"
$deadline = (Get-Date).AddSeconds($Seconds)
$last = $null
while ((Get-Date) -lt $deadline) {
    $raw = & curl.exe -s "http://localhost:8000/api/analysis/jobs/$JobId"
    $last = $raw
    if ($raw -match '"status"\s*:\s*"(completed|failed|cancelled)"') { break }
    Start-Sleep -Seconds 4
}
$last | & $py -c "import sys,json;d=json.load(sys.stdin);print({k:d.get(k) for k in ('id','status','processed_frames','total_frames','current_stage','current_stage_status','stage_step','error_message','events_count','candidates_count','rejected_count')})"