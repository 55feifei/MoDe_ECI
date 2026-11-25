param(
    [Parameter(Mandatory = $true)][int]$TrainerProcessId
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$output = Join-Path $workspace 'runs/roberta_dall_seed3407_gpu_e15_p5'
$expertRoot = Join-Path $workspace 'artifacts/v4_roberta_mlm_dall_seed3407_gpu_v1_e15_p5/seed_3407/fold_0/final/experts'
$python = 'F:\MoDE_ECI_research\.venv\Scripts\python.exe'
$originalProcess = Get-Process -Id $TrainerProcessId
$originalStartTime = $originalProcess.StartTime

function Write-HandoffState([string]$status, [string]$detail) {
    @{status = $status; detail = $detail; trainer_process_id = $TrainerProcessId;
      updated_at = (Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'stage_handoff.json') -Encoding utf8
}

try {
    Write-HandoffState 'waiting_for_final_demo_experts' 'Preserving the current D_all and specialist training.'
    while ($true) {
        $current = Get-Process -Id $TrainerProcessId -ErrorAction SilentlyContinue
        if ($null -eq $current -or $current.StartTime -ne $originalStartTime) {
            throw 'Original trainer ended before the demo stage boundary; inspect train.log and status.json.'
        }
        $ready = $true
        foreach ($branch in @('D1', 'D2', 'D3', 'D4')) {
            $directory = Join-Path $expertRoot $branch
            $summary = Join-Path $directory 'training.json'
            $provenance = Join-Path $directory 'provenance.json'
            if (-not (Test-Path -LiteralPath $summary) -or -not (Test-Path -LiteralPath $provenance)) {
                $ready = $false
                break
            }
            if ((Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json).status -ne 'complete') {
                $ready = $false
                break
            }
        }
        if ($ready) { break }
        Start-Sleep -Seconds 3
    }
    $identity = Get-CimInstance Win32_Process -Filter "ProcessId=$TrainerProcessId"
    if ($identity.CommandLine -notmatch 'run_demo_pretraining\.py') {
        throw 'Trainer process identity differs; refusing to stop it.'
    }
    Write-HandoffState 'switching_runner' 'Demo checkpoints are complete; continuing with early evaluation before C/S/M.'
    Stop-Process -Id $TrainerProcessId -Force
    $continuation = Start-Process -FilePath $python -ArgumentList @(
        '-u', (Join-Path $PSScriptRoot 'run_demo_pretraining.py'),
        '--config', (Join-Path $workspace 'configs/roberta_mlm_dall_seed3407_gpu.yaml'), '--fold', '0'
    ) -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $output 'train_continued.log') `
      -RedirectStandardError (Join-Path $output 'train_continued.err') -PassThru
    Write-HandoffState 'continuing' "Updated runner launched as process $($continuation.Id)."
    $continuation.WaitForExit()
    $result = Get-Content -LiteralPath (Join-Path $output 'status.json') -Raw | ConvertFrom-Json
    if ($result.status -ne 'complete') { throw "Continuation ended with status $($result.status)." }
    Write-HandoffState 'complete' 'Independent demo evaluation and the full fold 0 pipeline completed.'
} catch {
    Write-HandoffState 'failed' $_.Exception.Message
    throw
}
