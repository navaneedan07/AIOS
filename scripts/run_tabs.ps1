# Open the four terminal tabs of the scheduling demo, one window each.
#
# Each tab is an independent agent submitting to a *different* model, so the
# kernel log shows four distinct agents rather than four identical ones, and
# the priorities span all three levels.
#
# The two hosted models respond in a second or two; the two local Ollama models
# take longer, which is what makes queuing visible: type into a local (low)
# tab, then into the Gemini (high) tab, and under `priority` the Gemini reply
# comes back while the local request is still generating.
#
# Start the kernel first, in its own window:
#   .\scripts\run_kernel.ps1
#
# Usage:
#   .\scripts\run_tabs.ps1              # open the four tabs
#   .\scripts\run_tabs.ps1 -DryRun      # print the commands instead
#
# The kernel's own window is where the scheduling log appears (QUEUED / RUN /
# DONE / READY), so keep it visible while typing into the tabs.

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = (Get-Command python).Source
}

# name, model, priority ("" = derived from the model's backend)
# Four different models: two hosted (Gemini, Groq) and two local (Ollama).
$Tabs = @(
    @{ Name = "gemini_tab"; Model = "gemini:gemini-2.5-flash";  Priority = "high" },
    @{ Name = "groq_tab";   Model = "groq:openai/gpt-oss-20b";  Priority = "normal" },
    @{ Name = "gemma_tab";  Model = "ollama:gemma3:1b";         Priority = "low" },
    @{ Name = "qwen_tab";   Model = "ollama:qwen2:0.5b";        Priority = "low" }
)

foreach ($Tab in $Tabs) {
    $TabArgs = @(
        "scripts/run_terminal.py",
        "--name", $Tab.Name,
        "--model", $Tab.Model,
        "--mode", "chat"
    )
    if ($Tab.Priority -ne "") {
        $TabArgs += @("--priority", $Tab.Priority)
    }

    if ($DryRun) {
        Write-Host "& `"$Python`" $($TabArgs -join ' ')"
    } else {
        Start-Process -FilePath $Python -ArgumentList $TabArgs -WorkingDirectory $RootDir
    }
}

if (-not $DryRun) {
    Write-Host "Opened $($Tabs.Count) tabs. Type a message in each and watch the kernel window."
}
