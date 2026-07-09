<#
.SYNOPSIS
Runs a W&B sweep agent for this project.

.PARAMETER SweepId
Full W&B sweep path, for example entity/project/sweep_id.

.PARAMETER Count
Number of sweep runs to execute before exiting.

.PARAMETER Gpu
CUDA device index to expose. Use -1 to leave CUDA_VISIBLE_DEVICES unchanged.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SweepId,

    [int]$Count = 1,

    [int]$Gpu = -1
)

if ($Gpu -ge 0) {
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
}

uv run wandb agent --count $Count $SweepId
