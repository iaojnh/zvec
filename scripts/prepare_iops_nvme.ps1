param([string]$OutputDirectory = 'iops-results/nvme-setup')
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
Get-Disk | Select-Object Number,FriendlyName,BusType,Size,PartitionStyle,NumberOfPartitions,IsSystem,IsBoot |
    ConvertTo-Json | Out-File "$OutputDirectory/disks-before.json"
Get-Partition | Select-Object DiskNumber,PartitionNumber,DriveLetter,Size |
    ConvertTo-Json | Out-File "$OutputDirectory/partitions-before.json"
if ($env:GITHUB_ACTIONS -ne 'true' -or $env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    throw 'Scratch initialization is restricted to ephemeral GitHub-hosted VMs.'
}
$candidates = @(Get-Disk | Where-Object {
    $_.FriendlyName -eq 'Microsoft NVMe Direct Disk v2' -and $_.BusType -eq 'NVMe' -and
    $_.PartitionStyle -eq 'RAW' -and $_.NumberOfPartitions -eq 0 -and
    -not $_.IsSystem -and -not $_.IsBoot
})
if ($candidates.Count -ne 1) {
    Write-Host 'No unique verified blank local NVMe disk; system-volume probes only.'
    exit 0
}
$scratchDisk = $candidates[0]
if (@(Get-Partition -DiskNumber $scratchDisk.Number -ErrorAction SilentlyContinue).Count -ne 0) {
    throw 'Candidate has partitions; refusing to initialize it.'
}
if ($scratchDisk.IsOffline) { Set-Disk -Number $scratchDisk.Number -IsOffline $false }
if ($scratchDisk.IsReadOnly) { Set-Disk -Number $scratchDisk.Number -IsReadOnly $false }
Initialize-Disk -Number $scratchDisk.Number -PartitionStyle GPT
$partition = New-Partition -DiskNumber $scratchDisk.Number -UseMaximumSize -AssignDriveLetter
$partition | Format-Volume -FileSystem NTFS -NewFileSystemLabel ZvecIOPS -Confirm:$false
$probeRoot = "$($partition.DriveLetter):\zvec-storage-iops"
New-Item -ItemType Directory -Force $probeRoot | Out-Null
"IOPS_NVME_ROOT=$probeRoot" >> $env:GITHUB_ENV
Get-Partition -DiskNumber $scratchDisk.Number | Select-Object DiskNumber,PartitionNumber,DriveLetter,Size |
    ConvertTo-Json | Out-File "$OutputDirectory/partitions-after.json"
