$lines = Get-Content 'D:\CS2_Pinned\game\csgo\addons\counterstrikesharp\logs\log-BotPoseControl20260511_020.txt'
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match 'Failed to install hook') {
        $end = [Math]::Min($i + 40, $lines.Count - 1)
        for ($j = $i; $j -le $end; $j++) { Write-Host $lines[$j] }
        Write-Host '---'
    }
}
