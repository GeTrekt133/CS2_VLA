$lines = Get-Content 'D:\CS2_Pinned\game\csgo\addons\counterstrikesharp\logs\log-BotPoseControl20260511_020.txt'
# Show only last load section + hook install + first hits
$lastLoad = -1
for ($i = $lines.Count - 1; $i -ge 0; $i--) {
    if ($lines[$i] -match 'Loaded v0\.5') { $lastLoad = $i; break }
}
if ($lastLoad -lt 0) { Write-Host 'No v0.5 load found'; exit }

# Print from last load up to 20 hits log lines
$count = 0
for ($i = $lastLoad; $i -lt $lines.Count -and $count -lt 80; $i++) {
    if ($lines[$i] -match 'Loaded v0\.5|HumanViewHook|SignatureScanner|hook hits|Failed') {
        Write-Host $lines[$i]
        $count++
    }
}
