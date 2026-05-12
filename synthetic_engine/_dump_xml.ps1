$xml = [xml](Get-Content 'C:\Users\misas\.nuget\packages\counterstrikesharp.api\1.0.367\lib\net8.0\CounterStrikeSharp.API.xml')

Write-Host "=== Pawn / Player methods touching view/angle ==="
$xml.doc.members.member | Where-Object { $_.name -match 'CCSPlayer|CBasePlayer|CCSObserver' -and $_.name -match 'View|Angle|Snap|Camera|Spec|Force' } | Select-Object -First 60 | ForEach-Object {
    Write-Host $_.name
}

Write-Host ""
Write-Host "=== All Schema methods (for scanning) ==="
$xml.doc.members.member | Where-Object { $_.name -match 'Schema\.' } | Select-Object -First 30 | ForEach-Object {
    Write-Host $_.name
}

Write-Host ""
Write-Host "=== Listeners.* ==="
$xml.doc.members.member | Where-Object { $_.name -match 'Listeners\+' } | Select-Object -First 50 | ForEach-Object {
    Write-Host $_.name
}

Write-Host ""
Write-Host "=== View/Snap/Lock methods ==="
$xml.doc.members.member | Where-Object { $_.name -match 'M:.*(SnapTo|SetView|ForceTeleport|LockView)' } | Select-Object -First 20 | ForEach-Object {
    Write-Host $_.name
}
