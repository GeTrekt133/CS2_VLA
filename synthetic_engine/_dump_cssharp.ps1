Add-Type -Path 'C:\Users\misas\.nuget\packages\counterstrikesharp.api\1.0.305\lib\net8.0\CounterStrikeSharp.API.dll' -ErrorAction SilentlyContinue

$asm = [System.Reflection.Assembly]::LoadFrom('C:\Users\misas\.nuget\packages\counterstrikesharp.api\1.0.305\lib\net8.0\CounterStrikeSharp.API.dll')

Write-Host "=== Listeners ==="
$asm.GetTypes() | Where-Object { $_.FullName -like '*Listeners*' } | ForEach-Object {
    Write-Host "  $($_.FullName)"
}

Write-Host ""
Write-Host "=== Types containing 'UserCmd' or 'RunCommand' ==="
$asm.GetTypes() | Where-Object { $_.FullName -match 'UserCmd|RunCommand|CheckTransmit' } | ForEach-Object {
    Write-Host "  $($_.FullName)"
}

Write-Host ""
Write-Host "=== Methods on PlayerPawn containing 'Snap' or 'Set' ==="
$pawn = $asm.GetType('CounterStrikeSharp.API.Core.CCSPlayerPawn')
if ($pawn) {
    $pawn.GetMethods() | Where-Object { $_.Name -match 'Snap|SetView|Force|Lock' } | Select-Object -First 20 | ForEach-Object {
        Write-Host "  $($_.Name)"
    }
}
