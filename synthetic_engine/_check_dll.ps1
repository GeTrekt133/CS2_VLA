$srcPath = 'C:\Users\misas\CS2_NN\synthetic_engine\plugin\BotPoseControl\bin\Release\net8.0\BotPoseControl.dll'
$dstPath = 'D:\CS2_Pinned\game\csgo\addons\counterstrikesharp\plugins\BotPoseControl\BotPoseControl.dll'

$src = Get-FileHash -LiteralPath $srcPath -Algorithm SHA256
$dst = Get-FileHash -LiteralPath $dstPath -Algorithm SHA256

Write-Host "src hash: $($src.Hash)"
Write-Host "dst hash: $($dst.Hash)"
Write-Host "match:    $($src.Hash -eq $dst.Hash)"
Write-Host ""

$bytes = [IO.File]::ReadAllBytes($dstPath)
$ascii  = [Text.Encoding]::ASCII.GetString($bytes)
$utf16  = [Text.Encoding]::Unicode.GetString($bytes)

function Count($s, $needle) {
    $count = 0
    $idx = 0
    while (($idx = $s.IndexOf($needle, $idx)) -ge 0) { $count++; $idx++ }
    return $count
}

Write-Host "Strings in deployed DLL (ASCII / UTF-16):"
foreach ($needle in 'm_angEyeAngles','setang','Replay-Loadout','slot1','EyeAngles','use ','v_angle') {
    $a = Count $ascii $needle
    $u = Count $utf16 $needle
    Write-Host ("  {0,-20} : {1,3} / {2,3}" -f $needle, $a, $u)
}

$dstInfo = Get-Item $dstPath
Write-Host ""
Write-Host "Deployed: $($dstInfo.LastWriteTime)  ($($dstInfo.Length) bytes)"
