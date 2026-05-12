$b   = [IO.File]::ReadAllBytes('D:\CS2_Pinned\game\csgo\addons\counterstrikesharp\plugins\BotPoseControl\BotPoseControl.dll')
$u0  = [Text.Encoding]::Unicode.GetString($b, 0, $b.Length - ($b.Length % 2))
$u1  = [Text.Encoding]::Unicode.GetString($b, 1, $b.Length - 1 - (($b.Length - 1) % 2))

$names = 'setang','v_angle','m_angEyeAngles','EyeAngles','use weapon_','BCustomLoadout','Replay-Loadout','slot1','setang {0:F4}'

Write-Host "Aligned (offset 0):"
foreach ($n in $names) {
    $i = $u0.IndexOf($n)
    Write-Host ("  {0,-25} {1}" -f $n, $i)
}

Write-Host ""
Write-Host "Misaligned (offset 1):"
foreach ($n in $names) {
    $i = $u1.IndexOf($n)
    Write-Host ("  {0,-25} {1}" -f $n, $i)
}
