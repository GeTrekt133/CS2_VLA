$b = [IO.File]::ReadAllBytes('D:\CS2_Pinned\game\csgo\addons\counterstrikesharp\plugins\BotPoseControl\BotPoseControl.dll')
$u = [Text.Encoding]::Unicode.GetString($b)
foreach ($n in 'm_angEyeAngles','setang','v_angle','Replay-Loadout','slot1','EyeAngles','use weapon') {
    Write-Host ("UTF-16 IndexOf '{0,-20}' = {1}" -f $n, $u.IndexOf($n))
}
