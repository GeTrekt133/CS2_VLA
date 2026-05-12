$b = [IO.File]::ReadAllBytes('D:\CS2_Pinned\game\csgo\addons\counterstrikesharp\plugins\BotPoseControl\BotPoseControl.dll')
$u0 = [Text.Encoding]::Unicode.GetString($b, 0, $b.Length - ($b.Length % 2))
$u1 = [Text.Encoding]::Unicode.GetString($b, 1, $b.Length - 1 - (($b.Length - 1) % 2))

Write-Host 'Looking for v0.5 markers in deployed DLL:'
foreach ($n in 'v0.5.0-eyeangles', 'HumanViewHook', 'SignatureScanner', 'GetEyeAngles', 'Reloaded.Hooks') {
    $a0 = $u0.IndexOf($n)
    $a1 = $u1.IndexOf($n)
    Write-Host ('  {0,-30} aligned={1,7}  offset={2,7}' -f $n, $a0, $a1)
}
