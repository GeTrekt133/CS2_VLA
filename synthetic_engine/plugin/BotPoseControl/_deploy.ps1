$src = 'C:\Users\misas\CS2_NN\synthetic_engine\plugin\BotPoseControl\bin\Release\net8.0'
$dst = 'D:\CS2_Pinned\game\csgo\addons\counterstrikesharp\plugins\BotPoseControl'

# Files our plugin actually needs at runtime that CSSharp does NOT already load.
# Everything else (Microsoft.*, Serilog.*, Tomlyn, Scrutor, McMaster, CSSharp itself)
# is shipped by CSSharp host and would ABI-conflict if duplicated in the plugin ALC.
$wanted = @(
    'BotPoseControl.dll',
    'CS2TraceRay.dll',
    'Reloaded.Hooks.dll',
    'Reloaded.Hooks.Definitions.dll',
    'Reloaded.Memory.dll',
    'Reloaded.Memory.Buffers.dll',
    'Reloaded.Assembler.dll',
    'FASM.DLL',
    'FASMX64.DLL',
    'Iced.dll'
)

foreach ($f in $wanted) {
    $sp = Join-Path $src $f
    $dp = Join-Path $dst $f
    if (Test-Path $sp) {
        Copy-Item -LiteralPath $sp -Destination $dp -Force
        $info = Get-Item $dp
        Write-Host ('  ok   {0,12}  {1}' -f $info.Length, $f)
    } else {
        Write-Host ('  MISS              {0}' -f $f)
    }
}

Write-Host ''
Write-Host 'Deployed to:'
Write-Host "  $dst"
