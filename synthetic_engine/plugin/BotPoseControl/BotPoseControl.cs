using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using CounterStrikeSharp.API;
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Modules.Memory;
using CounterStrikeSharp.API.Modules.Utils;
using CS2TraceRay.Class;
using CS2TraceRay.Enum;
using CS2TraceRay.Struct;
using SysVec3 = System.Numerics.Vector3;
using Microsoft.Extensions.Logging;

namespace BotPoseControl;

public class BotPose
{
    [JsonPropertyName("slot")]    public int     Slot    { get; set; }
    [JsonPropertyName("pos")]     public float[]? Pos    { get; set; }   // [x, y, z]
    [JsonPropertyName("yaw")]     public float   Yaw     { get; set; }
    [JsonPropertyName("pitch")]   public float   Pitch   { get; set; }
    [JsonPropertyName("hp")]      public int?    Hp      { get; set; }
    [JsonPropertyName("armor")]   public int?    Armor   { get; set; }
    [JsonPropertyName("helmet")]  public bool?   Helmet  { get; set; }
    [JsonPropertyName("ducking")] public bool    Ducking { get; set; } = false;
    [JsonPropertyName("freeze")]  public bool    Freeze  { get; set; } = true;
    [JsonPropertyName("weapon")]  public string? Weapon  { get; set; }

    // Internal: tracks ticks since initial apply. Used for ground-stuck auto-recovery.
    [JsonIgnore] public int  TicksSinceApply  { get; set; } = 0;
    [JsonIgnore] public bool AutoCrouchTried  { get; set; } = false;
}

public class PoseRequest
{
    [JsonPropertyName("action")]  public string  Action  { get; set; } = "";
    [JsonPropertyName("tick_id")] public int     TickId  { get; set; }
    [JsonPropertyName("poses")]   public List<BotPose>? Poses { get; set; }
    [JsonPropertyName("team")]    public string? Team    { get; set; }
    [JsonPropertyName("count")]   public int     Count   { get; set; } = 1;
    [JsonPropertyName("n_ct")]    public int     NCt     { get; set; } = 0;
    [JsonPropertyName("n_t")]     public int     NT      { get; set; } = 0;
    [JsonPropertyName("offset")]  public long    Offset  { get; set; } = 0;
    [JsonPropertyName("module")]  public string? Module  { get; set; }
    [JsonPropertyName("yaw")]     public float?  Yaw     { get; set; }
    [JsonPropertyName("pitch")]   public float?  Pitch   { get; set; }
    [JsonPropertyName("scale")]   public float?  Scale   { get; set; }
    [JsonPropertyName("cmd")]     public string? Cmd     { get; set; }
    // For spawn_planted_bomb / spawn_dropped_weapon:
    [JsonPropertyName("pos")]     public float[]? Pos    { get; set; }   // [x, y, z]

    // For round replay — incremental chunked submission
    [JsonPropertyName("replay_init")]    public ReplayInit? ReplayInit { get; set; }
    [JsonPropertyName("replay_chunk")]   public ReplayChunk? ReplayChunk { get; set; }

    // For trace_visibility_batch — engine-level line-of-sight queries.
    [JsonPropertyName("from")]       public float[]?       From       { get; set; }   // [x, y, z] eye position
    [JsonPropertyName("targets")]    public List<float[]>? Targets    { get; set; }   // list of [x, y, z]
    [JsonPropertyName("skip_slots")] public List<int>?     SkipSlots  { get; set; }   // optional: slot of bot to additionally skip per-target
    [JsonPropertyName("tolerance")]  public float          Tolerance  { get; set; } = 12.0f;
}

public class ReplayInit
{
    [JsonPropertyName("tick_start")] public int TickStart { get; set; }
    [JsonPropertyName("tick_end")]   public int TickEnd   { get; set; }
    /// Map demo steamid → bot slot (Python pre-computes via list_bots + team).
    [JsonPropertyName("sid_to_slot")] public Dictionary<string, int>? SidToSlot { get; set; }
    /// Per-player initial loadout: sid → list of weapon names (without 'weapon_' prefix or with).
    [JsonPropertyName("inventory")]  public Dictionary<string, List<string>>? Inventory { get; set; }
    /// Where to put the human player during replay: "ct" / "t" / "spec".
    [JsonPropertyName("human_team")] public string? HumanTeam { get; set; }
    /// Expected bot counts per team — plugin kicks extras after restart-induced auto-balance.
    [JsonPropertyName("expect_ct")]  public int ExpectCt { get; set; } = 0;
    [JsonPropertyName("expect_t")]   public int ExpectT  { get; set; } = 0;
    /// If true, force human's V_angle (rigid lock to demo). If false, user controls view —
    /// reduces client prediction conflicts and lets weapon deploy animations play.
    [JsonPropertyName("lock_human_view")] public bool LockHumanView { get; set; } = false;
}

public class ReplayChunk
{
    /// Each tick row: [tick, sid, x, y, z, yaw, pitch, flags]
    [JsonPropertyName("ticks")] public List<List<object>>? Ticks { get; set; }
    /// Each fire row: [tick, sid]
    [JsonPropertyName("fires")] public List<List<object>>? Fires { get; set; }
    /// Each death row: [tick, victim_sid]
    [JsonPropertyName("deaths")] public List<List<object>>? Deaths { get; set; }
    /// Each weapon-change row: [tick, sid, weapon_name]
    [JsonPropertyName("weapons")] public List<List<object>>? Weapons { get; set; }
}

public class PoseResponse
{
    [JsonPropertyName("ok")]      public bool    Ok      { get; set; }
    [JsonPropertyName("tick_id")] public int     TickId  { get; set; }
    [JsonPropertyName("applied")] public int     Applied { get; set; }
    [JsonPropertyName("errors")]  public List<string> Errors { get; set; } = new();
    [JsonPropertyName("message")] public string? Message { get; set; }
}

public class BotPoseControlPlugin : BasePlugin
{
    public override string ModuleName        => "BotPoseControl";
    public override string ModuleVersion     => "0.5.0-eyeangles-detour";
    public override string ModuleAuthor      => "Misha";
    public override string ModuleDescription => "Pose control for CS2 synthetic data engine";

    private const int TCP_PORT = 27040;

    /// <summary>Read a single CTransform.Position (3 floats) from bone array at given index.</summary>
    private static float[]? ReadBone(IntPtr boneArrayBase, int index)
    {
        try
        {
            IntPtr addr = IntPtr.Add(boneArrayBase, index * 32);
            float x = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(addr));
            float y = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(addr + 4));
            float z = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(addr + 8));
            if (float.IsNaN(x) || float.IsNaN(y) || float.IsNaN(z)) return null;
            if (float.IsInfinity(x) || float.IsInfinity(y) || float.IsInfinity(z)) return null;
            return new[] { x, y, z };
        }
        catch { return null; }
    }

    // ---------------- Safe memory access via VirtualQuery ---------------- //
    [StructLayout(LayoutKind.Sequential)]
    private struct MEMORY_BASIC_INFORMATION
    {
        public IntPtr BaseAddress;
        public IntPtr AllocationBase;
        public uint   AllocationProtect;
        public ushort PartitionId;
        public IntPtr RegionSize;
        public uint   State;
        public uint   Protect;
        public uint   Type;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern UIntPtr VirtualQuery(IntPtr lpAddress, ref MEMORY_BASIC_INFORMATION lpBuffer, UIntPtr dwLength);
    private const uint MEM_COMMIT      = 0x1000;
    private const uint PAGE_NOACCESS   = 0x01;
    private const uint PAGE_GUARD      = 0x100;
    private const uint READABLE_PROTECT_MASK = 0x66;  // PAGE_READONLY|READWRITE|EXECUTE_READ|EXECUTE_READWRITE

    private static bool IsReadable(IntPtr addr, int size)
    {
        if (addr == IntPtr.Zero) return false;
        var info = new MEMORY_BASIC_INFORMATION();
        var result = VirtualQuery(addr, ref info, (UIntPtr)Marshal.SizeOf<MEMORY_BASIC_INFORMATION>());
        if (result == UIntPtr.Zero) return false;
        if ((info.State & MEM_COMMIT) == 0) return false;
        if ((info.Protect & PAGE_NOACCESS) != 0) return false;
        if ((info.Protect & PAGE_GUARD) != 0) return false;
        if ((info.Protect & READABLE_PROTECT_MASK) == 0) return false;

        long regionStart = (long)info.BaseAddress;
        long regionEnd   = regionStart + (long)info.RegionSize;
        long target      = (long)addr;
        return target >= regionStart && (target + size) <= regionEnd;
    }

    // dwViewMatrix offset in client.dll for our pinned CS2 build (2000807, Apr 29 2026).
    // Source: https://github.com/a2x/cs2-dumper (look up commit for that date)
    // PLACEHOLDER — must be updated to the actual offset before use.
    private const long VIEW_MATRIX_OFFSET = 0x0;

    // Configurable at runtime via 'set_view_matrix_offset' action so we can iterate
    // without rebuilding the plugin.
    private long _viewMatrixOffset = VIEW_MATRIX_OFFSET;

    private TcpListener?            _server;
    private CancellationTokenSource? _cts;
    private readonly ConcurrentDictionary<int, BotPose> _activePoses = new();

    // Punch-angle recording for ground-truth recoil extraction
    private bool _punchRecording = false;
    private readonly List<(int Tick, float Pitch, float Yaw)> _punchSamples = new();

    // Bullet impact recording for occlusion detection
    private bool _impactRecording = false;
    private readonly List<(int Tick, float X, float Y, float Z)> _impactRecords = new();
    // Diagnostics: count failure reasons during recording
    private int _diagTicks = 0, _diagNoViewer = 0, _diagNoPawn = 0,
                _diagBadCamPtr = 0, _diagNullCam = 0, _diagBadPunchPtr = 0, _diagNaN = 0;

    // ---------------- Round replay state ----------------
    // Per-tick pose data for all live players. Key = relative tick (0-based from replay start).
    // Inner dict: slot -> (x, y, z, yaw, pitch, flags)
    private readonly Dictionary<int, Dictionary<int, float[]>> _replayPoses = new();
    // Per-tick fire events: relative tick -> list of slots that fire
    private readonly Dictionary<int, List<int>> _replayFires = new();
    // Per-tick death events: relative tick -> list of slots that die
    private readonly Dictionary<int, List<int>> _replayDeaths = new();
    // Per-tick weapon changes: relative tick -> list of (slot, weapon_name)
    private readonly Dictionary<int, List<(int slot, string weapon)>> _replayWeapons = new();
    // Per-slot initial loadout (from demo): list of weapon_X names. Given at replay start.
    private readonly Dictionary<int, List<string>> _replayInventory = new();
    // Human team & expected counts (post-restart auto-balance correction)
    private string _replayHumanTeam = "ct";
    private int _replayExpectCt = 0;
    private int _replayExpectT  = 0;
    /// If true, override human's V_angle every tick (rigid demo lock).
    /// If false, only Teleport position — user controls view, weapon animations play smoothly.
    private bool _replayLockHumanView = false;
    // Latest target view for human (slot 0 by convention). Written from OnTick (in
    // ApplyReplayTick, alongside bot updates), consumed by OnServerPostEntityThink which
    // fires AFTER ProcessUsercmds — only at this point can we override the value the
    // client sent, otherwise it gets overwritten by the next prediction cycle.
    private (int slot, float pitch, float yaw)? _humanLockAng = null;
    // sid → slot (set in replay_init from Python)
    private Dictionary<string, int> _replaySidToSlot = new();
    private bool _replayActive = false;
    private int  _replayStartServerTick = 0;     // Server tick when replay started
    private int  _replayCurrentRelTick = -1;
    private int  _replayMaxRelTick = 0;
    private readonly HashSet<int> _replayDeadSlots = new();
    // Slots currently in attack (started fire on tick T, will release on tick T+2)
    private readonly Dictionary<int, int> _replayActiveAttacks = new();   // slot → release_tick

    public override void Load(bool hotReload)
    {
        Logger.LogInformation("[BotPoseControl] Loaded v0.5.0-eyeangles-detour (hotReload={HotReload})", hotReload);
        Server.ExecuteCommand($"echo \"[BotPoseControl] Plugin v0.5.0-eyeangles-detour loaded\"");

        StartTcpServer();
        RegisterListener<Listeners.OnTick>(OnTick);
        // Fires AFTER ProcessUsercmds — only point at which we can stamp our own
        // viewangles onto the human's pawn before snapshot is sent to the client.
        RegisterListener<Listeners.OnServerPostEntityThink>(OnPostEntityThink);

        // Native VFunc detour on CBasePlayerPawn::GetEyeAngles in server.dll.
        // The renderer (both server-side AI/visibility and the local listen-server
        // client's camera) calls this getter every frame to read where a pawn looks.
        // By detouring it for our locked human pawn, we return the demo's pitch/yaw
        // directly, bypassing all replication & client-side prediction layers.
        //
        // Reloaded.Hooks resolves FASM.DLL via current-directory fallback when
        // Assembly.Location is empty (CSSharp loads plugins from in-memory stream).
        // CS2's CWD points at the game binary dir, not the plugin's folder — so we
        // temporarily flip CWD over to ModuleDirectory while installing.
        string oldCwd = System.IO.Directory.GetCurrentDirectory();
        try
        {
            System.IO.Directory.SetCurrentDirectory(ModuleDirectory);
            Logger.LogInformation("[BotPoseControl] cwd={Cwd} (for FASM resolve) — moduleDir={MD}",
                                  ModuleDirectory, ModuleDirectory);
            if (!HumanViewHook.Install(Logger))
                Logger.LogWarning("[BotPoseControl] Human view detour unavailable — pitch lock on host won't work");
        }
        finally
        {
            System.IO.Directory.SetCurrentDirectory(oldCwd);
        }

        // Bullet impact recording for occlusion detection
        RegisterEventHandler<EventBulletImpact>((@event, info) =>
        {
            if (_impactRecording)
            {
                _impactRecords.Add((Server.TickCount, @event.X, @event.Y, @event.Z));
            }
            return HookResult.Continue;
        });
    }

    public override void Unload(bool hotReload)
    {
        Logger.LogInformation("[BotPoseControl] Unloading");
        HumanViewHook.Uninstall(Logger);
        _cts?.Cancel();
        _server?.Stop();
        _activePoses.Clear();
    }

    // ---------------------------- TCP server ---------------------------- //

    private void StartTcpServer()
    {
        _cts = new CancellationTokenSource();
        try
        {
            _server = new TcpListener(IPAddress.Loopback, TCP_PORT);
            _server.Start();
            Logger.LogInformation("[BotPoseControl] TCP listening on 127.0.0.1:{Port}", TCP_PORT);
            _ = Task.Run(() => AcceptLoop(_cts.Token));
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "[BotPoseControl] Failed to start TCP server");
        }
    }

    private async Task AcceptLoop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested && _server != null)
        {
            try
            {
                var client = await _server.AcceptTcpClientAsync(ct);
                _ = Task.Run(() => HandleClient(client, ct), ct);
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[BotPoseControl] Accept error");
            }
        }
    }

    private async Task HandleClient(TcpClient client, CancellationToken ct)
    {
        try
        {
            // UTF-8 without BOM — Python json.loads cannot handle BOM.
            var utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
            using (client)
            using (var stream = client.GetStream())
            using (var reader = new StreamReader(stream, utf8NoBom))
            using (var writer = new StreamWriter(stream, utf8NoBom) { AutoFlush = true, NewLine = "\n" })
            {
                while (!ct.IsCancellationRequested)
                {
                    var line = await reader.ReadLineAsync(ct);
                    if (line == null) break;
                    if (string.IsNullOrWhiteSpace(line)) continue;

                    var resp = await ProcessRequest(line);
                    var json = JsonSerializer.Serialize(resp);
                    await writer.WriteLineAsync(json);
                }
            }
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "[BotPoseControl] Client error");
        }
    }

    private Task<PoseResponse> ProcessRequest(string line)
    {
        try
        {
            var req = JsonSerializer.Deserialize<PoseRequest>(line);
            if (req == null)
                return Task.FromResult(new PoseResponse { Ok = false, Message = "Invalid JSON" });

            return req.Action switch
            {
                "ping"          => Task.FromResult(new PoseResponse { Ok = true, Message = "pong" }),
                "set_poses"     => ApplyPosesOnGameThread(req),
                "unfreeze_all"  => UnfreezeAllOnGameThread(),
                "list_bots"     => ListBotsOnGameThread(),
                "spawn_bot"     => SpawnBotOnGameThread(req),
                "kick_bots"     => KickBotsOnGameThread(),
                "setup_match"   => SetupMatchOnGameThread(),
                "restart_round" => RestartRoundOnGameThread(),
                "prepare_match" => PrepareMatchOnGameThread(req),
                "get_geometry"  => GetGeometryOnGameThread(req),
                "get_view_matrix"        => GetViewMatrixOnGameThread(),
                "get_module_info"        => GetModuleInfoOnGameThread(),
                "set_view_matrix_offset" => SetViewMatrixOffset(req),
                "get_bones"              => GetBonesOnGameThread(req),
                "set_player_view"        => SetPlayerViewOnGameThread(req),
                "host_timescale"         => SetHostTimescaleOnGameThread(req),
                "ensure_alive"           => EnsureAliveOnGameThread(),
                "start_attack"           => SetAttackOnGameThread(true),
                "stop_attack"            => SetAttackOnGameThread(false),
                "cleanup_inputs"         => CleanupInputsOnGameThread(),
                "start_punch_record"     => StartPunchRecord(),
                "stop_punch_record"      => StopPunchRecord(),
                "get_punch_record"       => GetPunchRecord(),
                "scan_punch_offset"      => ScanPunchOffsetOnGameThread(),
                "scan_aim_punch_offset"  => ScanAimPunchOffsetOnGameThread(),
                "start_impact_record"    => StartImpactRecord(),
                "stop_impact_record"     => StopImpactRecord(),
                "get_impacts"            => GetImpacts(),
                "clear_impacts"          => ClearImpacts(),
                "set_bot_health"         => SetBotHealthOnGameThread(req),
                "exec_cmd"               => ExecCmdOnGameThread(req),
                "give_weapon"            => GiveWeaponOnGameThread(req),
                "respawn_bots"           => RespawnBotsOnGameThread(req),
                "set_bot_teams"          => SetBotTeamsOnGameThread(req),
                "spawn_planted_bomb"     => SpawnPlantedBombOnGameThread(req),
                "remove_planted_bombs"   => RemovePlantedBombsOnGameThread(),
                "get_planted_bombs"      => GetPlantedBombsOnGameThread(),
                "replay_init"            => ReplayInitOnGameThread(req),
                "replay_chunk"           => ReplayChunkOnGameThread(req),
                "replay_start"           => ReplayStartOnGameThread(),
                "replay_stop"            => ReplayStopOnGameThread(),
                "replay_status"          => ReplayStatusOnGameThread(),
                "replay_set_viewer"      => ReplaySetViewerOnGameThread(req),
                "get_human_slot"         => GetHumanSlotOnGameThread(),
                "move_human_to_team"     => MoveHumanToTeamOnGameThread(req),
                "trace_visibility_batch" => TraceVisibilityBatchOnGameThread(req),
                _               => Task.FromResult(new PoseResponse { Ok = false, Message = $"Unknown action: {req.Action}" }),
            };
        }
        catch (Exception ex)
        {
            return Task.FromResult(new PoseResponse { Ok = false, Message = ex.Message });
        }
    }

    // ---------------------------- Game-thread actions ---------------------------- //

    private Task<PoseResponse> ApplyPosesOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            var resp = new PoseResponse { Ok = true, TickId = req.TickId };
            if (req.Poses == null) { tcs.SetResult(resp); return; }

            foreach (var pose in req.Poses)
            {
                try
                {
                    if (pose.Freeze)
                        _activePoses[pose.Slot] = pose;
                    else
                        _activePoses.TryRemove(pose.Slot, out _);

                    ApplyPose(pose, initialApply: true);  // full teleport including Z
                    resp.Applied++;
                }
                catch (Exception ex)
                {
                    resp.Errors.Add($"slot={pose.Slot}: {ex.Message}");
                }
            }
            if (resp.Errors.Count > 0) resp.Ok = false;
            tcs.SetResult(resp);
        });
        return tcs.Task;
    }

    private Task<PoseResponse> UnfreezeAllOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            int count = _activePoses.Count;
            _activePoses.Clear();
            tcs.SetResult(new PoseResponse { Ok = true, Message = $"Cleared {count} active poses", Applied = count });
        });
        return tcs.Task;
    }

    /// <summary>Move bots to specific teams. Request: poses=[{slot, team:"ct"|"t"}, ...].
    /// Switches CCSPlayerController.Team via ChangeTeam + respawn.</summary>
    private Task<PoseResponse> SetBotTeamsOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                int moved = 0;
                var errors = new List<string>();
                foreach (var p in req.Poses ?? new List<BotPose>())
                {
                    var bot = Utilities.GetPlayerFromSlot(p.Slot);
                    if (bot == null || !bot.IsValid) { errors.Add($"slot {p.Slot} not found"); continue; }
                    var teamStr = (p.Weapon ?? "ct").ToLowerInvariant();   // hijack Weapon field for team string
                    CsTeam targetTeam = teamStr == "t" ? CsTeam.Terrorist : CsTeam.CounterTerrorist;
                    if ((int)bot.Team == (int)targetTeam) continue;        // already on right team
                    try { bot.ChangeTeam(targetTeam); moved++; }
                    catch (Exception ex) { errors.Add($"slot {p.Slot}: {ex.Message}"); }
                }
                // After ChangeTeam, respawn so bots get fresh pawns on the new side
                AddTimer(0.3f, () =>
                {
                    foreach (var p in req.Poses ?? new List<BotPose>())
                    {
                        var bot = Utilities.GetPlayerFromSlot(p.Slot);
                        try { bot?.Respawn(); } catch { }
                    }
                    tcs.SetResult(new PoseResponse
                    {
                        Ok = errors.Count == 0,
                        Applied = moved,
                        Errors = errors,
                        Message = $"Moved {moved} bots to requested teams"
                    });
                });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = ex.Message });
            }
        });
        return tcs.Task;
    }

    private Task<PoseResponse> ListBotsOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                var bots = Utilities.GetPlayers().Where(p => p != null && p.IsValid && p.IsBot).ToList();
                var msg = string.Join(", ", bots.Select(b => $"slot={b.Slot} name={b.PlayerName} team={b.TeamNum}"));
                tcs.SetResult(new PoseResponse { Ok = true, Message = msg, Applied = bots.Count });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = ex.Message });
            }
        });
        return tcs.Task;
    }

    private Task<PoseResponse> SpawnBotOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                var team  = (req.Team ?? "ct").ToLowerInvariant();
                var cmd   = team == "t" ? "bot_add_t" : "bot_add_ct";
                var count = Math.Max(1, req.Count);
                for (int i = 0; i < count; i++)
                    Server.ExecuteCommand(cmd);
                tcs.SetResult(new PoseResponse { Ok = true, Applied = count, Message = $"Spawned {count} bot(s) on {team}" });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = ex.Message });
            }
        });
        return tcs.Task;
    }

    private Task<PoseResponse> KickBotsOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            _activePoses.Clear();
            Server.ExecuteCommand("bot_kick");
            tcs.SetResult(new PoseResponse { Ok = true, Message = "Kicked all bots, cleared active poses" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> SetupMatchOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            // Match settings for endless round + manual bot control.
            // Order matters: cvars first, then mp_restartgame to apply, then bot defaults.
            var phase1 = new[]
            {
                "sv_cheats 1",
                "mp_autoteambalance 0",            // do not move bots between teams
                "mp_limitteams 0",                 // allow team imbalance
                "mp_freezetime 20",                 // 20s freeze (visible in HUD before replay starts)
                "mp_warmuptime 0",
                "mp_warmup_end",
                "mp_team_intro_time 0",            // skip team intro cinematic
                "mp_roundtime 1.92",                // 1:55 standard CS2
                "mp_roundtime_defuse 1.92",
                "mp_roundtime_hostage 1.92",
                "mp_round_restart_delay 0",
                "mp_match_end_restart 0",
                "mp_ignore_round_win_conditions 1", // KEY: round never ends
                "mp_buy_anywhere 1",
                "sv_infinite_ammo 1",              // infinite mag — no reload needed for visibility/spray tests
                "bot_quota 0",                     // do not auto-spawn bots
                "bot_quota_mode normal",
                "bot_stop 1",                      // bots stand still
                "bot_dont_shoot 1",
                // bot_zombie removed — it suppresses animation updates,
                // leaving bones at local (0,0,0). bot_stop+dont_shoot is enough.
                "god",                             // player invincible
                "cl_drawhud 1",
                "r_drawviewmodel 1",
            };
            foreach (var cmd in phase1) Server.ExecuteCommand(cmd);

            // Apply restart so cvars like mp_freezetime take effect immediately.
            Server.ExecuteCommand("mp_restartgame 1");

            tcs.SetResult(new PoseResponse { Ok = true, Applied = phase1.Length, Message = "Match configured for endless round" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> GetGeometryOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                // Viewer = real player (slot=0 by convention; falls back to first non-bot)
                var allPlayers = Utilities.GetPlayers().Where(p => p != null && p.IsValid).ToList();
                var viewer = allPlayers.FirstOrDefault(p => !p.IsBot)
                          ?? allPlayers.FirstOrDefault();
                if (viewer == null || viewer.PlayerPawn?.Value == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "No viewer found" });
                    return;
                }

                var viewerPawn = viewer.PlayerPawn.Value;
                var origin     = viewerPawn.AbsOrigin!;
                var viewOff    = viewerPawn.ViewOffset!;  // (0,0,64) standing or (0,0,46) crouched
                var eye        = new[] { origin.X + viewOff.X, origin.Y + viewOff.Y, origin.Z + viewOff.Z };
                // EyeAngles is what the renderer actually uses (post-interpolation, smoothed).
                // V_angle is the player's intended/input angle. They usually match for static scenes,
                // but EyeAngles is preferred for projection.
                var ang        = viewerPawn.EyeAngles ?? viewerPawn.V_angle ?? new QAngle(0, 0, 0);

                // Read actual FOV from schema; default 90 if zero/unset.
                float fov = 90f;
                try {
                    int curFov = Schema.GetSchemaValue<int>(viewerPawn.Handle, "CBasePlayerPawn", "m_iFOV");
                    int defFov = Schema.GetSchemaValue<int>(viewerPawn.Handle, "CBasePlayerPawn", "m_iDefaultFOV");
                    fov = curFov > 0 ? curFov : (defFov > 0 ? defFov : 90f);
                } catch { /* schema field missing — use default */ }

                // Bots: 3D positions + ducking + alive
                var botList = new List<object>();
                foreach (var bot in allPlayers)
                {
                    if (!bot.IsBot) continue;
                    var p = bot.PlayerPawn?.Value;
                    if (p == null || !p.IsValid) continue;
                    bool alive   = p.Health > 0;
                    bool ducking = (p.Flags & FL_DUCKING) != 0;
                    var bo = p.AbsOrigin!;
                    var bvo = p.ViewOffset!;

                    // CS2 player model has ~25 bones. Reading beyond returns garbage memory.
                    // Indices 0 and 24 are not part of the hitbox (root/decorative); Python filters them.
                    // Named bones: 1=torso, 5=chest, 6=neck, 7=head,
                    // 9=L shoulder, 13=R shoulder, 11=L hand, 15=R hand,
                    // 19=L foot, 22=R foot.
                    int[] BONE_INDICES = Enumerable.Range(0, 25).ToArray();
                    var bonesDict = new Dictionary<string, float[]?>();
                    string boneDebug = "ok";
                    int boneFiltered = 0, boneRead = 0;
                    var bodyComp = p.CBodyComponent;
                    if (bodyComp == null) { boneDebug = "no_body_component"; }
                    else
                    {
                        var skelNode = bodyComp.SceneNode;
                        if (skelNode == null) { boneDebug = "no_scene_node"; }
                        else if (skelNode.Handle == IntPtr.Zero) { boneDebug = "scene_node_zero_handle"; }
                        else
                        {
                            IntPtr bonePtrAddr = IntPtr.Add(skelNode.Handle, 0x1B0);
                            if (!IsReadable(bonePtrAddr, IntPtr.Size)) { boneDebug = "bone_ptr_unreadable"; }
                            else
                            {
                                IntPtr boneArr = Marshal.ReadIntPtr(bonePtrAddr);
                                if (boneArr == IntPtr.Zero) { boneDebug = "bone_arr_null"; }
                                else if (!IsReadable(boneArr, 25 * 32)) { boneDebug = "bone_arr_unreadable"; }
                                else
                                {
                                    float[]? firstBone = null;
                                    foreach (var idx in BONE_INDICES)
                                    {
                                        var bp = ReadBone(boneArr, idx);
                                        if (bp == null) continue;
                                        boneRead++;
                                        if (firstBone == null) firstBone = bp;
                                        if (Math.Abs(bp[0] - bo.X) > 500 ||
                                            Math.Abs(bp[1] - bo.Y) > 500 ||
                                            Math.Abs(bp[2] - bo.Z) > 500)
                                        { boneFiltered++; continue; }
                                        bonesDict[idx.ToString()] = bp;
                                    }
                                    if (bonesDict.Count == 0)
                                    {
                                        string fb = firstBone == null ? "null" :
                                            $"({firstBone[0]:F1},{firstBone[1]:F1},{firstBone[2]:F1})";
                                        boneDebug = $"all_filtered read={boneRead} filt={boneFiltered} bone[0]={fb}";
                                    }
                                }
                            }
                        }
                    }

                    botList.Add(new
                    {
                        slot     = bot.Slot,
                        team     = bot.TeamNum == 3 ? "ct" : "t",
                        alive    = alive,
                        ducking  = ducking,
                        origin   = new[] { bo.X, bo.Y, bo.Z },
                        eye      = new[] { bo.X + bvo.X, bo.Y + bvo.Y, bo.Z + bvo.Z },
                        bones      = bonesDict,                // dict[str_idx → Vec3]
                        bone_debug = boneDebug,                // diagnostic
                        // Backward-compat aliases for code expecting these names:
                        chest_bone = bonesDict.GetValueOrDefault("5"),
                        neck_bone  = bonesDict.GetValueOrDefault("6"),
                        head_bone  = bonesDict.GetValueOrDefault("7"),
                    });
                }

                var payload = new
                {
                    viewer = new
                    {
                        slot        = viewer.Slot,
                        team        = viewer.TeamNum == 3 ? "ct" : (viewer.TeamNum == 2 ? "t" : "spec"),
                        eye         = eye,
                        view_angles = new[] { ang.X, ang.Y, ang.Z },
                        fov         = fov,
                    },
                    bots = botList,
                };

                tcs.SetResult(new PoseResponse
                {
                    Ok      = true,
                    Applied = botList.Count,
                    Message = JsonSerializer.Serialize(payload),
                });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Error: " + ex.Message });
            }
        });
        return tcs.Task;
    }

    // ------------------------------- Memory hook: view matrix ------------------------------- //

    /// <summary>Find a loaded module by name (case-insensitive).</summary>
    private ProcessModule? FindModule(string name)
    {
        var process = Process.GetCurrentProcess();
        foreach (ProcessModule m in process.Modules)
        {
            if (m.ModuleName.Equals(name, StringComparison.OrdinalIgnoreCase))
                return m;
        }
        return null;
    }

    private Task<PoseResponse> GetModuleInfoOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            var mods = new List<object>();
            var process = Process.GetCurrentProcess();
            foreach (ProcessModule m in process.Modules)
            {
                mods.Add(new
                {
                    name = m.ModuleName,
                    base_addr = (long)m.BaseAddress,
                    size = m.ModuleMemorySize,
                });
            }
            tcs.SetResult(new PoseResponse
            {
                Ok = true,
                Applied = mods.Count,
                Message = JsonSerializer.Serialize(mods),
            });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> SetViewMatrixOffset(PoseRequest req)
    {
        _viewMatrixOffset = req.Offset;
        return Task.FromResult(new PoseResponse
        {
            Ok = true,
            Message = $"View matrix offset set to 0x{_viewMatrixOffset:X}",
        });
    }

    private Task<PoseResponse> GetViewMatrixOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                if (_viewMatrixOffset == 0)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "View matrix offset not set. Call set_view_matrix_offset first." });
                    return;
                }

                // Find client.dll (Source 2 client module name).
                var module = FindModule("client.dll");
                if (module == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "client.dll not loaded. Use get_module_info to inspect loaded modules." });
                    return;
                }

                IntPtr matrixAddr = IntPtr.Add(module.BaseAddress, (int)_viewMatrixOffset);

                // Read 16 floats (4x4 matrix). Marshal.Copy is safe for in-process memory.
                float[] mat = new float[16];
                Marshal.Copy(matrixAddr, mat, 0, 16);

                var payload = new
                {
                    matrix = mat,
                    module_base = (long)module.BaseAddress,
                    module_size = module.ModuleMemorySize,
                    offset = _viewMatrixOffset,
                    matrix_addr = (long)matrixAddr,
                };
                tcs.SetResult(new PoseResponse
                {
                    Ok = true,
                    Applied = 16,
                    Message = JsonSerializer.Serialize(payload),
                });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "ReadMatrix error: " + ex.Message });
            }
        });
        return tcs.Task;
    }

    /// <summary>Read bone world positions for a bot via skeleton instance + raw memory.
    /// Tries common Source 2 schema field names; reports detailed errors so we can iterate.</summary>
    private Task<PoseResponse> SetPlayerViewOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
                if (viewer == null || viewer.PlayerPawn?.Value == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "No human player found" });
                    return;
                }
                var pawn = viewer.PlayerPawn.Value;
                float yaw   = req.Yaw   ?? 0f;
                float pitch = Math.Clamp(req.Pitch ?? 0f, -89f, 89f);

                // setang via console — needed to actually drive client view.
                // Direct V_angle write alone gets ignored by client prediction.
                Server.ExecuteCommand($"setang {pitch:F4} {yaw:F4} 0");

                // Plus direct write for redundancy.
                if (pawn.V_angle != null)
                {
                    pawn.V_angle.X = pitch;
                    pawn.V_angle.Y = yaw;
                    pawn.V_angle.Z = 0;
                    Utilities.SetStateChanged(pawn, "CBasePlayerPawn", "v_angle");
                }
                tcs.SetResult(new PoseResponse { Ok = true, Message = $"View set yaw={yaw} pitch={pitch}" });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Error: " + ex.Message });
            }
        });
        return tcs.Task;
    }

    /// <summary>Give a weapon to a specific bot (or player). Removes existing weapons first.
    /// req.Poses contains list of {slot, cmd=weapon_name (without 'weapon_' prefix)}.</summary>
    private Task<PoseResponse> GiveWeaponOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            if (req.Poses == null || req.Poses.Count == 0)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "No poses (need slot + cmd=weapon_name)" });
                return;
            }
            int given = 0;
            foreach (var p in req.Poses)
            {
                var bot = Utilities.GetPlayerFromSlot(p.Slot);
                if (bot == null || !bot.IsValid) continue;
                try
                {
                    bot.RemoveWeapons();
                    var name = p.Weapon ?? "ak47";
                    if (!name.StartsWith("weapon_")) name = "weapon_" + name;
                    bot.GiveNamedItem(name);
                    given++;
                }
                catch { /* skip */ }
            }
            tcs.SetResult(new PoseResponse { Ok = true, Applied = given, Message = $"Gave weapons to {given} bots" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> ExecCmdOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            if (string.IsNullOrEmpty(req.Cmd))
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "No 'cmd' field" });
                return;
            }
            // Allow ; separated commands
            foreach (var part in req.Cmd.Split(';'))
                Server.ExecuteCommand(part.Trim());
            tcs.SetResult(new PoseResponse { Ok = true, Message = $"Executed: {req.Cmd}" });
        });
        return tcs.Task;
    }

    /// <summary>
    /// Force-respawn the given bot slots. Pawn is killed via CommitSuicide,
    /// then Respawn() is called next frame. This forces the animation system
    /// to re-init, which is required when bones stay at local (0,0,0) after
    /// teleport (some bots never get their first animation tick after spawn).
    /// </summary>
    // ---------------------------- Planted bomb spawning + reading ---------------------------- //

    /// <summary>
    /// Spawn a planted_c4 entity at the given world position+yaw. Removes any pre-existing
    /// planted bombs first to avoid duplicates. The bomb's countdown is overridden so it
    /// won't explode mid-scenario.
    /// </summary>
    private Task<PoseResponse> SpawnPlantedBombOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            if (req.Pos == null || req.Pos.Length < 3)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Need pos [x,y,z]" });
                return;
            }
            try
            {
                // Wipe existing bombs first
                foreach (var existing in Utilities.FindAllEntitiesByDesignerName<CPlantedC4>("planted_c4"))
                {
                    if (existing != null && existing.IsValid)
                        existing.Remove();
                }

                var bomb = Utilities.CreateEntityByName<CPlantedC4>("planted_c4");
                if (bomb == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "CreateEntityByName(planted_c4) returned null" });
                    return;
                }

                bomb.HasExploded = false;
                bomb.BombSite = 0;
                bomb.TimerLength = 9999f;     // long timer so it doesn't explode mid-scenario
                bomb.DispatchSpawn();

                var pos = new Vector(req.Pos[0], req.Pos[1], req.Pos[2]);
                var rot = new QAngle(0, req.Yaw ?? 0f, 0);
                bomb.Teleport(pos, rot, new Vector(0, 0, 0));

                tcs.SetResult(new PoseResponse {
                    Ok = true, Applied = 1,
                    Message = $"Planted bomb at ({req.Pos[0]:F1},{req.Pos[1]:F1},{req.Pos[2]:F1}) yaw={req.Yaw ?? 0:F1}"
                });
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[BotPoseControl] spawn_planted_bomb failed");
                tcs.SetResult(new PoseResponse { Ok = false, Message = $"Spawn failed: {ex.Message}" });
            }
        });
        return tcs.Task;
    }

    /// <summary>Remove all planted_c4 entities currently in the world.</summary>
    private Task<PoseResponse> RemovePlantedBombsOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            int removed = 0;
            foreach (var b in Utilities.FindAllEntitiesByDesignerName<CPlantedC4>("planted_c4"))
            {
                if (b != null && b.IsValid) { b.Remove(); removed++; }
            }
            tcs.SetResult(new PoseResponse { Ok = true, Applied = removed,
                                              Message = $"Removed {removed} bombs" });
        });
        return tcs.Task;
    }

    /// <summary>
    /// Read all planted_c4 entities — return their world position, rotation, and
    /// collision AABB (mins/maxs in local space). Python projects 8 corners to screen.
    /// </summary>
    private Task<PoseResponse> GetPlantedBombsOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                var bombs = new List<object>();
                foreach (var b in Utilities.FindAllEntitiesByDesignerName<CPlantedC4>("planted_c4"))
                {
                    if (b == null || !b.IsValid) continue;
                    var origin = b.AbsOrigin;
                    var rot    = b.AbsRotation;
                    if (origin == null || rot == null) continue;

                    // Default fallback bbox if Collision data unavailable
                    float[] mins = new float[]{ -6f, -6f,  0f };
                    float[] maxs = new float[]{  6f,  6f,  4f };
                    try
                    {
                        var coll = b.Collision;
                        if (coll != null)
                        {
                            mins = new[]{ coll.Mins.X, coll.Mins.Y, coll.Mins.Z };
                            maxs = new[]{ coll.Maxs.X, coll.Maxs.Y, coll.Maxs.Z };
                        }
                    }
                    catch { /* keep fallback */ }

                    bombs.Add(new {
                        index    = (int)b.Index,
                        pos      = new[] { origin.X, origin.Y, origin.Z },
                        rotation = new[] { rot.X, rot.Y, rot.Z },
                        mins     = mins,
                        maxs     = maxs,
                    });
                }
                tcs.SetResult(new PoseResponse {
                    Ok = true, Applied = bombs.Count,
                    Message = JsonSerializer.Serialize(new { bombs = bombs }),
                });
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[BotPoseControl] get_planted_bombs failed");
                tcs.SetResult(new PoseResponse { Ok = false, Message = ex.Message });
            }
        });
        return tcs.Task;
    }

    // ============================================================================
    // Round replay — streams per-tick poses + fire/death events from a recorded round
    // ============================================================================

    /// <summary>Initialize a new replay session. Clears any previous replay state.</summary>
    private Task<PoseResponse> ReplayInitOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            if (req.ReplayInit == null)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Need 'replay_init' field" });
                return;
            }
            _replayPoses.Clear();
            _replayFires.Clear();
            _replayDeaths.Clear();
            _replayWeapons.Clear();
            _replayBotWeapons.Clear();
            _replayInventory.Clear();
            _replayActiveAttacks.Clear();
            _replayDeadSlots.Clear();
            _replayActive = false;
            _replayCurrentRelTick = -1;
            _replayMaxRelTick = req.ReplayInit.TickEnd - req.ReplayInit.TickStart;
            _replaySidToSlot = req.ReplayInit.SidToSlot ?? new Dictionary<string, int>();

            _replayHumanTeam = (req.ReplayInit.HumanTeam ?? "ct").ToLowerInvariant();
            _replayExpectCt = req.ReplayInit.ExpectCt;
            _replayExpectT  = req.ReplayInit.ExpectT;
            _replayLockHumanView = req.ReplayInit.LockHumanView;
            Logger.LogInformation("[Replay-Init] lock_human_view = {Lock}, human_team = {Team}",
                                  _replayLockHumanView, _replayHumanTeam);

            // Map sid-keyed inventory to slot-keyed
            // Don't filter knives from inventory — they're auto-given on spawn anyway,
            // and we need the option to switch back to them during the round.
            if (req.ReplayInit.Inventory != null)
            {
                foreach (var (sid, weps) in req.ReplayInit.Inventory)
                {
                    if (!_replaySidToSlot.TryGetValue(sid, out int slot)) continue;
                    var normalized = weps
                        .Select(NormalizeWeaponName)
                        .Where(n => !string.IsNullOrEmpty(n))
                        .Distinct().ToList();
                    _replayInventory[slot] = normalized;
                }
            }

            tcs.SetResult(new PoseResponse {
                Ok = true,
                Message = $"Replay init: {_replayMaxRelTick} ticks, {_replaySidToSlot.Count} slots, " +
                          $"{_replayInventory.Count} inventories"
            });
        });
        return tcs.Task;
    }

    /// <summary>Append a chunk of replay data (called multiple times to bypass TCP message size).</summary>
    private Task<PoseResponse> ReplayChunkOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            if (req.ReplayChunk == null)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Need 'replay_chunk' field" });
                return;
            }
            int nT = 0, nF = 0, nD = 0;

            // Tick rows: [relTick, sid, x, y, z, yaw, pitch, flags]
            if (req.ReplayChunk.Ticks != null)
            {
                foreach (var row in req.ReplayChunk.Ticks)
                {
                    if (row.Count < 8) continue;
                    int rt = AsInt(row[0]);
                    string sid = row[1]?.ToString() ?? "";
                    if (!_replaySidToSlot.TryGetValue(sid, out int slot)) continue;
                    if (!_replayPoses.TryGetValue(rt, out var slotMap))
                    {
                        slotMap = new Dictionary<int, float[]>();
                        _replayPoses[rt] = slotMap;
                    }
                    slotMap[slot] = new float[] {
                        AsFloat(row[2]), AsFloat(row[3]), AsFloat(row[4]),    // pos x,y,z
                        AsFloat(row[5]), AsFloat(row[6]),                      // yaw, pitch
                        AsFloat(row[7]),                                        // flags
                        row.Count > 8 ? AsFloat(row[8]) : 0f,                  // walk bit
                    };
                    nT++;
                }
            }

            // Fire rows: [relTick, sid]
            if (req.ReplayChunk.Fires != null)
            {
                foreach (var row in req.ReplayChunk.Fires)
                {
                    if (row.Count < 2) continue;
                    int rt = AsInt(row[0]);
                    string sid = row[1]?.ToString() ?? "";
                    if (!_replaySidToSlot.TryGetValue(sid, out int slot)) continue;
                    if (!_replayFires.TryGetValue(rt, out var lst))
                    {
                        lst = new List<int>();
                        _replayFires[rt] = lst;
                    }
                    lst.Add(slot);
                    nF++;
                }
            }

            // Death rows: [relTick, victim_sid]
            if (req.ReplayChunk.Deaths != null)
            {
                foreach (var row in req.ReplayChunk.Deaths)
                {
                    if (row.Count < 2) continue;
                    int rt = AsInt(row[0]);
                    string sid = row[1]?.ToString() ?? "";
                    if (!_replaySidToSlot.TryGetValue(sid, out int slot)) continue;
                    if (!_replayDeaths.TryGetValue(rt, out var lst))
                    {
                        lst = new List<int>();
                        _replayDeaths[rt] = lst;
                    }
                    lst.Add(slot);
                    nD++;
                }
            }

            // Weapon-change rows: [relTick, sid, weapon_name]
            int nW = 0;
            if (req.ReplayChunk.Weapons != null)
            {
                foreach (var row in req.ReplayChunk.Weapons)
                {
                    if (row.Count < 3) continue;
                    int rt = AsInt(row[0]);
                    string sid = row[1]?.ToString() ?? "";
                    string wep = row[2]?.ToString() ?? "";
                    if (!_replaySidToSlot.TryGetValue(sid, out int slot)) continue;
                    if (string.IsNullOrEmpty(wep)) continue;
                    if (!_replayWeapons.TryGetValue(rt, out var lst))
                    {
                        lst = new List<(int, string)>();
                        _replayWeapons[rt] = lst;
                    }
                    lst.Add((slot, wep));
                    nW++;
                }
            }

            tcs.SetResult(new PoseResponse {
                Ok = true,
                Applied = nT + nF + nD + nW,
                Message = $"Chunk: +{nT} ticks, +{nF} fires, +{nD} deaths, +{nW} weapons"
            });
        });
        return tcs.Task;
    }

    private static int AsInt(object o) =>
        o is System.Text.Json.JsonElement je
            ? (je.ValueKind == System.Text.Json.JsonValueKind.Number ? je.GetInt32() : int.Parse(je.GetString() ?? "0"))
            : Convert.ToInt32(o);
    private static float AsFloat(object o) =>
        o is System.Text.Json.JsonElement je
            ? (float)je.GetDouble()
            : Convert.ToSingle(o);

    // Per-slot inventory tracking during replay (avoid RemoveWeapons every event)
    private readonly Dictionary<int, HashSet<string>> _replayBotWeapons = new();
    // True once initial loadout was given (in OnTick first iteration after replay_active=true)
    private bool _replayLoadoutGiven = false;

    /// <summary>Begin replay. Triggers fresh restart with proper round/freeze cvars,
    /// then starts OnTick replay AFTER freeze period elapses.</summary>
    private Task<PoseResponse> ReplayStartOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            // Force key cvars + restart so they actually take effect on running round.
            Server.ExecuteCommand("sv_cheats 1");          // setang/setpos/god require this
            Server.ExecuteCommand("host_timescale 1");
            Server.ExecuteCommand("mp_freezetime 20");
            Server.ExecuteCommand("mp_roundtime 1.92");
            Server.ExecuteCommand("mp_roundtime_defuse 1.92");
            Server.ExecuteCommand("mp_ignore_round_win_conditions 1");
            Server.ExecuteCommand("bot_stop 1");
            Server.ExecuteCommand("bot_zombie 0");
            Server.ExecuteCommand("bot_dont_shoot 1");
            Server.ExecuteCommand("mp_restartgame 1");      // ← reapply freeze + roundtime to HUD

            _replayBotWeapons.Clear();

            // Phase 2: 1.5s after restart → freeze period is running, NOW start replay tick=0.
            // Demo's tick_start = round prestart (i.e. start of freeze). Demo's freeze actions
            // (mostly idle / buying) will play during our HUD freezetime — visually correct.
            AddTimer(1.5f, () =>
            {
                Server.ExecuteCommand("god");
                Server.ExecuteCommand("mp_autoteambalance 0");
                Server.ExecuteCommand("mp_limitteams 0");

                // Move human to required team (no race with restart now — restart already done)
                var human = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
                if (human != null)
                {
                    CsTeam wantTeam = _replayHumanTeam == "t" ? CsTeam.Terrorist
                                       : _replayHumanTeam == "spec" ? CsTeam.Spectator
                                       : CsTeam.CounterTerrorist;
                    if ((int)human.Team != (int)wantTeam)
                    {
                        try { human.ChangeTeam(wantTeam); }
                        catch (Exception ex) { Logger.LogWarning("[Replay] team-move failed: {Err}", ex.Message); }
                        AddTimer(0.3f, () => {
                            try { human.Respawn(); } catch { /* ignore */ }
                        });
                    }
                }

                // Kick extras (auto-balance may have over-spawned)
                AddTimer(0.6f, () => {
                    var bots = Utilities.GetPlayers()
                        .Where(p => p != null && p.IsValid && p.IsBot).ToList();
                    var ctBots = bots.Where(b => b.TeamNum == 3).ToList();
                    var tBots  = bots.Where(b => b.TeamNum == 2).ToList();
                    while (ctBots.Count > _replayExpectCt)
                    {
                        var b = ctBots[ctBots.Count - 1];
                        Logger.LogInformation("[Replay] kicking extra CT bot: {Name}", b.PlayerName);
                        Server.ExecuteCommand($"bot_kick \"{b.PlayerName}\"");
                        ctBots.RemoveAt(ctBots.Count - 1);
                    }
                    while (tBots.Count > _replayExpectT)
                    {
                        var b = tBots[tBots.Count - 1];
                        Logger.LogInformation("[Replay] kicking extra T bot: {Name}", b.PlayerName);
                        Server.ExecuteCommand($"bot_kick \"{b.PlayerName}\"");
                        tBots.RemoveAt(tBots.Count - 1);
                    }
                });

                foreach (var slot in _replaySidToSlot.Values)
                {
                    var bot = Utilities.GetPlayerFromSlot(slot);
                    if (bot?.PlayerPawn?.Value != null)
                        bot.PlayerPawn.Value.Health = 9999;
                }

                // Loadout giving moved to OnTick first-tick handler (server state more stable
                // there than during the restart-phase + freeze-period window we're in now).
                _replayLoadoutGiven = false;

                _replayStartServerTick = Server.TickCount;
                _replayCurrentRelTick = 0;
                _replayActive = true;
                _activePoses.Clear();
                Logger.LogInformation("[Replay] Active @ server tick {Tick}, {Max} ticks to play",
                                      _replayStartServerTick, _replayMaxRelTick);
            });

            tcs.SetResult(new PoseResponse {
                Ok = true,
                Message = "Replay scheduled: 1.5s settle → start tick=0 (during freeze HUD)"
            });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> ReplayStopOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            _replayActive = false;
            // Release any held attacks
            foreach (var slot in _replayActiveAttacks.Keys.ToList())
                ReleaseBotAttack(slot);
            _replayActiveAttacks.Clear();
            tcs.SetResult(new PoseResponse { Ok = true, Message = "Replay stopped" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> ReplayStatusOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            // current_tick = -1 means replay_init done but replay_start AddTimer
            // hasn't fired yet → return 0 so Python doesn't break the loop.
            int curTick = _replayCurrentRelTick < 0 ? 0 : _replayCurrentRelTick;
            var payload = new {
                active        = _replayActive,
                current_tick  = curTick,
                max_tick      = _replayMaxRelTick,
                progress      = _replayMaxRelTick > 0
                                ? (float)curTick / _replayMaxRelTick : 0f,
                dead_slots    = _replayDeadSlots.ToArray(),
                queued_ticks  = _replayPoses.Count,
            };
            tcs.SetResult(new PoseResponse {
                Ok = true,
                Message = JsonSerializer.Serialize(payload),
            });
        });
        return tcs.Task;
    }

    /// <summary>Give all bots + human their full demo loadout. Called once from OnTick when
    /// replay first becomes active, since server state is more stable than during the restart
    /// settle phase.</summary>
    private void GiveInitialLoadouts()
    {
        foreach (var (slot, weps) in _replayInventory)
        {
            var p = Utilities.GetPlayerFromSlot(slot);
            if (p == null || !p.IsValid)
            {
                Logger.LogWarning("[Replay-Loadout] slot {Slot} not valid, skip", slot);
                continue;
            }
            try
            {
                p.RemoveWeapons();
                // Engine auto-gives team-appropriate knife on spawn (weapon_knife for CT,
                // weapon_knife_t for T). Don't try to explicitly add — would duplicate or fail.
                int given = 0;
                foreach (var w in weps)
                {
                    // Skip knife — already on player from spawn
                    if (w.StartsWith("weapon_knife") || w.StartsWith("weapon_bayonet")) continue;
                    var ent = p.GiveNamedItem(w);
                    if (ent != null) given++;
                }
                // Initial active = best non-grenade weapon
                var pawn = p.PlayerPawn?.Value;
                if (pawn != null && weps.Count > 0)
                {
                    // Skip knife/bayonet too — we want PRIMARY (rifle/sniper) first,
                    // then secondary (pistol). Knife is the literal worst pick for "initial active".
                    string initial = weps.FirstOrDefault(w =>
                        !w.Contains("grenade") && !w.Contains("flash") &&
                        !w.Contains("smoke") && !w.Contains("decoy") &&
                        !w.Contains("molotov") && !w.Contains("incgrenade") &&
                        !w.StartsWith("weapon_knife") && !w.StartsWith("weapon_bayonet"))
                        ?? weps.FirstOrDefault(w =>
                            !w.StartsWith("weapon_knife") && !w.StartsWith("weapon_bayonet"))
                        ?? weps[0];
                    bool switched = TrySwitchActiveWeapon(p, pawn, initial);
                    Logger.LogInformation("[Replay-Loadout] slot {Slot} ({Name}): given {N}/{Total} weapons, switched={Sw} → {Init}",
                                           slot, p.PlayerName, given, weps.Count, switched, initial);

                    // For HUMAN: the immediate `use weapon_X` often races GiveNamedItem —
                    // weapon entity isn't in MyWeapons yet, so client ignores the switch and
                    // we end up with the knife, with the rifle dropped on the ground.
                    // Re-issue slot1 + explicit `use` after a short delay so the entity is attached.
                    if (!p.IsBot)
                    {
                        string targetWep = initial;
                        var humanCtrl = p;
                        Logger.LogInformation("[Replay-Loadout] HUMAN slot={Slot} scheduling delayed switch → {Wep}",
                                               slot, targetWep);
                        AddTimer(0.5f, () =>
                        {
                            try
                            {
                                if (humanCtrl == null || !humanCtrl.IsValid)
                                {
                                    Logger.LogWarning("[Replay-Loadout] HUMAN @0.5s: ctrl invalid, skip");
                                    return;
                                }
                                humanCtrl.ExecuteClientCommand("slot1");
                                humanCtrl.ExecuteClientCommand($"use {targetWep}");
                                Logger.LogInformation("[Replay-Loadout] HUMAN @0.5s: sent slot1 + use {Wep}", targetWep);
                            }
                            catch (Exception ex) { Logger.LogWarning("[Replay-Loadout] HUMAN @0.5s err: {E}", ex.Message); }
                        });
                        AddTimer(1.2f, () =>
                        {
                            try
                            {
                                if (humanCtrl == null || !humanCtrl.IsValid)
                                {
                                    Logger.LogWarning("[Replay-Loadout] HUMAN @1.2s: ctrl invalid, skip");
                                    return;
                                }
                                var pawnLate = humanCtrl.PlayerPawn?.Value;
                                if (pawnLate == null)
                                {
                                    Logger.LogWarning("[Replay-Loadout] HUMAN @1.2s: pawn null");
                                    return;
                                }
                                bool ok = TrySwitchActiveWeapon(humanCtrl, pawnLate, targetWep);
                                // Dump current MyWeapons for diagnosis
                                var ws = pawnLate.WeaponServices;
                                var weps = new List<string>();
                                if (ws != null)
                                    foreach (var wH in ws.MyWeapons)
                                    {
                                        var w = wH?.Value;
                                        if (w != null && w.IsValid) weps.Add(w.DesignerName ?? "?");
                                    }
                                Logger.LogInformation("[Replay-Loadout] HUMAN @1.2s: switch={Sw}, target={T}, MyWeapons=[{All}]",
                                                       ok, targetWep, string.Join(", ", weps));
                            }
                            catch (Exception ex) { Logger.LogWarning("[Replay-Loadout] HUMAN @1.2s err: {E}", ex.Message); }
                        });
                    }
                }
                else
                {
                    Logger.LogInformation("[Replay-Loadout] slot {Slot} ({Name}): given {N} weapons (no init)",
                                           slot, p.PlayerName, given);
                }
            }
            catch (Exception ex)
            {
                Logger.LogWarning("[Replay-Loadout] slot {Slot} failed: {Err}", slot, ex.Message);
            }
        }
    }

    /// <summary>Apply replay state at given relative tick. Called from OnTick.</summary>
    private void ApplyReplayTick(int relTick)
    {
        // 1. Position update for all live bots present at this tick
        if (_replayPoses.TryGetValue(relTick, out var slotMap))
        {
            foreach (var (slot, data) in slotMap)
            {
                if (_replayDeadSlots.Contains(slot)) continue;
                var bot = Utilities.GetPlayerFromSlot(slot);
                var pawn = bot?.PlayerPawn?.Value;
                if (pawn == null || !pawn.IsValid) continue;

                bool isHuman = !bot.IsBot;
                // For human: skip V_angle override unless explicitly locked. Constant V_angle
                // overwrites cause client prediction rejection → weapon deploy animations don't render.
                bool overrideView = !isHuman || _replayLockHumanView;

                // Compute velocity from next tick if available — animation interpolates smoothly
                Vector? velocity = null;
                if (_replayPoses.TryGetValue(relTick + 1, out var next) && next.TryGetValue(slot, out var n))
                {
                    velocity = new Vector(
                        (n[0] - data[0]) * 64f,    // 64 ticks/sec
                        (n[1] - data[1]) * 64f,
                        (n[2] - data[2]) * 64f);
                }
                pawn.Teleport(
                    new Vector(data[0], data[1], data[2]),
                    new QAngle(0, data[3], 0),     // body yaw only
                    velocity);
                if (overrideView && pawn.V_angle != null)
                {
                    pawn.V_angle.X = data[4];      // pitch
                    pawn.V_angle.Y = data[3];      // yaw
                    Utilities.SetStateChanged(pawn, "CBasePlayerPawn", "v_angle");
                }

                // Bot view: V_angle write alone doesn't reach the renderer — head bone
                // orientation comes from m_angEyeAngles. Set it explicitly so pitch
                // (look up/down) is actually visible on the model.
                if (!isHuman && pawn.EyeAngles != null)
                {
                    pawn.EyeAngles.X = data[4];    // pitch
                    pawn.EyeAngles.Y = data[3];    // yaw
                    pawn.EyeAngles.Z = 0;
                    Utilities.SetStateChanged(pawn, "CBasePlayerPawn", "m_angEyeAngles");
                }

                // Human view: client prediction throws away server-side V_angle writes.
                // Only setang console command actually drives the human's look direction.
                // Issue per tick when lock is on (animations may glitch — accepted for ML capture).
                if (isHuman && _replayLockHumanView)
                {
                    // Just stash the target — the actual write happens in
                    // OnServerPostEntityThink (after ProcessUsercmds), otherwise
                    // the client's prediction overwrites our value the same tick.
                    float pitchH = Math.Clamp(data[4], -89f, 89f);
                    _humanLockAng = (slot, pitchH, data[3]);

                    if (relTick % 64 == 0)
                        Logger.LogInformation("[Replay-View] human slot={Slot} target pitch={P:F2} yaw={Y:F2} (queued for PostEntityThink)",
                                              slot, pitchH, data[3]);
                }
                else if (isHuman)
                {
                    _humanLockAng = null;
                    if (relTick % 64 == 0)
                        Logger.LogInformation("[Replay-View] human slot={Slot} NOT setting view (lock={Lock})",
                                              slot, _replayLockHumanView);
                }
                // Crouch state from flags bit 1 — apply fully (FL_DUCKING flag + 4 schema fields)
                bool ducking = (((int)data[5]) & 2) != 0;
                const uint FL_DUCKING_BIT = 2;
                uint flags = pawn.Flags;
                uint newFlags = ducking ? (flags | FL_DUCKING_BIT) : (flags & ~FL_DUCKING_BIT);
                if (newFlags != flags)
                {
                    pawn.Flags = newFlags;
                    Utilities.SetStateChanged(pawn, "CBaseEntity", "m_fFlags");
                }
                var ms = pawn.MovementServices;
                if (ms != null && ms.Handle != IntPtr.Zero)
                {
                    Schema.SetSchemaValue<float>(ms.Handle, "CCSPlayer_MovementServices",
                                                  "m_flDuckAmount", ducking ? 1.0f : 0.0f);
                    Schema.SetSchemaValue<bool>(ms.Handle, "CCSPlayer_MovementServices",
                                                  "m_bDucked", ducking);
                    Schema.SetSchemaValue<bool>(ms.Handle, "CCSPlayer_MovementServices",
                                                  "m_bDucking", ducking);
                    Schema.SetSchemaValue<bool>(ms.Handle, "CCSPlayer_MovementServices",
                                                  "m_bDesiresDuck", ducking);
                }

                // Walk (shift) — reduce max speed for slow-walk audio + animation cycle.
                // Default walking speed in CS2 ≈ 130 units/sec, running ≈ 250.
                bool walking = data.Length > 6 && data[6] > 0.5f;
                Schema.SetSchemaValue<float>(pawn.Handle, "CCSPlayerPawn",
                                              "m_flVelocityModifier", walking ? 0.52f : 1.0f);

                pawn.Health = 9999;     // keep alive — explicit deaths via death events
            }
        }

        // 2. Fire events at this tick — start +attack on those slots
        if (_replayFires.TryGetValue(relTick, out var fireSlots))
        {
            foreach (var slot in fireSlots)
            {
                if (_replayDeadSlots.Contains(slot)) continue;
                TriggerBotAttack(slot);
                _replayActiveAttacks[slot] = relTick + 2;    // release after 2 ticks
            }
        }

        // 3. Release fires that have completed their burst
        var toRelease = _replayActiveAttacks.Where(kv => kv.Value <= relTick).Select(kv => kv.Key).ToList();
        foreach (var slot in toRelease)
        {
            ReleaseBotAttack(slot);
            _replayActiveAttacks.Remove(slot);
        }

        // 4. Death events — mark slot dead (stop applying poses, hide bot)
        if (_replayDeaths.TryGetValue(relTick, out var deathSlots))
        {
            foreach (var slot in deathSlots)
            {
                _replayDeadSlots.Add(slot);
                var bot = Utilities.GetPlayerFromSlot(slot);
                if (bot?.PlayerPawn?.Value != null)
                {
                    try { bot.PlayerPawn.Value.CommitSuicide(false, true); } catch { /* ignore */ }
                }
            }
        }

        // 5. Weapon SWITCH — match demo's active weapon. Bot already has full loadout from start;
        // we just switch active. Fallback: if weapon not in initial inventory (mid-round pickup),
        // give it now and switch.
        if (_replayWeapons.TryGetValue(relTick, out var weaponChanges))
        {
            foreach (var (slot, weapon) in weaponChanges)
            {
                if (_replayDeadSlots.Contains(slot)) continue;
                var bot = Utilities.GetPlayerFromSlot(slot);
                if (bot == null || !bot.IsValid) continue;
                var pawn = bot.PlayerPawn?.Value;
                if (pawn == null || !pawn.IsValid) continue;

                string name = NormalizeWeaponName(weapon);
                if (string.IsNullOrEmpty(name)) continue;
                // Knife switches ARE applied — players use them for movement speed in CS

                bool inInventory = _replayInventory.TryGetValue(slot, out var inv) && inv.Contains(name);

                // Try to find the weapon entity in bot's MyWeapons and set as active.
                // For knife, TrySwitchActiveWeapon fuzzy-matches against any weapon_knife*/bayonet* variant.
                bool switched = TrySwitchActiveWeapon(bot, pawn, name);

                if (!switched)
                {
                    // Fallback: not in inventory or switch failed — give it (pickup case)
                    try
                    {
                        bot.GiveNamedItem(name);
                        // Track as added to inventory for next change
                        if (!_replayInventory.ContainsKey(slot))
                            _replayInventory[slot] = new List<string>();
                        if (!_replayInventory[slot].Contains(name))
                            _replayInventory[slot].Add(name);
                        Logger.LogInformation("[Replay] tick {T}: slot {Slot} pickup {Weapon}",
                                               relTick, slot, name);
                    }
                    catch (Exception ex)
                    {
                        Logger.LogWarning("[Replay] give pickup failed slot={Slot} weapon={Weapon}: {Err}",
                                          slot, name, ex.Message);
                    }
                }
            }
        }
    }

    /// <summary>Returns the human player's slot — used so that replay can map a demo sid to it.</summary>
    private Task<PoseResponse> GetHumanSlotOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
            if (viewer == null)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "No human player" });
                return;
            }
            tcs.SetResult(new PoseResponse {
                Ok = true,
                Applied = (int)viewer.Slot,
                Message = JsonSerializer.Serialize(new {
                    slot = (int)viewer.Slot,
                    name = viewer.PlayerName,
                    team = viewer.TeamNum
                })
            });
        });
        return tcs.Task;
    }

    /// <summary>Move human to specified team (2=T, 3=CT) and respawn so they're alive for replay.</summary>
    private Task<PoseResponse> MoveHumanToTeamOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
            if (viewer == null)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "No human player" });
                return;
            }
            string teamStr = (req.Team ?? "ct").ToLowerInvariant();
            CsTeam team = teamStr == "t" ? CsTeam.Terrorist : CsTeam.CounterTerrorist;
            try
            {
                viewer.ChangeTeam(team);
                AddTimer(0.4f, () => {
                    try { viewer.Respawn(); } catch { /* maybe already alive */ }
                });
                tcs.SetResult(new PoseResponse {
                    Ok = true,
                    Message = $"Human moved to team {team}, respawning"
                });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = $"ChangeTeam failed: {ex.Message}" });
            }
        });
        return tcs.Task;
    }

    /// <summary>Engine-level visibility trace: from one eye position to many target points.
    ///
    /// Uses CS2's GameTraceManager (the same routine bullets use) via CS2TraceRay.TraceHull
    /// (we go through TraceHull with a zero-radius Line ray because the simpler TraceShape
    /// signature ("TraceFunc") is broken in post-2026-01 CS2 builds).
    ///
    /// Filter is sourced from the local human player's pawn (so its InteractsWith mask
    /// matches what bullets care about), with HitEntities/Triggers/Hitboxes disabled so
    /// other bots don't shadow each other — visibility stays purely geometric.
    ///
    /// Response JSON: {"visible": [bool], "fractions": [float], "n_visible": int, "n_total": int}
    /// </summary>
    private Task<PoseResponse> TraceVisibilityBatchOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                if (req.From == null || req.From.Length != 3 || req.Targets == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "Need 'from' (3 floats) and 'targets' (list of 3-float arrays)" });
                    return;
                }

                // Source filter from human pawn — its InteractsWith mask is the right
                // shape for "what blocks bullets coming from a player".
                var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
                var pawn = viewer?.PlayerPawn?.Value;
                if (pawn == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "trace_visibility_batch: no human pawn for filter" });
                    return;
                }

                Vector start = new Vector(req.From[0], req.From[1], req.From[2]);
                // tolerance: hit counts as "reached target" if within this many units of it.
                // 30 covers player capsule radius (~16) + some slack so bones inside the
                // hitbox register as visible even at long range where fraction is tight.
                float tolerance = req.Tolerance > 0 ? req.Tolerance : 30.0f;

                var visible   = new List<bool>(req.Targets.Count);
                var fractions = new List<float>(req.Targets.Count);
                var contents  = new List<long>(req.Targets.Count);
                var hitEnts   = new List<long>(req.Targets.Count);
                var hitNames  = new List<string>(req.Targets.Count);  // DesignerName of hit entity
                int nVis = 0;

                // MASK_SHOT: bullet-block mask — solid world + players + npcs + windows
                // + debris + hitbox. Same set the game uses for actual bullet traces, so
                // bots/players in the line of fire correctly shadow each other.
                const ulong MASK_SHOT =
                    (1UL << 0)    // SOLID
                  | (1UL << 1)    // HITBOX
                  | (1UL << 12)   // WINDOW
                  | (1UL << 18)   // PLAYER
                  | (1UL << 19)   // NPC
                  | (1UL << 20);  // DEBRIS
                // Things bullets pass THROUGH but movement collision blocks:
                //   bit  2 = TRIGGER  (func_buyzone, hurt, push, etc.)
                //   bit  4 = PLAYER_CLIP (invisible "no-go" brush for players)
                //   bit  5 = NPC_CLIP
                //   bit 11 = NODRAW (mapper hidden surfaces)
                const ulong MASK_EXCLUDE =
                    (1UL << 2)    // TRIGGER
                  | (1UL << 4)    // PLAYER_CLIP
                  | (1UL << 5)    // NPC_CLIP
                  | (1UL << 11);  // NODRAW

                // Push start 40 units along the ray direction so we clear the viewer's
                // own pawn (player capsule radius ~16 + held-weapon viewmodel reach).
                // EntityIdsToIgnore alone isn't reliable here (pawn.Index format mismatch),
                // but a physical offset always works.
                const float START_NUDGE = 40f;
                Ray ray = new Ray(new SysVec3(0f, 0f, 0f));

                // Pre-resolve skip slots → pawn.Index, parallel to req.Targets.
                // 0 = no extra skip for that target. Used so a per-bone trace can pass
                // through the bot the bone belongs to (otherwise we'd always hit its
                // own collision capsule near fraction~0.97).
                uint[] perTargetSkip = new uint[req.Targets.Count];
                if (req.SkipSlots != null)
                {
                    for (int i = 0; i < req.Targets.Count && i < req.SkipSlots.Count; i++)
                    {
                        int s = req.SkipSlots[i];
                        if (s < 0) { perTargetSkip[i] = 0; continue; }
                        var p = Utilities.GetPlayerFromSlot(s);
                        var pp = p?.PlayerPawn?.Value;
                        perTargetSkip[i] = (pp != null && pp.IsValid) ? pp.Index : 0;
                    }
                }

                for (int ti = 0; ti < req.Targets.Count; ti++)
                {
                    var t = req.Targets[ti];
                    if (t == null || t.Length != 3)
                    {
                        visible.Add(false);
                        fractions.Add(0f);
                        contents.Add(0);
                        hitEnts.Add(0);
                        hitNames.Add("");
                        continue;
                    }

                    float dx = t[0] - req.From[0];
                    float dy = t[1] - req.From[1];
                    float dz = t[2] - req.From[2];
                    float fullDist = (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
                    if (fullDist <= START_NUDGE + 1f) {
                        visible.Add(true); fractions.Add(1f); contents.Add(0); hitEnts.Add(0); hitNames.Add(""); nVis++; continue;
                    }
                    float invLen = 1f / fullDist;
                    Vector startNudged = new Vector(
                        req.From[0] + dx * invLen * START_NUDGE,
                        req.From[1] + dy * invLen * START_NUDGE,
                        req.From[2] + dz * invLen * START_NUDGE);
                    Vector end = new Vector(t[0], t[1], t[2]);

                    // Per-trace filter so we can additionally skip the target bot's pawn
                    // (so trace goes through it and only stops at real walls / other bots).
                    CTraceFilter perFilter = new CTraceFilter(pawn.Index);
                    perFilter.QueryShapeAttributes.InteractsWith    = MASK_SHOT;
                    perFilter.QueryShapeAttributes.InteractsExclude = MASK_EXCLUDE;   // pass through clips/triggers
                    perFilter.QueryShapeAttributes.InteractsAs      = MASK_SHOT;
                    perFilter.QueryShapeAttributes.HitSolid         = true;
                    perFilter.QueryShapeAttributes.HitTrigger       = false;
                    perFilter.QueryShapeAttributes.ObjectSetMask    = 0x0F;
                    perFilter.IterateEntities                     = true;
                    if (perTargetSkip[ti] != 0)
                        unsafe { perFilter.QueryShapeAttributes.EntityIdsToIgnore[1] = perTargetSkip[ti]; }

                    CGameTrace trace = TraceRay.TraceHullManaged(startNudged, end, perFilter, ray);
                    float remaining = fullDist - START_NUDGE;
                    float hitDist = trace.Fraction * remaining;

                    // Treat hits whose CONTENTS are entirely clip/trigger/nodraw as
                    // non-blocking (bullets pass through them). E.g. cont=0x40000030
                    // = player_clip+npc_clip+? from the world entity for invisible brushes.
                    bool nonBlockingHit = trace.Contents != 0
                                          && (trace.Contents & ~MASK_EXCLUDE & 0x7FFFFFFFu) == 0;

                    bool isVis;
                    if (trace.AllSolid)              isVis = true;
                    else if (trace.Fraction >= 1.0f) isVis = true;
                    else if (nonBlockingHit)         isVis = true;
                    else                             isVis = (hitDist + tolerance >= remaining);

                    visible.Add(isVis);
                    fractions.Add(trace.Fraction);
                    contents.Add((long)trace.Contents);
                    hitEnts.Add(trace.HitEntity.ToInt64());

                    // Resolve DesignerName of the hit entity (so we can identify
                    // the mystery occluder by class name).
                    string designer = "";
                    try
                    {
                        if (trace.HitEntity != IntPtr.Zero)
                        {
                            var ent = new CEntityInstance(trace.HitEntity);
                            designer = ent.DesignerName ?? "";
                        }
                    }
                    catch { }
                    hitNames.Add(designer);

                    if (isVis) nVis++;
                }

                var payload = new {
                    visible    = visible,
                    fractions  = fractions,
                    contents   = contents,
                    hit_ents   = hitEnts,
                    hit_names  = hitNames,
                    n_visible  = nVis,
                    n_total    = req.Targets.Count,
                };
                tcs.SetResult(new PoseResponse {
                    Ok = true,
                    Message = JsonSerializer.Serialize(payload),
                });
            }
            catch (Exception ex)
            {
                // Walk InnerException chain — TypeInitializationException hides the real
                // signature-load failure inside ex.InnerException.
                string detail = ex.GetType().Name + ": " + ex.Message;
                Exception? inner = ex.InnerException;
                int depth = 0;
                while (inner != null && depth < 5) {
                    detail += " | " + inner.GetType().Name + ": " + inner.Message;
                    inner = inner.InnerException;
                    depth++;
                }
                Logger.LogError(ex, "[BotPoseControl] trace_visibility_batch threw");
                tcs.SetResult(new PoseResponse { Ok = false, Message = "trace_visibility_batch: " + detail });
            }
        });
        return tcs.Task;
    }

    /// <summary>Move local human player to spectator team and lock to a bot's first-person POV.
    /// User sees that bot's HUD + weapons, like a real spectator FPV.</summary>
    private Task<PoseResponse> ReplaySetViewerOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            int targetSlot = req.Poses?.FirstOrDefault()?.Slot ?? -1;
            var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
            var target = targetSlot >= 0 ? Utilities.GetPlayerFromSlot(targetSlot) : null;
            if (viewer == null)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "No human player" });
                return;
            }
            if (target == null || !target.IsValid || target.PlayerPawn?.Value == null)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = $"Target slot {targetSlot} not valid" });
                return;
            }

            string botName = target.PlayerName ?? $"BOT_slot_{targetSlot}";

            // Move to spectator team — kills current pawn, creates observer pawn.
            try
            {
                viewer.ChangeTeam(CsTeam.Spectator);
            }
            catch (Exception ex)
            {
                Logger.LogWarning("[Replay] ChangeTeam failed: {Err}", ex.Message);
            }

            // After short delay, spec_player to the bot + first-person mode.
            AddTimer(0.6f, () =>
            {
                try
                {
                    viewer.ExecuteClientCommand($"spec_player \"{botName}\"");
                    viewer.ExecuteClientCommand("spec_mode 4");   // 4 = in-eye / first-person
                    Logger.LogInformation("[Replay] Viewer {V} now spectating {B} (slot {S}) in FPV",
                                           viewer.PlayerName, botName, targetSlot);
                }
                catch (Exception ex)
                {
                    Logger.LogWarning("[Replay] spec_player failed: {Err}", ex.Message);
                }
            });

            tcs.SetResult(new PoseResponse {
                Ok = true,
                Message = $"Viewer scheduled to spec {botName} (slot {targetSlot}) in first-person"
            });
        });
        return tcs.Task;
    }

    /// <summary>Switch a player's active weapon. Uses 'use weapon_X' command which triggers
    /// proper Holster→Deploy animation. Setting ActiveWeapon ptr alone often skips deploy.
    /// For bots: bot_command "name" "use weapon_X". For human: ExecuteClientCommand.
    /// Also sets ActiveWeapon raw pointer as backup.
    ///
    /// Knife handling: T players have weapon_knife_t (default), CT have weapon_knife.
    /// Demo collapses all knives → "weapon_knife", so we fuzzy-match any knife variant.</summary>
    private static bool TrySwitchActiveWeapon(CCSPlayerController controller, CCSPlayerPawn pawn,
                                                string weaponName)
    {
        if (controller == null || pawn == null) return false;
        var ws = pawn.WeaponServices;
        if (ws == null) return false;

        bool wantKnife = weaponName.StartsWith("weapon_knife") ||
                         weaponName.StartsWith("weapon_bayonet");

        // CS2 maps several demo classnames to a single in-game entity (e.g. demo says
        // weapon_m4a1_silencer but GiveNamedItem creates entity with DesignerName=weapon_m4a1).
        // We allow a small set of equivalences so the lookup actually finds the entity.
        static bool WeaponEquiv(string demo, string designer)
        {
            if (demo == designer) return true;
            // Strip common suffixes for both, compare bases.
            string a = demo.Replace("_silencer", "");
            string b = designer.Replace("_silencer", "");
            if (a == b) return true;
            // M4 family: m4a1 ↔ m4a1_silencer
            if ((demo == "weapon_m4a1" || demo == "weapon_m4a1_silencer") &&
                (designer == "weapon_m4a1" || designer == "weapon_m4a1_silencer")) return true;
            // USP/P2000: weapon_usp_silencer ↔ weapon_hkp2000 (CT default pistol slot)
            if ((demo == "weapon_usp_silencer" || demo == "weapon_hkp2000") &&
                (designer == "weapon_usp_silencer" || designer == "weapon_hkp2000")) return true;
            return false;
        }

        bool found = false;
        try
        {
            // Set raw pointer (replicates immediately)
            foreach (var wHandle in ws.MyWeapons)
            {
                var w = wHandle?.Value;
                if (w == null || !w.IsValid) continue;
                string designer = w.DesignerName ?? "";
                bool match = WeaponEquiv(weaponName, designer) ||
                             (wantKnife && (designer.StartsWith("weapon_knife") ||
                                            designer.StartsWith("weapon_bayonet")));
                if (match)
                {
                    ws.ActiveWeapon.Raw = wHandle.Raw;
                    Utilities.SetStateChanged(pawn, "CCSPlayer_WeaponServices", "m_hActiveWeapon");
                    found = true;
                    weaponName = designer;   // use exact (designer) name in 'use' command below
                    break;
                }
            }
        }
        catch { /* continue to use-command */ }

        // Always issue use-command — it triggers proper Deploy animation client-side
        try
        {
            if (controller.IsBot)
            {
                // bot_command "BotName" "use weapon_X"
                Server.ExecuteCommand($"bot_command \"{controller.PlayerName}\" \"use {weaponName}\"");
            }
            else
            {
                controller.ExecuteClientCommand($"use {weaponName}");
            }
        }
        catch { /* ignore, raw set is fallback */ }

        return found;
    }

    /// <summary>Demo-source weapon names: 'ak47', 'CKnife', 'weapon_glock', 'CWeaponAK47' etc.
    /// Normalize to 'weapon_ak47' format that GiveNamedItem expects.</summary>
    private static string NormalizeWeaponName(string raw)
    {
        if (string.IsNullOrEmpty(raw)) return "";
        string s = raw.Trim();
        if (s.StartsWith("CWeapon")) s = s.Substring(7).ToLowerInvariant();
        else if (s.StartsWith("CKnife")) s = "knife";
        else if (s.StartsWith("C")) s = s.Substring(1).ToLowerInvariant();
        if (!s.StartsWith("weapon_")) s = "weapon_" + s;
        return s;
    }

    private void TriggerBotAttack(int slot)
    {
        // V2 TODO: trigger fire on a specific bot.
        //   - CSSharp doesn't expose pawn.Buttons directly in this version.
        //   - Possible paths: schema 'm_pButtonPressedCmd' or weapon's native PrimaryAttack.
        //   - For V1 we only replay positions + deaths; bots don't actually shoot.
        //     Visual reproduction of movement is the key Phase 1 verification.
    }

    private void ReleaseBotAttack(int slot) { /* V2 TODO — see TriggerBotAttack */ }

    private Task<PoseResponse> RespawnBotsOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            if (req.Poses == null || req.Poses.Count == 0)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "No poses (need slot)" });
                return;
            }
            int respawned = 0;
            foreach (var p in req.Poses)
            {
                var bot = Utilities.GetPlayerFromSlot(p.Slot);
                if (bot == null || !bot.IsValid) continue;
                try
                {
                    bot.PlayerPawn?.Value?.CommitSuicide(false, true);
                    respawned++;
                }
                catch { /* ignore */ }
            }
            // Wait one frame, then respawn all killed bots
            Server.NextFrame(() =>
            {
                foreach (var p in req.Poses)
                {
                    var bot = Utilities.GetPlayerFromSlot(p.Slot);
                    if (bot == null || !bot.IsValid) continue;
                    try { bot.Respawn(); } catch { /* ignore */ }
                }
                tcs.SetResult(new PoseResponse {
                    Ok = true, Applied = respawned,
                    Message = $"Respawned {respawned} bots"
                });
            });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> SetBotHealthOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            if (req.Poses == null || req.Poses.Count == 0)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "No poses (need slot+hp)" });
                return;
            }
            int updated = 0;
            foreach (var p in req.Poses)
            {
                var bot = Utilities.GetPlayerFromSlot(p.Slot);
                if (bot?.PlayerPawn?.Value == null) continue;
                int hp = p.Hp ?? 100;
                bot.PlayerPawn.Value.Health = hp;

                // Also patch active pose so OnTick keeps re-applying this hp
                if (_activePoses.TryGetValue(p.Slot, out var existing))
                    existing.Hp = hp;
                updated++;
            }
            tcs.SetResult(new PoseResponse { Ok = true, Applied = updated, Message = $"HP set on {updated} bots" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> StartImpactRecord()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            _impactRecords.Clear();
            _impactRecording = true;
            tcs.SetResult(new PoseResponse { Ok = true, Message = "Impact recording started" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> StopImpactRecord()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            _impactRecording = false;
            tcs.SetResult(new PoseResponse {
                Ok = true,
                Applied = _impactRecords.Count,
                Message = $"Stopped, {_impactRecords.Count} impacts recorded"
            });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> ClearImpacts()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            _impactRecords.Clear();
            tcs.SetResult(new PoseResponse { Ok = true, Message = "Impacts cleared" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> GetImpacts()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            var impacts = _impactRecords
                .Select(i => new { tick = i.Tick, x = i.X, y = i.Y, z = i.Z })
                .ToList();
            tcs.SetResult(new PoseResponse {
                Ok = true,
                Applied = impacts.Count,
                Message = JsonSerializer.Serialize(new { impacts }),
            });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> ScanAimPunchOffsetOnGameThread()
    {
        // Scan for m_pAimPunchServices: pointer in pawn, dereferenced struct has
        // m_predictableBaseAngle (Vec3, deg) at offset 0x50.
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
                var pawn = viewer?.PlayerPawn?.Value;
                if (pawn == null) { tcs.SetResult(new PoseResponse { Ok = false, Message = "No pawn" }); return; }

                const int INNER = 0x50;  // m_predictableBaseAngle inside CCSPlayer_AimPunchServices
                var matches = new List<object>();
                for (long off = 0x100; off < 0x2000; off += 8)
                {
                    IntPtr ptrAddr = IntPtr.Add(pawn.Handle, (int)off);
                    if (!IsReadable(ptrAddr, IntPtr.Size)) continue;
                    IntPtr targetPtr = Marshal.ReadIntPtr(ptrAddr);
                    if (targetPtr == IntPtr.Zero) continue;

                    IntPtr vecAddr = IntPtr.Add(targetPtr, INNER);
                    if (!IsReadable(vecAddr, 12)) continue;

                    float p = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(vecAddr));
                    float y = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(vecAddr + 4));
                    float r = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(vecAddr + 8));

                    if (float.IsNaN(p) || float.IsNaN(y) || float.IsNaN(r)) continue;
                    if (float.IsInfinity(p) || float.IsInfinity(y) || float.IsInfinity(r)) continue;
                    if (Math.Abs(p) > 30 || Math.Abs(y) > 30 || Math.Abs(r) > 5) continue;
                    if (Math.Abs(p) > 0 && Math.Abs(p) < 1e-10) continue;
                    if (Math.Abs(y) > 0 && Math.Abs(y) < 1e-10) continue;

                    matches.Add(new
                    {
                        offset_in_pawn = $"0x{off:X}",
                        target_ptr = $"0x{(long)targetPtr:X}",
                        vec_at_0x50 = new[] { p, y, r },
                    });
                }

                tcs.SetResult(new PoseResponse {
                    Ok = true,
                    Applied = matches.Count,
                    Message = JsonSerializer.Serialize(new {
                        pawn_handle = $"0x{(long)pawn.Handle:X}",
                        candidates = matches,
                    }, new JsonSerializerOptions { WriteIndented = true }),
                });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Error: " + ex.Message });
            }
        });
        return tcs.Task;
    }

    private Task<PoseResponse> ScanPunchOffsetOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
                var pawn = viewer?.PlayerPawn?.Value;
                if (pawn == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "No pawn" });
                    return;
                }

                // Scan offsets in pawn struct, look for pointers that lead to a Vec3
                // at +0x48 with all-zero values (punch_angle is 0 when not firing).
                // Then we re-check after firing to confirm it changes.
                var matches = new List<object>();
                for (long off = 0x100; off < 0x2000; off += 8)
                {
                    IntPtr ptrAddr = IntPtr.Add(pawn.Handle, (int)off);
                    if (!IsReadable(ptrAddr, IntPtr.Size)) continue;
                    IntPtr targetPtr = Marshal.ReadIntPtr(ptrAddr);
                    if (targetPtr == IntPtr.Zero) continue;

                    // Try to read Vec3 at targetPtr + 0x48 (m_vecCsViewPunchAngle offset)
                    IntPtr vecAddr = IntPtr.Add(targetPtr, 0x48);
                    if (!IsReadable(vecAddr, 12)) continue;

                    float p = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(vecAddr));
                    float y = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(vecAddr + 4));
                    float r = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(vecAddr + 8));

                    // Filter: only finite values in plausible degree range
                    if (float.IsNaN(p) || float.IsNaN(y) || float.IsNaN(r)) continue;
                    if (float.IsInfinity(p) || float.IsInfinity(y) || float.IsInfinity(r)) continue;
                    // Plausible punch values: pitch ±20, yaw ±10, roll ~0
                    if (Math.Abs(p) > 20 || Math.Abs(y) > 10 || Math.Abs(r) > 1) continue;
                    // Skip clearly garbage denormal floats
                    if (Math.Abs(p) > 0 && Math.Abs(p) < 1e-10) continue;
                    if (Math.Abs(y) > 0 && Math.Abs(y) < 1e-10) continue;

                    matches.Add(new
                    {
                        offset_in_pawn = $"0x{off:X}",
                        target_ptr = $"0x{(long)targetPtr:X}",
                        vec_at_0x48 = new[] { p, y, r },
                    });
                }

                tcs.SetResult(new PoseResponse {
                    Ok = true,
                    Applied = matches.Count,
                    Message = JsonSerializer.Serialize(new {
                        pawn_handle = $"0x{(long)pawn.Handle:X}",
                        candidates = matches,
                    }, new JsonSerializerOptions { WriteIndented = true }),
                });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Error: " + ex.Message });
            }
        });
        return tcs.Task;
    }

    private Task<PoseResponse> StartPunchRecord()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            _punchSamples.Clear();
            _diagTicks = _diagNoViewer = _diagNoPawn = 0;
            _diagBadCamPtr = _diagNullCam = _diagBadPunchPtr = _diagNaN = 0;
            _punchRecording = true;
            tcs.SetResult(new PoseResponse { Ok = true, Message = "Punch recording started" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> StopPunchRecord()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            _punchRecording = false;
            string diag = $"ticks={_diagTicks} noViewer={_diagNoViewer} noPawn={_diagNoPawn} " +
                          $"badCamPtr={_diagBadCamPtr} nullCam={_diagNullCam} " +
                          $"badPunchPtr={_diagBadPunchPtr} NaN={_diagNaN}";
            tcs.SetResult(new PoseResponse {
                Ok = true,
                Applied = _punchSamples.Count,
                Message = $"Stopped, {_punchSamples.Count} samples. {diag}"
            });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> StartPunchRecordReset()
    {
        // wrapper that also resets diagnostics
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            _diagTicks = _diagNoViewer = _diagNoPawn = 0;
            _diagBadCamPtr = _diagNullCam = _diagBadPunchPtr = _diagNaN = 0;
            tcs.SetResult(new PoseResponse { Ok = true });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> GetPunchRecord()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            var samples = _punchSamples.Select(s => new[] { (float)s.Tick, s.Pitch, s.Yaw }).ToList();
            tcs.SetResult(new PoseResponse {
                Ok = true,
                Applied = samples.Count,
                Message = JsonSerializer.Serialize(new { samples }),
            });
        });
        return tcs.Task;
    }

    /// <summary>Release ALL buttons that we might have set + reset timescale.
    /// Call after each spray test to prevent stuck input state.</summary>
    private Task<PoseResponse> CleanupInputsOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            // Release every common +command multiple times (button state can get queued)
            string[] releases = { "-attack", "-attack2", "-reload", "-use",
                                  "-jump", "-duck", "-speed", "-forward", "-back",
                                  "-moveleft", "-moveright" };
            for (int i = 0; i < 3; i++)  // triple-tap to ensure release
            {
                foreach (var c in releases)
                    Server.ExecuteCommand(c);
            }
            Server.ExecuteCommand("host_timescale 1");
            tcs.SetResult(new PoseResponse { Ok = true, Message = "Inputs cleaned" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> SetAttackOnGameThread(bool start)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            // +attack/-attack work for the local listen-server player on console.
            Server.ExecuteCommand(start ? "+attack" : "-attack");
            tcs.SetResult(new PoseResponse { Ok = true, Message = start ? "+attack" : "-attack" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> EnsureAliveOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
                if (viewer == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "No human player" });
                    return;
                }
                var pawn = viewer.PlayerPawn?.Value;
                bool wasDead = pawn == null || !pawn.IsValid || pawn.Health <= 0;
                if (wasDead)
                {
                    // Player is dead — restart round to respawn on current team.
                    // Avoid jointeam (causes self-kill if switching teams).
                    Server.ExecuteCommand("mp_restartround 1");
                }
                Server.ExecuteCommand("god");
                Server.ExecuteCommand("noclip 0");
                tcs.SetResult(new PoseResponse {
                    Ok = true,
                    Message = wasDead ? "Player was dead — respawn triggered + god re-applied" : "Player alive, god re-applied",
                });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Error: " + ex.Message });
            }
        });
        return tcs.Task;
    }

    private Task<PoseResponse> SetHostTimescaleOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            float scale = req.Scale ?? 1f;
            Server.ExecuteCommand($"host_timescale {scale:F4}");
            tcs.SetResult(new PoseResponse { Ok = true, Message = $"timescale={scale}" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> GetBonesOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            try
            {
                // Pick bot: explicit slot if given, otherwise first alive bot.
                CCSPlayerController? bot = null;
                if (req.Poses != null && req.Poses.Count > 0)
                    bot = Utilities.GetPlayerFromSlot(req.Poses[0].Slot);
                else
                    bot = Utilities.GetPlayers().FirstOrDefault(p => p?.IsBot == true && p.IsValid && (p.PlayerPawn?.Value?.Health ?? 0) > 0);

                if (bot == null || bot.PlayerPawn?.Value == null)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "No suitable bot found" });
                    return;
                }
                var pawn = bot.PlayerPawn.Value;

                var sceneNode = pawn.CBodyComponent?.SceneNode;
                if (sceneNode == null || sceneNode.Handle == IntPtr.Zero)
                {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "No SceneNode on pawn" });
                    return;
                }

                // CSkeletonInstance.m_modelState is a CModelState struct (sub-object, not pointer).
                // Within CModelState: m_pBoneArrayForCode (IntPtr), m_nBoneArrayCount or m_nBoneCount (int).
                var debug = new Dictionary<string, object>();
                int msOffset, boneArrayOffset, boneCountOffset;
                try {
                    msOffset        = Schema.GetSchemaOffset("CSkeletonInstance", "m_modelState");
                    debug["msOffset"] = $"0x{msOffset:X}";
                } catch (Exception ex) {
                    tcs.SetResult(new PoseResponse { Ok = false, Message = "GetSchemaOffset(CSkeletonInstance.m_modelState) failed: " + ex.Message });
                    return;
                }

                // Collect candidate offsets for ALL likely field names — log them all
                var ptrCandidates = new Dictionary<string, object>();
                foreach (var name in new[] {
                    "m_pBoneArrayForCode", "m_pBoneArray", "m_pBones", "m_pBoneArrayBuffer",
                    "m_pBoneArrayCallback", "m_modelState", "m_pBoneTransforms"
                })
                {
                    try {
                        int off = Schema.GetSchemaOffset("CModelState", name);
                        IntPtr v = Marshal.ReadIntPtr(IntPtr.Add(sceneNode.Handle, msOffset + off));
                        ptrCandidates[name] = new { offset = $"0x{off:X}", value = $"0x{(long)v:X}" };
                    } catch (Exception) { /* skip */ }
                }
                debug["ptrCandidates"] = ptrCandidates;

                var intCandidates = new Dictionary<string, object>();
                foreach (var name in new[] {
                    "m_nBoneCount", "m_nBoneArrayCount", "m_boneCount", "m_nBones",
                    "m_nIdealMotionType", "m_nForceLOD", "m_nClothUpdateFlags",
                    "m_MeshGroupMask"
                })
                {
                    try {
                        int off = Schema.GetSchemaOffset("CModelState", name);
                        int v = Marshal.ReadInt32(IntPtr.Add(sceneNode.Handle, msOffset + off));
                        intCandidates[name] = new { offset = $"0x{off:X}", value = v };
                    } catch (Exception) { /* skip */ }
                }
                debug["intCandidates"] = intCandidates;

                // CS2 build 2000807: bone array pointer is at sceneNode + 0x1B0,
                // pointing to flat CTransform[N] array (32 bytes each, world-space).
                // Discovered via memory scan; if CS2 updates, re-run the scanner to refresh offset.
                const int BONE_ARRAY_PTR_OFFSET = 0x1B0;
                const int BONE_STRIDE           = 32;

                var origin = pawn.AbsOrigin!;
                float ox = origin.X, oy = origin.Y, oz = origin.Z;

                // Read bone array if discovery mode not requested
                bool discover = req.Module == "scan";
                if (!discover)
                {
                    IntPtr bonePtrAddr = IntPtr.Add(sceneNode.Handle, BONE_ARRAY_PTR_OFFSET);
                    if (!IsReadable(bonePtrAddr, IntPtr.Size))
                    {
                        tcs.SetResult(new PoseResponse { Ok = false, Message = "Bone ptr address unreadable" });
                        return;
                    }
                    IntPtr boneArrayPtr = Marshal.ReadIntPtr(bonePtrAddr);
                    if (boneArrayPtr == IntPtr.Zero)
                    {
                        tcs.SetResult(new PoseResponse { Ok = false, Message = "Bone array ptr is null" });
                        return;
                    }

                    // We don't have a reliable bone count source — read up to MAX_BONES, validate each transform.
                    // CS2 player models typically have ~100-130 bones.
                    const int MAX_BONES = 200;
                    int requiredBytes = MAX_BONES * BONE_STRIDE;
                    if (!IsReadable(boneArrayPtr, requiredBytes))
                    {
                        // try smaller
                        for (int n = MAX_BONES; n > 0; n /= 2)
                        {
                            if (IsReadable(boneArrayPtr, n * BONE_STRIDE))
                            {
                                requiredBytes = n * BONE_STRIDE;
                                break;
                            }
                        }
                    }
                    int actualMax = requiredBytes / BONE_STRIDE;

                    static bool IsFinite_(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

                    var bonesOut = new List<object>();
                    int validRun = 0;
                    for (int i = 0; i < actualMax; i++)
                    {
                        IntPtr addr = IntPtr.Add(boneArrayPtr, i * BONE_STRIDE);
                        float bx = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(addr));
                        float by = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(addr + 4));
                        float bz = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(addr + 8));
                        if (!IsFinite_(bx) || !IsFinite_(by) || !IsFinite_(bz))
                        {
                            // Stop on first invalid: probably past the end of the bone array.
                            break;
                        }
                        // Sanity: bones should be within ~500 units of origin
                        if (Math.Abs(bx - ox) > 500 || Math.Abs(by - oy) > 500 || Math.Abs(bz - oz) > 500)
                            break;
                        bonesOut.Add(new { idx = i, pos = new[] { bx, by, bz } });
                        validRun = i + 1;
                    }

                    tcs.SetResult(new PoseResponse {
                        Ok      = true,
                        Applied = validRun,
                        Message = JsonSerializer.Serialize(new {
                            slot       = bot.Slot,
                            team       = bot.TeamNum == 3 ? "ct" : "t",
                            origin     = new[] { ox, oy, oz },
                            bone_count = validRun,
                            bones      = bonesOut,
                        }),
                    });
                    return;
                }

                // --- Discovery mode (req.Module == "scan"): full memory scan ---

                static bool IsFinite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

                var matches = new List<object>();
                for (long scanOff = 0x100; scanOff < 0x600; scanOff += 8)
                {
                    IntPtr ptrLocation = IntPtr.Add(sceneNode.Handle, (int)scanOff);
                    if (!IsReadable(ptrLocation, IntPtr.Size)) continue;

                    IntPtr testPtr = Marshal.ReadIntPtr(ptrLocation);
                    if (testPtr == IntPtr.Zero) continue;
                    if (!IsReadable(testPtr, 64)) continue;

                    float bx = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(testPtr));
                    float by = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(testPtr + 4));
                    float bz = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(testPtr + 8));
                    if (!IsFinite(bx) || !IsFinite(by) || !IsFinite(bz)) continue;

                    bool worldSpaceMatch = Math.Abs(bx - ox) < 5 && Math.Abs(by - oy) < 5 && Math.Abs(bz - oz) < 100;
                    bool modelSpaceMatch = Math.Abs(bx) < 5 && Math.Abs(by) < 5 && Math.Abs(bz) < 5;

                    if (worldSpaceMatch || modelSpaceMatch)
                    {
                        float b1x = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(testPtr + 32));
                        float b1y = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(testPtr + 36));
                        float b1z = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(testPtr + 40));
                        if (!IsFinite(b1x) || !IsFinite(b1y) || !IsFinite(b1z)) continue;

                        matches.Add(new
                        {
                            offset_in_skel = $"0x{scanOff:X}",
                            ptr            = $"0x{(long)testPtr:X}",
                            space          = worldSpaceMatch ? "world" : "model",
                            bone0          = new[] { bx, by, bz },
                            bone1          = new[] { b1x, b1y, b1z },
                        });
                    }
                }

                tcs.SetResult(new PoseResponse {
                    Ok      = true,
                    Applied = matches.Count,
                    Message = JsonSerializer.Serialize(new {
                        slot       = bot.Slot,
                        sceneNodeHandle = $"0x{(long)sceneNode.Handle:X}",
                        bot_origin = new[] { ox, oy, oz },
                        matches    = matches,
                        debug      = debug,
                    }, new JsonSerializerOptions { WriteIndented = true }),
                });
                return;
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = "Error: " + ex.Message });
            }
        });
        return tcs.Task;
    }

    private Task<PoseResponse> RestartRoundOnGameThread()
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        Server.NextFrame(() =>
        {
            Server.ExecuteCommand("mp_restartgame 1");
            tcs.SetResult(new PoseResponse { Ok = true, Message = "Round restarted" });
        });
        return tcs.Task;
    }

    private Task<PoseResponse> PrepareMatchOnGameThread(PoseRequest req)
    {
        var tcs = new TaskCompletionSource<PoseResponse>();
        int nCt = Math.Max(0, req.NCt);
        int nT  = Math.Max(0, req.NT);

        // Phase 1: kick + clear poses + apply settings
        Server.NextFrame(() =>
        {
            try
            {
                _activePoses.Clear();
                Server.ExecuteCommand("bot_kick");

                var settings = new[]
                {
                    "sv_cheats 1",
                    "mp_autoteambalance 0",
                    "mp_limitteams 0",
                    "mp_freezetime 0",
                    "mp_warmuptime 0",
                    "mp_warmup_end",
                    "mp_team_intro_time 0",            // skip team intro cinematic
                    "mp_roundtime 60",
                    "mp_roundtime_defuse 60",
                    "mp_roundtime_hostage 60",
                    "mp_round_restart_delay 0",
                    "mp_match_end_restart 0",
                    "mp_ignore_round_win_conditions 1",
                    "mp_buy_anywhere 1",
                    "sv_infinite_ammo 1",              // infinite mag — no reload needed
                    "bot_quota 0",
                    "bot_quota_mode normal",
                    "bot_stop 1",
                    "bot_dont_shoot 1",
                    // bot_zombie removed — it suppresses animation updates,
                    // leaving bones at local (0,0,0). bot_stop+dont_shoot is enough.
                    "god",
                    "cl_drawhud 1",
                    "r_drawviewmodel 1",
                };
                foreach (var cmd in settings) Server.ExecuteCommand(cmd);

                Server.ExecuteCommand("mp_restartgame 1");

                // Phase 2: after restart settles (~1.5 sec), re-apply god to player + spawn bots
                AddTimer(1.5f, () =>
                {
                    // Re-apply player invincibility AFTER restart (god mode is reset on respawn)
                    Server.ExecuteCommand("god");
                    Server.ExecuteCommand("sv_infinite_ammo 1");

                    for (int i = 0; i < nCt; i++) Server.ExecuteCommand("bot_add_ct");
                    for (int i = 0; i < nT;  i++) Server.ExecuteCommand("bot_add_t");

                    // Re-apply bot_stop after spawn (new bots inherit defaults)
                    AddTimer(1.0f, () =>
                    {
                        Server.ExecuteCommand("bot_stop 1");
                        Server.ExecuteCommand("bot_dont_shoot 1");
                        // bot_zombie removed (suppresses animation update)
                        // Triple-down on god: AddTimer fires after restart fully settles.
                        Server.ExecuteCommand("god");
                        tcs.SetResult(new PoseResponse
                        {
                            Ok = true,
                            Applied = nCt + nT,
                            Message = $"Match ready: kicked old bots, applied settings, spawned {nCt} CT + {nT} T"
                        });
                    });
                });
            }
            catch (Exception ex)
            {
                tcs.SetResult(new PoseResponse { Ok = false, Message = ex.Message });
            }
        });

        return tcs.Task;
    }

    // ---------------------------- Pose application ---------------------------- //

    // Source SDK player flags
    private const uint FL_ONGROUND = 1 << 0;  // 1
    private const uint FL_DUCKING  = 1 << 1;  // 2

    // After N OnTick re-applies without landing, force-crouch as last resort
    // (some demo positions are in tight corridors where standing hitbox cannot fit).
    private const int AUTO_CROUCH_AFTER_TICKS = 32;  // ~0.5 sec at 64Hz

    private void ApplyPose(BotPose pose, bool initialApply = false)
    {
        var bot = Utilities.GetPlayerFromSlot(pose.Slot);
        if (bot == null || !bot.IsValid)
            throw new Exception($"Bot in slot {pose.Slot} not found");

        var pawn = bot.PlayerPawn?.Value;
        if (pawn == null || !pawn.IsValid)
            throw new Exception($"Pawn for slot {pose.Slot} not valid");

        // Pitch clamped to realistic view range (humans cannot look beyond 89°)
        var pitch = Math.Clamp(pose.Pitch, -89f, 89f);

        // Auto-crouch fallback: if bot has been failing to land for too long,
        // force ducking — most often this means hitbox doesn't fit between demo Z and ceiling.
        bool wantsDuck = pose.Ducking;
        if (!wantsDuck && !pose.AutoCrouchTried && !initialApply)
        {
            bool onGround = (pawn.Flags & FL_ONGROUND) != 0;
            if (!onGround && pose.TicksSinceApply >= AUTO_CROUCH_AFTER_TICKS)
            {
                wantsDuck = true;
                pose.AutoCrouchTried = true;
                Logger.LogWarning(
                    "[BotPoseControl] Slot {Slot} could not land standing after {Ticks} ticks — auto-crouching",
                    pose.Slot, pose.TicksSinceApply);
            }
        }

        // Position: full teleport on first apply (initialApply=true);
        // on OnTick re-apply, keep X/Y but use bot's CURRENT Z so gravity can settle the bot to ground.
        Vector? newPos = null;
        if (pose.Pos != null && pose.Pos.Length >= 3)
        {
            float x = pose.Pos[0], y = pose.Pos[1], z = pose.Pos[2];
            if (!initialApply && pawn.AbsOrigin != null)
                z = pawn.AbsOrigin.Z;
            newPos = new Vector(x, y, z);
        }

        // Body rotation: only yaw — humans don't lie down. Pitch goes only into V_angle.
        var bodyAng = new QAngle(0, pose.Yaw, 0);

        // Velocity:
        //   initial apply  → strong downward kick so bot falls fast onto the ground;
        //   re-apply       → null = preserve current velocity, so gravity accumulates naturally.
        //                    When bot hits the ground, normal force from the floor zeroes it out.
        Vector? velocity = initialApply
            ? new Vector(0, 0, -500)
            : (Vector?)null;

        pawn.Teleport(newPos, bodyAng, velocity);

        // Crouch state — must be done AFTER teleport so flag isn't reset.
        // Setting FL_DUCKING flag alone is NOT sufficient in CS2 — the engine reads ducking state
        // from CCSPlayer_MovementServices schema fields (m_flDuckAmount, m_bDucked, m_bDucking).
        // We set all of: flag (collision/AI), duck amount (visuals/hitbox), ducked state (camera height).
        uint flags = pawn.Flags;
        if (wantsDuck) flags |= FL_DUCKING;
        else           flags &= ~FL_DUCKING;
        if (flags != pawn.Flags)
        {
            pawn.Flags = flags;
            Utilities.SetStateChanged(pawn, "CBaseEntity", "m_fFlags");
        }

        var ms = pawn.MovementServices;
        if (ms != null && ms.Handle != IntPtr.Zero)
        {
            Schema.SetSchemaValue<float>(ms.Handle, "CCSPlayer_MovementServices", "m_flDuckAmount",  wantsDuck ? 1.0f : 0.0f);
            Schema.SetSchemaValue<bool> (ms.Handle, "CCSPlayer_MovementServices", "m_bDucked",       wantsDuck);
            Schema.SetSchemaValue<bool> (ms.Handle, "CCSPlayer_MovementServices", "m_bDucking",      wantsDuck);
            Schema.SetSchemaValue<bool> (ms.Handle, "CCSPlayer_MovementServices", "m_bDesiresDuck",  wantsDuck);
        }

        // View angle (where bot is aiming/looking) — separate from body rotation.
        if (pawn.V_angle != null)
        {
            pawn.V_angle.X = pitch;
            pawn.V_angle.Y = pose.Yaw;
            pawn.V_angle.Z = 0;
            Utilities.SetStateChanged(pawn, "CBasePlayerPawn", "v_angle");
        }

        // Only set HP if explicitly requested in pose. Otherwise let bot take damage
        // and die naturally — matches real game distribution (death after kill scenario).
        if (pose.Hp.HasValue && pose.Hp.Value > 0)
            pawn.Health = pose.Hp.Value;

        // Armor / helmet — only applied on initial pose set (not every tick), to avoid
        // resetting the value as it's depleted by bullet damage.
        if (initialApply)
        {
            if (pose.Armor.HasValue)
            {
                pawn.ArmorValue = pose.Armor.Value;
                Utilities.SetStateChanged(pawn, "CCSPlayerPawn", "m_ArmorValue");
            }
            if (pose.Helmet.HasValue)
            {
                try
                {
                    Schema.SetSchemaValue<bool>(pawn.ItemServices!.Handle,
                        "CCSPlayer_ItemServices", "m_bHasHelmet", pose.Helmet.Value);
                }
                catch { /* ItemServices unavailable on this entity — skip */ }
            }
        }

        if (initialApply)
        {
            pose.TicksSinceApply  = 0;
            pose.AutoCrouchTried  = false;
        }
        else
        {
            pose.TicksSinceApply++;
        }
    }

    // ---------- Human view override (post-think) ---------- //

    /// <summary>Fires after ProcessUsercmds — the only point where we can write
    /// over the angles the client just sent and have it reach the snapshot.
    /// Stashed value comes from ApplyReplayTick / OnTick path.</summary>
    private int _humanViewLogCounter = 0;
    private void OnPostEntityThink()
    {
        if (!_replayActive || !_replayLockHumanView || _humanLockAng == null)
        {
            HumanViewHook.Disable();
            return;
        }
        var (slot, pitch, yaw) = _humanLockAng.Value;

        var ctrl = Utilities.GetPlayerFromSlot(slot);
        if (ctrl == null || !ctrl.IsValid) { HumanViewHook.Disable(); return; }
        var pawn = ctrl.PlayerPawn?.Value;
        if (pawn == null || !pawn.IsValid) { HumanViewHook.Disable(); return; }

        // PRIMARY mechanism — native detour on CBasePlayerPawn::GetEyeAngles. The
        // renderer queries this getter every frame; our detour returns the demo
        // angle directly, bypassing replication / client prediction layers that
        // ignored every server-side schema write we tried.
        HumanViewHook.Update(pawn.Handle, pitch, yaw);

        // Belt-and-suspenders: also keep server-side schema fields in sync so any
        // OTHER subsystem (server-authoritative AI, demo recording) still sees
        // the right angles. These writes used to be the only mechanism — now
        // they're just informational backups.
        if (pawn.V_angle != null)
        {
            pawn.V_angle.X = pitch;
            pawn.V_angle.Y = yaw;
            pawn.V_angle.Z = 0;
            Utilities.SetStateChanged(pawn, "CBasePlayerPawn", "v_angle");
        }
        if (pawn.EyeAngles != null)
        {
            pawn.EyeAngles.X = pitch;
            pawn.EyeAngles.Y = yaw;
            pawn.EyeAngles.Z = 0;
            Utilities.SetStateChanged(pawn, "CBasePlayerPawn", "m_angEyeAngles");
        }

        if ((++_humanViewLogCounter) % 64 == 0)
        {
            Logger.LogInformation(
                "[Replay-PostThink] human slot={Slot} forced pitch={P:F2} yaw={Y:F2} | hook hits={Hits} overrides={Ovr} (active={Act})",
                slot, pitch, yaw,
                HumanViewHook.DetourHits, HumanViewHook.DetourOverrides, HumanViewHook.IsActive);
        }
    }

    // ---------------------------- OnTick override ---------------------------- //

    private void OnTick()
    {
        // Round replay — drives all bot positions/fire events from recorded data
        if (_replayActive)
        {
            try
            {
                int relTick = Server.TickCount - _replayStartServerTick;
                if (relTick > _replayMaxRelTick)
                {
                    _replayActive = false;
                    foreach (var slot in _replayActiveAttacks.Keys.ToList())
                        ReleaseBotAttack(slot);
                    _replayActiveAttacks.Clear();
                    Logger.LogInformation("[BotPoseControl] Replay complete @ rel tick {Tick}", relTick);
                }
                else if (relTick != _replayCurrentRelTick)
                {
                    // First live tick — give loadouts now (server state is stable, freeze running)
                    if (!_replayLoadoutGiven)
                    {
                        GiveInitialLoadouts();
                        _replayLoadoutGiven = true;
                    }
                    // Apply state for every tick we passed (handles tickrate jitter)
                    for (int rt = _replayCurrentRelTick + 1; rt <= relTick; rt++)
                        ApplyReplayTick(rt);
                    _replayCurrentRelTick = relTick;
                }
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[BotPoseControl] Replay error");
            }
        }
        // Standard bot pose maintenance (only when not replaying)
        else if (!_activePoses.IsEmpty)
        {
            var deadSlots = new List<int>();
            foreach (var (slot, pose) in _activePoses)
            {
                try { ApplyPose(pose, initialApply: false); }
                catch { deadSlots.Add(slot); }
            }
            foreach (var s in deadSlots) _activePoses.TryRemove(s, out _);
        }

        if (_punchRecording)
        {
            _diagTicks++;
            try
            {
                var viewer = Utilities.GetPlayers().FirstOrDefault(p => p != null && p.IsValid && !p.IsBot);
                if (viewer == null) { _diagNoViewer++; return; }
                var pawn = viewer.PlayerPawn?.Value;
                if (pawn == null || !pawn.IsValid) { _diagNoPawn++; return; }

                // Use AimPunchServices (affects bullet trajectory), not CameraServices
                // (visual shake only).
                // m_pAimPunchServices found at pawn+0xD28 via find_aim_punch_offset.py.
                // m_predictableBaseAngle inside AimPunchServices is at +0x50.
                const int CAM_SERVICES_OFFSET = 0xD28;
                const int PUNCH_OFFSET = 0x50;

                IntPtr camPtrAddr = IntPtr.Add(pawn.Handle, CAM_SERVICES_OFFSET);
                if (!IsReadable(camPtrAddr, IntPtr.Size)) { _diagBadCamPtr++; return; }

                IntPtr camServices = Marshal.ReadIntPtr(camPtrAddr);
                if (camServices == IntPtr.Zero) { _diagNullCam++; return; }

                IntPtr punchAddr = IntPtr.Add(camServices, PUNCH_OFFSET);
                if (!IsReadable(punchAddr, 12)) { _diagBadPunchPtr++; return; }

                float pitch = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(punchAddr));
                float yaw   = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(punchAddr + 4));
                if (float.IsNaN(pitch) || float.IsInfinity(pitch)) { _diagNaN++; return; }

                _punchSamples.Add((Server.TickCount, pitch, yaw));
            }
            catch { /* ignore */ }
        }
    }
}
