using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using Reloaded.Hooks;
using Reloaded.Hooks.Definitions;
using Reloaded.Hooks.Definitions.X64;

namespace BotPoseControl;

/// <summary>
/// Native VFunc detour on <c>CBasePlayerPawn::GetEyeAngles</c> in <c>server.dll</c>.
///
/// Why: server-side writes to <c>v_angle</c> / <c>m_angEyeAngles</c> never reach the
/// local listen-server host's renderer — the renderer queries <c>GetEyeAngles</c> per
/// frame and the local client owns its own view state. Detouring the getter lets us
/// return our forced angle directly to the renderer for the targeted pawn, bypassing
/// every replication / prediction layer.
///
/// Signature courtesy of CS2Fixes' gamedata (Source2ZE/CS2Fixes), known-good for
/// recent CS2 builds. Calling convention is Microsoft x64:
///     <c>QAngle* GetEyeAngles(CBasePlayerPawn* pPawn, QAngle* pRet)</c>
/// where the caller passes a hidden output pointer in RDX and the function fills it
/// with three floats (pitch, yaw, roll) and returns the same pointer in RAX.
/// </summary>
public static class HumanViewHook
{
    // GetEyeAngles signature, with 0xFF marking wildcard bytes (originally \x2A in CS2Fixes).
    private static readonly byte[] PATTERN = {
        0x48, 0x89, 0x5C, 0x24, 0x00, 0x57, 0x48, 0x81, 0xEC, 0x00, 0x00, 0x00, 0x00,
        0x48, 0x8B, 0xF9, 0x48, 0x8B, 0xDA, 0x48, 0x8B, 0x89, 0x00, 0x00, 0x00, 0x00,
        0x48, 0x85, 0xC9,
    };
    private static readonly bool[] WILDCARD = {
        false, false, false, false,  true, false, false, false, false,  true,  true,  true,  true,
        false, false, false, false, false, false, false, false, false,  true,  true,  true,  true,
        false, false, false,
    };

    // Hot state — written by OnPostEntityThink, read by the detour from the render thread.
    // Scalar ref-type assignments on x64 are atomic (CLR memory model), so torn reads of
    // the IntPtr or floats can't happen; a stale value by one tick is fine for our use case.
    public static volatile bool   Enabled    = false;
    public static          IntPtr LockedPawn = IntPtr.Zero;
    public static          float  LockedPitch;
    public static          float  LockedYaw;

    // Telemetry (read from outside)
    public static long DetourHits     = 0;
    public static long DetourOverrides = 0;

    [Function(CallingConventions.Microsoft)]
    public delegate IntPtr GetEyeAnglesFn(IntPtr pPawn, IntPtr pRet);

    private static IHook<GetEyeAnglesFn>? _hook;
    private static GetEyeAnglesFn?        _detourKeepAlive;   // prevent GC of delegate

    public static IntPtr HookedAddress { get; private set; } = IntPtr.Zero;
    public static bool   IsActive      => _hook != null && _hook.IsHookActivated;

    // Secondary hook on the client-side C_BasePlayerPawn::GetEyeAngles. The local
    // listen-server renderer reads the LOCAL player pawn (a client-side mirror
    // entity) for its camera angle, NOT the server pawn. We hook both.
    private static IHook<GetEyeAnglesFn>? _hookClient;
    private static GetEyeAnglesFn?        _detourKeepAliveClient;
    public  static IntPtr HookedAddressClient { get; private set; } = IntPtr.Zero;

    private static unsafe IntPtr DetourClient(IntPtr pPawn, IntPtr pRet)
    {
        // Same logic as the server-side detour — the only difference is which
        // module's function we're intercepting. We can't compare pPawn against
        // the server pawn pointer (different entity instance on client), so we
        // unconditionally override while a lock is active.
        System.Threading.Interlocked.Increment(ref DetourHits);
        if (Enabled && pRet != IntPtr.Zero)
        {
            float* pf = (float*)pRet;
            pf[0] = LockedPitch;
            pf[1] = LockedYaw;
            pf[2] = 0f;
            System.Threading.Interlocked.Increment(ref DetourOverrides);
            return pRet;
        }
        return _hookClient!.OriginalFunction(pPawn, pRet);
    }

    private static unsafe IntPtr Detour(IntPtr pPawn, IntPtr pRet)
    {
        System.Threading.Interlocked.Increment(ref DetourHits);
        if (Enabled && pPawn == LockedPawn && pRet != IntPtr.Zero)
        {
            float* pf = (float*)pRet;
            pf[0] = LockedPitch;
            pf[1] = LockedYaw;
            pf[2] = 0f;
            System.Threading.Interlocked.Increment(ref DetourOverrides);
            return pRet;
        }
        return _hook!.OriginalFunction(pPawn, pRet);
    }

    public static bool Install(ILogger logger)
    {
        if (_hook != null)
        {
            logger.LogWarning("[HumanViewHook] Already installed @ 0x{Addr:X}", HookedAddress.ToInt64());
            return true;
        }

        IntPtr addr = SignatureScanner.Find("server.dll", PATTERN, WILDCARD, logger);
        if (addr == IntPtr.Zero)
        {
            logger.LogError("[HumanViewHook] GetEyeAngles signature NOT FOUND in server.dll — pitch lock unavailable");
            SignatureScanner.DumpModulesPublic(logger);
            return false;
        }

        try
        {
            var hooks = ReloadedHooks.Instance;
            _detourKeepAlive = Detour;
            _hook = hooks.CreateHook<GetEyeAnglesFn>(_detourKeepAlive, addr.ToInt64());
            _hook.Activate();
            HookedAddress = addr;
            logger.LogInformation("[HumanViewHook] Installed server-side @ 0x{Addr:X}", addr.ToInt64());
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "[HumanViewHook] Failed to install server hook");
            return false;
        }

        // Now try the client-side mirror in client.dll — what the local listen-server
        // renderer actually reads for the camera. Same byte signature works because
        // C_BasePlayerPawn shares its base class with CBasePlayerPawn in Source 2.
        IntPtr clientAddr = SignatureScanner.Find("client.dll", PATTERN, WILDCARD, logger);
        if (clientAddr != IntPtr.Zero)
        {
            try
            {
                var hooks = ReloadedHooks.Instance;
                _detourKeepAliveClient = DetourClient;
                _hookClient = hooks.CreateHook<GetEyeAnglesFn>(_detourKeepAliveClient, clientAddr.ToInt64());
                _hookClient.Activate();
                HookedAddressClient = clientAddr;
                logger.LogInformation("[HumanViewHook] Installed CLIENT-side @ 0x{Addr:X} (this is what the local renderer reads)", clientAddr.ToInt64());
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "[HumanViewHook] Failed to install client hook (server hook still active)");
            }
        }
        else
        {
            logger.LogWarning("[HumanViewHook] Pattern not found in client.dll — server-only hook (won't fix host renderer)");
        }
        return true;
    }

    public static void Uninstall(ILogger logger)
    {
        try
        {
            _hook?.Disable();
            _hookClient?.Disable();
            _hook = null;
            _hookClient = null;
            _detourKeepAlive = null;
            _detourKeepAliveClient = null;
            Enabled = false;
            logger.LogInformation("[HumanViewHook] Uninstalled");
        }
        catch (Exception ex)
        {
            logger.LogWarning("[HumanViewHook] Uninstall failed: {Err}", ex.Message);
        }
    }

    public static void Update(IntPtr pawn, float pitch, float yaw)
    {
        LockedPitch = pitch;
        LockedYaw   = yaw;
        LockedPawn  = pawn;
        Enabled     = pawn != IntPtr.Zero;
    }

    public static void Disable()
    {
        Enabled    = false;
        LockedPawn = IntPtr.Zero;
    }
}

/// <summary>Byte-pattern scanner over a loaded module's executable pages.</summary>
internal static class SignatureScanner
{
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

    private const uint MEM_COMMIT = 0x1000;
    private const uint PAGE_EXECUTE          = 0x10;
    private const uint PAGE_EXECUTE_READ     = 0x20;
    private const uint PAGE_EXECUTE_READWRITE = 0x40;
    private const uint PAGE_EXECUTE_WRITECOPY = 0x80;
    private const uint EXECUTABLE_MASK = PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;

    public static IntPtr Find(string moduleName, byte[] pattern, bool[] wildcard, ILogger logger)
    {
        // Multiple modules with the same name can be loaded (CSSharp's stub loader
        // ships its own "server.dll", different from the actual CS2 game server.dll
        // at csgo\bin\win64\server.dll). Pick the LARGEST one — the game server is
        // tens of MB while loader stubs are <1 MB.
        var candidates = Process.GetCurrentProcess().Modules
            .Cast<ProcessModule>()
            .Where(m => string.Equals(m.ModuleName, moduleName, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(m => m.ModuleMemorySize)
            .ToList();

        if (candidates.Count == 0)
        {
            logger.LogError("[SignatureScanner] No module named {Mod} loaded", moduleName);
            DumpAllModules(logger, moduleName);
            return IntPtr.Zero;
        }

        foreach (var c in candidates)
            logger.LogInformation("[SignatureScanner] candidate {Mod}: base=0x{Base:X} size=0x{Size:X} path={Path}",
                                  c.ModuleName, c.BaseAddress.ToInt64(), c.ModuleMemorySize, c.FileName);

        var module = candidates.First();
        IntPtr modBase = module.BaseAddress;
        long   modSize = module.ModuleMemorySize;
        long   modEnd  = modBase.ToInt64() + modSize;
        logger.LogInformation("[SignatureScanner] picked {Mod} base=0x{Base:X} size=0x{Size:X}",
                              module.ModuleName, modBase.ToInt64(), modSize);

        // Walk virtual memory regions and only scan executable pages.
        IntPtr cursor = modBase;
        while (cursor.ToInt64() < modEnd)
        {
            var mbi = new MEMORY_BASIC_INFORMATION();
            UIntPtr ret = VirtualQuery(cursor, ref mbi, (UIntPtr)Marshal.SizeOf<MEMORY_BASIC_INFORMATION>());
            if (ret == UIntPtr.Zero) break;

            long regionSize = mbi.RegionSize.ToInt64();
            if (mbi.State == MEM_COMMIT && (mbi.Protect & EXECUTABLE_MASK) != 0)
            {
                IntPtr hit = ScanRange(mbi.BaseAddress, regionSize, pattern, wildcard);
                if (hit != IntPtr.Zero)
                {
                    logger.LogInformation("[SignatureScanner] Found pattern @ 0x{Hit:X} (region 0x{Base:X} prot=0x{Prot:X})",
                                          hit.ToInt64(), mbi.BaseAddress.ToInt64(), mbi.Protect);
                    return hit;
                }
            }
            long next = mbi.BaseAddress.ToInt64() + regionSize;
            if (next <= cursor.ToInt64()) break;
            cursor = (IntPtr)next;
        }
        return IntPtr.Zero;
    }

    public static void DumpModulesPublic(ILogger logger) => DumpAllModules(logger, "(scanner failed)");

    private static void DumpAllModules(ILogger logger, string searchedFor)
    {
        logger.LogInformation("[SignatureScanner] === all loaded modules (searching for {Q}) ===", searchedFor);
        foreach (var m in Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
                                  .OrderByDescending(m => m.ModuleMemorySize)
                                  .Take(40))
        {
            logger.LogInformation("[SignatureScanner]   {Name,-40} size={Size,10} path={Path}",
                                  m.ModuleName, m.ModuleMemorySize, m.FileName);
        }
    }

    private static unsafe IntPtr ScanRange(IntPtr regionBase, long regionSize, byte[] pattern, bool[] wildcard)
    {
        byte* p = (byte*)regionBase;
        long  end = regionSize - pattern.Length;
        for (long i = 0; i <= end; i++)
        {
            bool ok = true;
            for (int j = 0; j < pattern.Length; j++)
            {
                if (wildcard[j]) continue;
                if (p[i + j] != pattern[j]) { ok = false; break; }
            }
            if (ok) return (IntPtr)(p + i);
        }
        return IntPtr.Zero;
    }
}
