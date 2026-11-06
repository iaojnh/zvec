# Android and iOS memory budgets

The configured `memory_limit_bytes` is a process RSS soft target shared by
zvec's vector storage, workspaces and RocksDB. Include the host application's
memory when choosing this value. It is not a private allowance on top of the
host's memory, and it does not prevent the OS from terminating the application.

Android and iOS default to the smaller of 128 MiB and one eighth of detected
system/container memory. A host can override this before initialization. The
mobile configuration minimum is 16 MiB; initialization can still fail when the
selected budget cannot cover fixed library and worker costs. Desktop defaults
and their 100 MiB configuration minimum remain unchanged.

On iOS devices (iOS 13 or later), admission also considers `phys_footprint` and
`os_proc_available_memory()`, with a reserve below the application limit. RSS
and footprint are separate constraints, not summed charges. Untouched backing
remains charged across samples. A failed sample retains the previous estimate;
zero available bytes stops new backing. Simulator runs use RSS and host
notifications because host limits do not describe a device application's limit.

## Forward host lifecycle and pressure events

After initialization, C/JNI/Swift hosts can call:

```c
zvec_set_memory_pressure(ZVEC_MEMORY_PRESSURE_BACKGROUND);
/* Once normal cache growth is appropriate again: */
zvec_set_memory_pressure(ZVEC_MEMORY_PRESSURE_NORMAL);
```

C++ hosts use `GlobalConfig::Instance().set_memory_pressure()` with
`GlobalConfig::MemoryPressure::{kNormal,kBackground,kCritical}`.

| Notification | Effect |
| --- | --- |
| NORMAL | Clears host-imposed pressure; RSS and iOS native limits still apply. |
| BACKGROUND | Stops optional page-cache growth and retires unpinned cached/idle backing. Mandatory work may use the remaining budget. |
| CRITICAL | Also refuses new managed backing until the host sends another level. Existing pinned data stays valid. |

Calls enqueue work on the existing maintenance thread; they do not wait for
cache reclamation or perform application callbacks. Notifications do not alter
the configured budget. The most recent notification remains effective until
the host changes it, including across RSS samples.

For Android, forward UI-hidden/background lifecycle and applicable
`onTrimMemory` events as BACKGROUND, and send NORMAL on foreground recovery.
Android 14 and later do not deliver the legacy low-memory trim levels; do not
depend on those notifications. CRITICAL is available for host-detected severe
pressure, with explicit recovery required.

For iOS, forward background transitions and memory warnings as BACKGROUND.
Send NORMAL when normal caching is appropriate again, such as a foreground
transition after pressure subsides. Use CRITICAL only when the host can also
signal recovery. Native iOS headroom sampling remains active independently.

The library does not register Android Activity or UIKit observers itself.
Applications own these callbacks and must forward them to the same library
instance used for their collections. Embed one zvec runtime per process;
independent static copies in multiple native modules do not share a budget.

See [Android memory management](https://developer.android.com/topic/performance/memory/manage-app-memory)
and [Apple application headroom](https://developer.apple.com/documentation/os/os_proc_available_memory).
