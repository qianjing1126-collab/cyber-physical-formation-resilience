# ns-3 installation status

Status: **passed on 2026-08-06**.

- Host: Windows with WSL 2.7.11.0
- Distribution: Ubuntu 24.04.4 LTS, WSL2
- Linux user: `tnsmsim` (non-root build and execution)
- ns-3: 3.47
- Source directory: `/opt/tnsm-ns3/ns-3.47`
- Official archive SHA-1: `897c3f2db2c6de6a8291e46b3020dbeb1af75269`
- Build system: CMake 3.28.3 and Ninja 1.11.1
- Toolchain isolation: Windows PATH injection disabled in `/etc/wsl.conf`
- 802.11p smoke result: `G3_80211P_SMOKE_OK nodes=2 devices=2`

The project-side smoke source is `ns3/g3_80211p_smoke.cc`. It deliberately uses
the ns-3.47 Wi-Fi API (`WIFI_STANDARD_80211p`) rather than the retired
standalone WAVE helper API.

Re-run from PowerShell:

```powershell
wsl -d Ubuntu-24.04 -u tnsmsim -- bash -lc "cd /opt/tnsm-ns3/ns-3.47 && ./ns3 run scratch/tnsm-80211p-smoke"
```

Expected terminal marker:

```text
G3_80211P_SMOKE_OK nodes=2 devices=2
```

This status certifies the simulator installation only. It does not certify the
Python communication adapter or any paper result.
