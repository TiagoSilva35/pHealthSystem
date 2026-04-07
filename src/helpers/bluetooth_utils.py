import bluetooth


def discover_nearby_devices(scan_duration):
    try:
        raw = bluetooth.discover_devices(duration=scan_duration, lookup_names=True)
    except Exception as exc:
        print(f"[WARN] Failed to scan nearby Bluetooth devices: {exc}")
        return []

    normalized = []
    for item in raw:
        if isinstance(item, tuple) and len(item) >= 2:
            addr, name = item[0], item[1]
        else:
            addr, name = item, "Unknown"
        normalized.append((str(addr), str(name)))

    normalized.sort(key=lambda device: 0 if "bitalino" in device[1].lower() else 1)
    return normalized


def choose_device(discovered_devices, fallback_mac):
    if not discovered_devices:
        if fallback_mac:
            print(f"[INFO] No devices discovered. Falling back to configured MAC: {fallback_mac}")
            return fallback_mac
        raise RuntimeError("No Bluetooth devices discovered and no fallback MAC configured.")

    print("\nNearby Bluetooth devices:")
    for idx, (addr, name) in enumerate(discovered_devices, start=1):
        marker = " [BITalino?]" if "bitalino" in name.lower() else ""
        print(f"  {idx}. {name} ({addr}){marker}")

    while True:
        selection = input("Choose device number (Enter = first one): ").strip()
        if not selection:
            return discovered_devices[0][0]
        if not selection.isdigit():
            print("Invalid input. Please enter a valid number.")
            continue
        index = int(selection)
        if 1 <= index <= len(discovered_devices):
            return discovered_devices[index - 1][0]
        print(f"Invalid index. Please choose a value between 1 and {len(discovered_devices)}.")