"""PhysioBank format reader for loading ECG signals from .hea and .dat files.

Supports the WFDB (WaveForm DataBase) format used by PhysioBank databases.
"""

import numpy as np
from pathlib import Path
import re


def _parse_numeric_prefix(token, cast=float, default=0):
    """Parse leading numeric value from a WFDB header token."""
    if token is None:
        return default

    match = re.match(r"^[-+]?\d+(?:\.\d+)?", str(token))
    if not match:
        return default

    value = float(match.group(0))
    if cast is int:
        return int(round(value))
    return cast(value)


def parse_header(hea_file):
    """Parse a .hea (header) file and return metadata and signal info.
    
    Returns dict with:
    - record_name: name of the record
    - num_signals: number of signals in the record
    - sampling_rate: sampling frequency in Hz
    - num_samples: total number of samples
    - signals: list of signal metadata dicts
    """
    hea_path = Path(hea_file)
    record_name = hea_path.stem
    
    with open(hea_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError(f"Invalid header file {hea_file}: too few lines")
    
    # Parse first line (record info)
    first_line = lines[0].strip().split()
    if len(first_line) < 4:
        raise ValueError(f"Invalid record line in {hea_file}")
    
    num_signals = int(first_line[1])
    sampling_rate = int(first_line[2])
    num_samples = int(first_line[3])
    
    # Parse signal lines
    signals = []
    for i in range(1, min(num_signals + 1, len(lines))):
        parts = lines[i].strip().split()
        if len(parts) < 3:
            continue
        
        signal_info = {
            # WFDB signal line convention:
            # filename format gain adc_res adu_zero init_value checksum block_size description...
            'filename': parts[0],
            'byte_offset': 0,
            'format': _parse_numeric_prefix(parts[1] if len(parts) > 1 else None, cast=int, default=212),
            'units_per_adu': _parse_numeric_prefix(parts[2] if len(parts) > 2 else None, cast=float, default=1.0),
            'adu_zero': _parse_numeric_prefix(parts[4] if len(parts) > 4 else None, cast=float, default=0.0),
            'init_value': _parse_numeric_prefix(parts[5] if len(parts) > 5 else None, cast=float, default=0.0),
            'signal_name': " ".join(parts[8:]).strip() if len(parts) > 8 else f"Signal{i}",
        }
        signals.append(signal_info)
    
    return {
        'record_name': record_name,
        'num_signals': num_signals,
        'sampling_rate': sampling_rate,
        'num_samples': num_samples,
        'signals': signals,
    }


def load_signal_16bit(dat_file, byte_offset, num_samples):
    """Load a 16-bit signed integer signal from a .dat file.
    
    Args:
        dat_file: path to the .dat file
        byte_offset: byte offset where signal starts
        num_samples: number of samples to read
        
    Returns:
        numpy array of int16 samples
    """
    with open(dat_file, 'rb') as f:
        f.seek(byte_offset)
        data = np.frombuffer(f.read(num_samples * 2), dtype=np.int16)
    
    return data


def _decode_format_212(dat_file, byte_offset, num_samples):
    """Decode WFDB format 212 (two 12-bit signed samples per 3 bytes)."""
    n_triplets = (num_samples + 1) // 2
    n_bytes = n_triplets * 3

    with open(dat_file, 'rb') as f:
        f.seek(byte_offset)
        raw = np.frombuffer(f.read(n_bytes), dtype=np.uint8)

    if raw.size < 3:
        return np.array([], dtype=np.int16)

    raw = raw[: (raw.size // 3) * 3]
    b0 = raw[0::3].astype(np.int32)
    b1 = raw[1::3].astype(np.int32)
    b2 = raw[2::3].astype(np.int32)

    s1 = ((b1 & 0x0F) << 8) | b0
    s2 = ((b1 & 0xF0) << 4) | b2

    s1 = np.where(s1 >= 2048, s1 - 4096, s1)
    s2 = np.where(s2 >= 2048, s2 - 4096, s2)

    out = np.empty(s1.size + s2.size, dtype=np.int16)
    out[0::2] = s1.astype(np.int16)
    out[1::2] = s2.astype(np.int16)
    return out[:num_samples]


def load_signal(dat_file, signal_format, byte_offset, num_samples):
    """Load signal data based on WFDB format code.
    
    Common format codes:
    - 16: 16-bit signed integer (alternate encoding)
    - 61: 16-bit signed integer (WFDB alternate)
    - 212: 16-bit signed integer (packed format)
    - 100: 8-bit unsigned integer
    - 400: 16-bit signed integer (used in CU database)
    """
    if signal_format == 212:
        return _decode_format_212(dat_file, byte_offset, num_samples)

    # Formats 16, 61, and 400 are 16-bit signed formats
    if signal_format in (16, 61, 400):
        return load_signal_16bit(dat_file, byte_offset, num_samples)
    else:
        raise NotImplementedError(f"Format {signal_format} not yet supported")


def load_physiobank_record(record_path, signal_index=0):
    """Load a PhysioBank record and return times, ECG signal, and sampling rate.
    
    Args:
        record_path: path to the record file (without extension, e.g., 'cu01')
        signal_index: which signal to load (default 0 = first signal)
        
    Returns:
        tuple of (times, signal, sampling_rate)
    """
    record_path = Path(record_path)
    hea_file = record_path.with_suffix('.hea')
    
    if not hea_file.exists():
        raise FileNotFoundError(f"Header file not found: {hea_file}")
    # Parse header
    header = parse_header(hea_file)
    sampling_rate = header['sampling_rate']
    num_samples = header['num_samples']
    
    if signal_index >= len(header['signals']):
        raise ValueError(f"Signal index {signal_index} out of range (record has {len(header['signals'])} signals)")
    
    signal_info = header['signals'][signal_index]
    dat_file = record_path.parent / signal_info['filename']

    if not dat_file.exists():
        raise FileNotFoundError(f"Data file not found: {dat_file}")
    
    # Load signal data
    raw_samples = load_signal(
        dat_file,
        signal_info['format'],
        signal_info['byte_offset'],
        num_samples
    )
    
    # Convert ADC counts to physical units (usually mV)
    # physical_value = (adc_count - baseline) / units_per_adu
    units_per_adu = signal_info['units_per_adu'] if signal_info['units_per_adu'] else 1.0
    signal = (raw_samples.astype(float) - signal_info['adu_zero']) / units_per_adu
    
    # Generate time vector
    times = np.arange(num_samples) / sampling_rate
    
    return times, signal, sampling_rate


def list_records(database_path):
    """List all available records in a PhysioBank database directory.
    
    Returns list of record paths (without extension).
    """
    db_path = Path(database_path)
    records = set()
    
    for hea_file in db_path.glob('*.hea'):
        record_name = hea_file.stem
        # Skip backup files (those ending with -)
        if not record_name.endswith('-'):
            records.add(record_name)
    
    return sorted(records)
