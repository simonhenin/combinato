#!/usr/bin/env python3
"""
Neuralynx SDMA Folder Reader

This script reads Neuralynx SDMA (Spike Data Multi-Array) folders and extracts
neural recording data from various file types commonly found in Neuralynx recordings.

Supported file types:
- .ncs (Continuously Sampled Channels)
- .nev (Event files)
- .nse (Spike files)
- .ntt (Tetrode files)
- .nvt (Video tracking files)
- .bin (FieldTrip binary data files)
- .mat/.json (Metadata files for FieldTrip data)
"""

import os
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

class NeuralynxReader:
    """Reader for Neuralynx data files"""
    
    def __init__(self, folder_path: str):
        
        if len(folder_path) > 0:
            self.folder_path = Path(folder_path)
            if not self.folder_path.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
        
            self.files = self._scan_folder()
        
    def _scan_folder(self) -> Dict[str, List[Path]]:
        """Scan folder for Neuralynx files"""
        files = {
            'ncs': [],  # Continuously sampled channels
            'nev': [],  # Events
            'nse': [],  # Single electrode spikes
            'ntt': [],  # Tetrode files
            'nvt': [],  # Video tracking
            'bin': [],  # FieldTrip binary files
            'mat': [],  # MATLAB files (often accompanying .bin)
            'json': [], # JSON metadata files
            'other': []
        }
        
        for file_path in self.folder_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.ncs', '.nev', '.nse', '.ntt', '.nvt']:
                    files[ext[1:]].append(file_path)
                elif ext == '.bin':
                    files['bin'].append(file_path)
                elif ext == '.mat':
                    files['mat'].append(file_path)
                elif ext == '.json':
                    files['json'].append(file_path)
                else:
                    files['other'].append(file_path)
        
        return files
    
    def _read_header(self, file_path: Path) -> Dict[str, Any]:
        """Read Neuralynx file header"""
        header = {}
        
        with open(file_path, 'rb') as f:
            # Read first 16KB as header
            header_bytes = f.read(16384)
            header_str = header_bytes.decode('utf-8', errors='ignore')
            
            # Parse header lines
            lines = header_str.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-'):
                    # Parse parameter lines
                    parts = line.split(None, 1)
                    if len(parts) >= 2:
                        key = parts[0][1:]  # Remove leading '-'
                        value = parts[1]
                        header[key] = value
        
        return header
    
    def read_ncs_file(self, file_path: Path) -> Dict[str, Any]:
        """Read .ncs (Continuously Sampled Channel) file"""
        header = self._read_header(file_path)
        
        # NCS record structure:
        # uint64: timestamp (microseconds)
        # uint32: channel number
        # uint32: sample frequency
        # uint32: number of valid samples
        # int16[512]: data samples
        
        records = []
        timestamps = []
        data_samples = []
        
        with open(file_path, 'rb') as f:
            f.seek(16384)  # Skip header
            
            while True:
                record = f.read(1044)  # 8+4+4+4+1024 bytes per record
                if len(record) < 1044:
                    break
                
                # Unpack record
                timestamp, channel, freq, valid_samples = struct.unpack('<QIII', record[:20])
                samples = struct.unpack('<512h', record[20:])  # 512 int16 samples
                
                timestamps.append(timestamp)
                data_samples.extend(samples[:valid_samples])
        
        return {
            'header': header,
            'timestamps': np.array(timestamps),
            'data': np.array(data_samples),
            'sampling_rate': int(header.get('SamplingFrequency', 32000)),
            'channel': header.get('AcqEntName', 'Unknown')
        }
    
    def read_nev_file(self, file_path: Path) -> Dict[str, Any]:
        """Read .nev (Event) file"""
        header = self._read_header(file_path)
        
        # NEV record structure:
        # int16: packet_id
        # int16: packet_data_size
        # uint64: timestamp
        # int16: event_id
        # int16: ttl_value
        # int16: extras[8]
        # string: event_string
        
        events = []
        
        with open(file_path, 'rb') as f:
            f.seek(16384)  # Skip header
            
            while True:
                record_header = f.read(4)
                if len(record_header) < 4:
                    break
                
                packet_id, packet_size = struct.unpack('<HH', record_header)
                
                if packet_size > 0:
                    record_data = f.read(packet_size - 4)
                    if len(record_data) < packet_size - 4:
                        break
                    
                    if len(record_data) >= 24:  # Minimum for timestamp + event data
                        timestamp, event_id, ttl = struct.unpack('<QHH', record_data[:12])
                        
                        event = {
                            'timestamp': timestamp,
                            'event_id': event_id,
                            'ttl_value': ttl,
                            'packet_id': packet_id
                        }
                        
                        # Try to extract event string if present
                        if len(record_data) > 24:
                            try:
                                event_string = record_data[24:].decode('utf-8', errors='ignore').rstrip('\x00')
                                event['event_string'] = event_string
                            except:
                                pass
                        
                        events.append(event)
        
        return {
            'header': header,
            'events': events
        }
    
    def _find_metadata_file(self, bin_path: Path) -> Optional[Path]:
        """Find associated metadata file for a .bin file"""
        # Common FieldTrip metadata file patterns
        base_name = bin_path.stem.split('.')[0]
        possible_extensions = ['.json', '.mat', '.txt', '.cfg']
        
        for ext in possible_extensions:
            metadata_path = bin_path.parent / f"{base_name}{ext}"
            if metadata_path.exists():
                return metadata_path
        
        # Look for generic metadata files in the same directory
        for pattern in ['info.json', 'header.json', 'config.json', 'metadata.mat']:
            metadata_path = bin_path.parent / pattern
            if metadata_path.exists():
                return metadata_path
        
        return None
    
    def _read_json_metadata(self, json_path: Path) -> Dict[str, Any]:
        """Read JSON metadata file"""
        import json
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read JSON metadata from {json_path}: {e}")
            return {}
    
    def _read_mat_metadata(self, mat_path: Path) -> Dict[str, Any]:
        """Read MATLAB metadata file"""
        try:
            import scipy.io
            mat_data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
            # Remove MATLAB-specific keys
            metadata = {k: v for k, v in mat_data.items() if not k.startswith('__')}
            return metadata
        except ImportError:
            print("Warning: scipy not available for reading .mat files")
            return {}
        except Exception as e:
            print(f"Warning: Could not read MAT metadata from {mat_path}: {e}")
            return {}
    
    def _read_txt_metadata(self, txt_path: Path) -> Dict[str, Any]:
        """Read Neuralynx-style text metadata file"""
        metadata = {}
    
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
            # Split into lines and process each line
            lines = content.split('\n')
        
            for line in lines:
                line = line.strip()
            
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
            
                # Process parameter lines starting with '-'
                if line.startswith('-'):
                    # Remove leading '-' and split on first space
                    line = line[1:].strip()
                
                    # Handle different formats
                    if ' ' in line:
                        parts = line.split(' ', 1)
                        key = parts[0]
                        value_str = parts[1].strip()
                    
                        # Remove quotes if present
                        if value_str.startswith('"') and value_str.endswith('"'):
                            value_str = value_str[1:-1]
                    
                        # Try to convert to appropriate type
                        value = self._parse_metadata_value(value_str)
                        metadata[key] = value
                    else:
                        # Key without value
                        metadata[line] = True
                    
                # Handle other formats (key-value pairs without '-')
                elif ':' in line:
                    key, value_str = line.split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    value = self._parse_metadata_value(value_str)
                    metadata[key] = value
                
                elif '=' in line:
                    key, value_str = line.split('=', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    value = self._parse_metadata_value(value_str)
                    metadata[key] = value
    
        except Exception as e:
            print(f"Warning: Could not read text metadata from {txt_path}: {e}")
            return {}
    
        return metadata

    def _parse_metadata_value(self, value_str: str) -> Any:
        """Parse a metadata value string to appropriate Python type"""
        value_str = value_str.strip()
    
        # Remove quotes
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
    
        # Try to parse as number
        try:
            # Check if it's an integer
            if '.' not in value_str and 'e' not in value_str.lower():
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            pass
    
        # Check for boolean values
        if value_str.lower() in ['true', 'yes', 'on', '1']:
            return True
        elif value_str.lower() in ['false', 'no', 'off', '0']:
            return False
    
        # Return as string
        return value_str

    def read_txt_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Read standalone text metadata file"""
        return {
            'metadata': self._read_txt_metadata(file_path),
            'file_path': str(file_path),
            'file_type': 'neuralynx_header'
        }
        """Read MATLAB metadata file"""
        try:
            import scipy.io
            mat_data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
            # Remove MATLAB-specific keys
            metadata = {k: v for k, v in mat_data.items() if not k.startswith('__')}
            return metadata
        except ImportError:
            print("Warning: scipy not available for reading .mat files")
            return {}
        except Exception as e:
            print(f"Warning: Could not read MAT metadata from {mat_path}: {e}")
            return {}
    
    def read_bin_file(self, file_path: Path, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Read FieldTrip .bin file with associated metadata"""
        
        # Try to find and read metadata
        if metadata is None:
            metadata_path = self._find_metadata_file(file_path)
            if metadata_path:
                if metadata_path.suffix == '.json':
                    metadata = self._read_json_metadata(metadata_path)
                elif metadata_path.suffix == '.mat':
                    metadata = self._read_mat_metadata(metadata_path)
                elif metadata_path.suffix == '.txt':
                    metadata = self._read_txt_metadata(metadata_path)
                else:
                    metadata = {}
            else:
                metadata = {}
        
        
        # Extract key parameters from metadata
        n_channels = metadata.get('nChans', metadata.get('n_channels', 1))
        sampling_rate = metadata.get('sampleRate', metadata.get('fs', metadata.get('sampling_rate', 30000)))
        
        # Map data types to numpy dtypes
        dtype_map = {
            'int16': np.int16,
            'int32': np.int32,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'single': np.float32,
            'double': np.float64,
            'float32': np.float32,
            'float64': np.float64
        }
        
        numpy_dtype = dtype_map.get(data_type, np.int32)
        
        # Read binary data
        try:
            with open(file_path, 'rb') as f:
                dataformat = f.read(8).decode('utf-8'); # data type stored as 8-byte header
                numpy_dtype = dtype_map.get(dataformat, np.int32)
                data = np.frombuffer(f.read(), dtype=numpy_dtype)
            
            # Reshape data if multi-channel
            if n_channels > 1:
                # Data is typically interleaved: [ch1_sample1, ch2_sample1, ..., chN_sample1, ch1_sample2, ...]
                n_samples = len(data) // n_channels
                if len(data) % n_channels == 0:
                    data = data.reshape(n_samples, n_channels).T  # Shape: (n_channels, n_samples)
                else:
                    print(f"Warning: Data length ({len(data)}) not divisible by n_channels ({n_channels})")
            
            # Convert to microvolts if scaling factor is available
            scaling_factor = metadata.get('scalingFactor', metadata.get('scale', 1.0))
            if scaling_factor != 1.0:
                data = data.astype(np.float32) * scaling_factor
            
            return {
                'data': data,
                'metadata': metadata,
                'n_channels': n_channels,
                'sampling_rate': sampling_rate,
                'data_type': data_type,
                'n_samples': data.shape[-1] if n_channels > 1 else len(data),
                'file_path': str(file_path)
            }
            
        except Exception as e:
            print(f"Error reading binary file {file_path}: {e}")
            return {
                'data': None,
                'metadata': metadata,
                'error': str(e),
                'file_path': str(file_path)
            }
    
    def detect_fieldtrip_structure(self) -> Dict[str, Any]:
        """Detect if this is a FieldTrip-generated folder and analyze structure"""
        structure_info = {
            'is_fieldtrip': False,
            'has_bin_files': len(self.files['bin']) > 0,
            'has_metadata': len(self.files['json']) > 0 or len(self.files['mat']) > 0,
            'structure_type': 'unknown'
        }
        
        if structure_info['has_bin_files']:
            structure_info['is_fieldtrip'] = True
            
            # Analyze file naming patterns
            bin_files = [f.stem for f in self.files['bin']]
            
            # Common FieldTrip patterns
            if any('continuous' in name.lower() for name in bin_files):
                structure_info['structure_type'] = 'continuous_data'
            elif any('trial' in name.lower() for name in bin_files):
                structure_info['structure_type'] = 'epoched_data'
            elif len(bin_files) == 1:
                structure_info['structure_type'] = 'single_file'
            else:
                structure_info['structure_type'] = 'multi_file'
        
        return structure_info
        """Read .ntt (Tetrode) file"""
        header = self._read_header(file_path)
        
        # NTT record structure varies, but typically:
        # uint64: timestamp
        # uint32: spike classification
        # int16[32*4]: waveform data (32 samples per 4 channels)
        
        spikes = []
        
        with open(file_path, 'rb') as f:
            f.seek(16384)  # Skip header
            
            while True:
                record = f.read(304)  # Typical NTT record size
                if len(record) < 304:
                    break
                
                try:
                    timestamp, sc_number = struct.unpack('<QL', record[:12])
                    
                    # Read waveforms for 4 channels (32 samples each)
                    waveforms = []
                    for ch in range(4):
                        start_idx = 12 + ch * 64  # 32 samples * 2 bytes
                        channel_data = struct.unpack('<32h', record[start_idx:start_idx + 64])
                        waveforms.append(list(channel_data))
                    
                    spike = {
                        'timestamp': timestamp,
                        'sc_number': sc_number,
                        'waveforms': waveforms  # 4 channels x 32 samples
                    }
                    
                    spikes.append(spike)
                    
                except struct.error:
                    break
        
        return {
            'header': header,
            'spikes': spikes,
            'tetrode': header.get('AcqEntName', 'Unknown')
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of files in the SDMA folder"""
        summary = {
            'folder_path': str(self.folder_path),
            'file_counts': {k: len(v) for k, v in self.files.items()},
            'files': {}
        }
        
        for file_type, file_list in self.files.items():
            if file_type != 'other' and file_list:
                summary['files'][file_type] = [str(f.name) for f in file_list]
        
        return summary
    
    def read_all_data(self) -> Dict[str, Any]:
        """Read all supported files in the folder"""
        data = {
            'ncs': {},
            'nev': {},
            'ntt': {},
            'bin': {},
            'fieldtrip_info': self.detect_fieldtrip_structure(),
            'summary': self.get_summary()
        }
        
        # Read NCS files
        for ncs_file in self.files['ncs']:
            try:
                print(f"Reading NCS file: {ncs_file.name}")
                data['ncs'][ncs_file.stem] = self.read_ncs_file(ncs_file)
            except Exception as e:
                print(f"Error reading {ncs_file.name}: {e}")
        
        # Read NEV files
        for nev_file in self.files['nev']:
            try:
                print(f"Reading NEV file: {nev_file.name}")
                data['nev'][nev_file.stem] = self.read_nev_file(nev_file)
            except Exception as e:
                print(f"Error reading {nev_file.name}: {e}")
        
        # Read NTT files
        for ntt_file in self.files['ntt']:
            try:
                print(f"Reading NTT file: {ntt_file.name}")
                data['ntt'][ntt_file.stem] = self.read_ntt_file(ntt_file)
            except Exception as e:
                print(f"Error reading {ntt_file.name}: {e}")
        
        # Read BIN files (FieldTrip format)
        for bin_file in self.files['bin']:
            try:
                print(f"Reading BIN file: {bin_file.name}")
                data['bin'][bin_file.stem] = self.read_bin_file(bin_file)
            except Exception as e:
                print(f"Error reading {bin_file.name}: {e}")
        
        return data

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Read Neuralynx SDMA folder')
    parser.add_argument('folder_path', help='Path to Neuralynx SDMA folder')
    parser.add_argument('--summary-only', action='store_true', 
                       help='Show only file summary without reading data')
    parser.add_argument('--output', help='Save data to pickle file')
    
    args = parser.parse_args()
    
    try:
        # Initialize reader
        reader = NeuralynxReader(args.folder_path)
        
        # Print summary
        summary = reader.get_summary()
        print("Neuralynx SDMA Folder Summary:")
        print(f"Folder: {summary['folder_path']}")
        print("File counts:")
        for file_type, count in summary['file_counts'].items():
            if count > 0:
                print(f"  {file_type.upper()}: {count} files")
        
        if not args.summary_only:
            # Read all data
            print("\nReading data...")
            data = reader.read_all_data()
            
            # Print basic info about loaded data
            print("\nData loaded:")
            for data_type in ['ncs', 'nev', 'ntt', 'bin']:
                if data[data_type]:
                    print(f"  {data_type.upper()}: {len(data[data_type])} files")
                    for name, file_data in data[data_type].items():
                        if data_type == 'ncs' and 'data' in file_data:
                            print(f"    {name}: {len(file_data['data'])} samples")
                        elif data_type == 'nev' and 'events' in file_data:
                            print(f"    {name}: {len(file_data['events'])} events")
                        elif data_type == 'ntt' and 'spikes' in file_data:
                            print(f"    {name}: {len(file_data['spikes'])} spikes")
                        elif data_type == 'bin' and 'data' in file_data and file_data['data'] is not None:
                            shape = file_data['data'].shape
                            sr = file_data.get('sampling_rate', 'unknown')
                            print(f"    {name}: {shape} samples @ {sr} Hz")
            
            # Print FieldTrip info if detected
            if data['fieldtrip_info']['is_fieldtrip']:
                print(f"\nFieldTrip structure detected:")
                print(f"  Structure type: {data['fieldtrip_info']['structure_type']}")
                print(f"  Binary files: {data['fieldtrip_info']['has_bin_files']}")
                print(f"  Metadata files: {data['fieldtrip_info']['has_metadata']}")
            
            # Save data if requested
            if args.output:
                import pickle
                with open(args.output, 'wb') as f:
                    pickle.dump(data, f)
                print(f"\nData saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())