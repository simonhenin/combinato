# -*- coding: utf-8 -*-
"""
Basic i/o definitions for Neuralynx files
"""

from __future__ import print_function, division, absolute_import
from os import stat
from datetime import datetime
import re
import numpy as np
# pylint: disable=E1101

NCS_SAMPLES_PER_REC = 512
NLX_OFFSET = 16 * 1024
NCS_RECSIZE = 1044

# Time Pattern tries to deal with the messed up
# time representation in Neuralynx ncs file headers
TIME_PATTERN = re.compile(r'(\d{1,2}:\d{1,2}:\d{1,2}).(\d{1,3})')

nev_type = np.dtype([('', 'V6'),
                     ('timestamp', 'u8'),
                     ('id', 'i2'),
                     ('nttl', 'i2'),
                     ('', 'V38'),
                     ('ev_string', 'S128')])

ncs_type = np.dtype([('timestamp', 'u8'),
                     ('info', ('i4', 3)),
                     ('data', ('i2', 512))])


def time_upsample(time, timestep):
    """
    fills in NCS_SAMPLES_PER_REC timestamps with
    dist timestep after each timestamp given
    """
    filler = NCS_SAMPLES_PER_REC
    timestep *= 1e6
    base = np.linspace(0, timestep*(filler - 1), filler)
    return np.array([base + x for x in time]).ravel()


def nev_read(filename):
    """
    Neuralynx .nev file reading function.
    Returns an array of timestamps and nttls.
    """
    eventmap = np.memmap(filename, dtype=nev_type, mode='r', offset=NLX_OFFSET)
    return np.array([eventmap['timestamp'], eventmap['nttl']]).T


def nev_string_read(filename):
    """
    reading function for string events
    """
    eventmap = np.memmap(filename, dtype=nev_type, mode='r', offset=NLX_OFFSET)
    return np.array([eventmap['timestamp'], eventmap['ev_string']]).T


class NcsFile(object):
    """
    represents ncs files, allows to read data and time
    """
    def __init__(self, filename):
        self.file = None
        self.filename = filename
        self.num_recs = ncs_num_recs(filename)
        self.header = ncs_info(filename)
        self.file = open(filename, 'rb')
        if self.num_recs > 0:
            timestamp = self.read(0, 2, 'timestamp')
            self.timestep = float((timestamp[1] - timestamp[0]))
            self.timestep /= NCS_SAMPLES_PER_REC * 1e6
        else:
            self.timestep = None

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def read(self, start=0, stop=None, mode='data'):
        """
        read data, timestamps, or info fields from ncs file
        """
        if stop > start:
            length = stop - start
        else:
            length = 1
        if start + length > self.num_recs + 1:
            raise IOError("Request to read beyond EOF,"
                          "filename %s, start %i, stop %i" %
                          (self.filename, start, stop))
        else:
            self.file.seek(NLX_OFFSET + start * NCS_RECSIZE)
            data = self.file.read(length * NCS_RECSIZE)
            array_length = int(len(data) / NCS_RECSIZE)
            array_data = np.ndarray(array_length, ncs_type, data)
            if mode == 'both':
                return (array_data['data'].flatten(),
                        array_data['timestamp'].flatten())
            elif mode in ('data', 'timestamp', 'info'):
                return array_data[mode].flatten()


class BinFile(object):
    """
    represents neuralynx bin files, allows to read data and time
    """
    def __init__(self, filename):
        self.file = None
        self.filename = filename
        
        # Initialize reader and load metadata
        from pathlib import Path
        bin_path = Path(self.filename)

        
        metadata = bin_info(self.filename)


        self.metadata = metadata
           
        # Extract key parameters
        self.sampling_rate = self.metadata.get('SamplingFrequency', 
                                               self.metadata.get('fs', 30000))
        self.timestep = 1.0 / self.sampling_rate
        self.n_channels = self.metadata.get('nChans', 1)
        
        # Data type mapping
        dtype_map = {
            'int16': np.int16, 'int32': np.int32,
            'uint16': np.uint16, 'uint32': np.uint32,
            'single': np.float32, 'double': np.float64,
            'float32': np.float32, 'float64': np.float64
        }
        
        # Read data type from file header or metadata
        self.data_type = self.metadata.get('dataType', 'int16')
        self.numpy_dtype = dtype_map.get(self.data_type, np.int16)
        self.bin_file = open(self.filename, 'rb')

        # Open file and read header to get data type
        try:
            self.bin_file.seek(0)
            header_bytes = self.bin_file.read(8)
            if len(header_bytes) == 8:
                dataformat = header_bytes.decode('utf-8').rstrip('\x00')
                if dataformat in dtype_map:
                    self.numpy_dtype = dtype_map[dataformat]
                    self.data_offset = 8
                else:
                    self.data_offset = 0
            else:
                self.data_offset = 0
            
            # Get file size to calculate total samples
            self.bin_file.seek(0)
            self.bin_file.seek(0, 2)  # Seek to end
            file_size = self.bin_file.tell()
            data_size = file_size - self.data_offset
            bytes_per_sample = np.dtype(self.numpy_dtype).itemsize
            total_values = data_size // bytes_per_sample
            
            if self.n_channels > 1:
                self.total_samples = total_values // self.n_channels
            else:
                self.total_samples = total_values
                
        except Exception as e:
            print(f"Warning: Could not read file header: {e}")
            self.data_offset = 0
            # Fallback: calculate from file size
            file_size = os.path.getsize(fname)
            bytes_per_sample = np.dtype(self.numpy_dtype).itemsize
            total_values = file_size // bytes_per_sample
            if self.n_channels > 1:
                self.total_samples = total_values // self.n_channels
            else:
                self.total_samples = total_values
        
        # Set up header info compatible with NcsFile format
        scaling_factor = self.metadata.get('ADBitVolts', 
                                         self.metadata.get('scale', 1.0))
        self.header = {
            'ADBitVolts': scaling_factor,  # Convert to ADBitVolts format
            'SamplingFrequency': self.sampling_rate,
            'AcqEntName': f'bin_channel_0'
        }

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def read(self, start=0, stop=None):
        """
        read data, timestamps, or info fields from ncs file
        """
        if stop <= start:
            stop = start + SAMPLES_PER_REC
        
        # Ensure we don't read beyond file bounds
        start_sample = max(0, start)
        stop_sample = min(stop, self.total_samples)
        n_samples = stop_sample - start_sample
        
        if n_samples <= 0:
            return (np.array([], dtype=np.float32), np.array([]), self.timestep)
        
        # Read data from file 
        self.bin_file.seek(self.data_offset + start_sample * self.n_channels * np.dtype(self.numpy_dtype).itemsize)
        
        # Read the required number of values
        n_values_to_read = n_samples * self.n_channels
        data_bytes = self.bin_file.read(n_values_to_read * np.dtype(self.numpy_dtype).itemsize)
        
        if len(data_bytes) < n_values_to_read * np.dtype(self.numpy_dtype).itemsize:
            # Adjust for actual data read
            n_values_actual = len(data_bytes) // np.dtype(self.numpy_dtype).itemsize
            n_samples = n_values_actual // self.n_channels
        
        # Convert to numpy array
        raw_data = np.frombuffer(data_bytes, dtype=self.numpy_dtype)
        
        # Reshape if multi-channel (take first channel for compatibility)
        if self.n_channels > 1 and len(raw_data) >= self.n_channels:
            # Data is interleaved, take every n_channels-th sample starting from 0
            data = raw_data[0::self.n_channels][:n_samples]
        else:
            data = raw_data[:n_samples]
        
        # Convert to float32 and apply scaling
        fdata = data.astype(np.float32)
        fdata *= (1e6 * self.header['ADBitVolts']) # convert to microvolts
        
        # # Handle reference subtraction
        # if self.ref_file is not None:
        #     # Convert sample indices to record indices for ref file
        #     start_rec = start // SAMPLES_PER_REC
        #     stop_rec = (stop + SAMPLES_PER_REC - 1) // SAMPLES_PER_REC
            
        #     print('Reading reference data from {}'.format(self.ref_file.filename))
        #     ref_data = self.ref_file.read(start_rec, stop_rec, 'data')
        #     fref_data = np.array(ref_data).astype(np.float32)
        #     fref_data *= 1e6 * self.ref_file.header['ADBitVolts']
            
        #     # Trim ref data to match our data length
        #     min_len = min(len(fdata), len(fref_data))
        #     fdata[:min_len] -= fref_data[:min_len]
        
        # Generate timestamps (in milliseconds)
        start_time_us = start_sample * self.timestep * 1e6
        sample_times_us = start_time_us + np.arange(len(fdata)) * self.timestep * 1e6
        atimes = sample_times_us / 1e3  # Convert to milliseconds

        return fdata, atimes
            

def ncs_info(filename):
    """
    Neuralynx .ncs file header extraction function.

    Returns a dictionary of header fields and values.
    """
    d = dict()

    with open(filename, 'rb') as f:
        header = f.read(NLX_OFFSET)
    f.close()
    for line in header.splitlines():
        try:
            field = [fil.decode() for fil in line.split()]
        except UnicodeDecodeError:
            continue

        if len(field) == 2:
            try:
                field[1] = int(field[1])
            except ValueError:
                try:
                    field[1] = float(field[1])
                except ValueError:
                    pass
            d[field[0][1:]] = field[1]

        #Dealing with Pegasus Opened/Closed date&time strings
        elif len(field) == 3:
            if field[0] in ('-TimeCreated', '-TimeClosed'):
                pddate = datetime.strptime(field[1] + ' ' + field[2], '%Y/%m/%d %H:%M:%S')
                dt = datetime(pddate.year, pddate.month, pddate.day, pddate.hour, \
                                   pddate.minute, pddate.second)
                d[field[0][1:]] = dt
        
        #Dealing with Cheetah Opened/Closed date&time strings
        elif len(field) == 7:
            if field[0] == '##':
                if field[2] in ('Opened', 'Closed'):
                    timeg = TIME_PATTERN.match(field[6]).groups()
                    pdt = datetime.strptime(
                        field[4] + ' ' + timeg[0],
                        '%m/%d/%Y %H:%M:%S')
                    dt = datetime(pdt.year,
                                  pdt.month,
                                  pdt.day,
                                  pdt.hour,
                                  pdt.minute,
                                  pdt.second,
                                  int(timeg[1])*1000)
                d[field[2].lower()] = dt
    if 'AcqEntName' not in d:
        d[u'AcqEntName'] = 'channel' + str(d['ADChannel'])
    return d


def ncs_num_recs(filename):
    """
    Calculates theoretical number of records in a .ncs file
    """
    data_size = stat(filename).st_size - NLX_OFFSET
    if data_size % NCS_RECSIZE:
        raise Exception("%s has the wrong size" % filename)
    else:
        return int(data_size / NCS_RECSIZE)



def bin_info(filename):
    """
    Neuralynx .bin file header extraction function.

    Returns a dictionary of header fields and values.
    """
    # Initialize reader and load metadata
    from pathlib import Path
    bin_path = Path(filename)
    
    # Find and read metadata file
    base_name = bin_path.stem.split('.')[0]
    possible_extensions = ['.json', '.mat', '.txt', '.cfg']
    for ext in possible_extensions:
        metadata_path = bin_path.parent / f"{base_name}{ext}"
        if metadata_path.exists():
            break

    
    metadata = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8', errors='ignore') as f:
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
                    value = _parse_metadata_value(value_str)
                    metadata[key] = value
                else:
                    # Key without value
                    metadata[line] = True
                
            # Handle other formats (key-value pairs without '-')
            elif ':' in line:
                key, value_str = line.split(':', 1)
                key = key.strip()
                value_str = value_str.strip()
                value = _parse_metadata_value(value_str)
                metadata[key] = value
            
            elif '=' in line:
                key, value_str = line.split('=', 1)
                key = key.strip()
                value_str = value_str.strip()
                value = _parse_metadata_value(value_str)
                metadata[key] = value

    except Exception as e:
        print(f"Warning: Could not read text metadata from {metadata_path}: {e}")

    return metadata

def _parse_metadata_value(value_str):
    """
    Helper function to parse metadata values from strings.
    Attempts to convert to int, float, or leaves as string.
    """
    # Remove quotes if present
    if value_str.startswith('"') and value_str.endswith('"'):
        value_str = value_str[1:-1]
    
    # Try to convert to appropriate type
    try:
        value = int(value_str)
    except ValueError:
        try:
            value = float(value_str)
        except ValueError:
            value = value_str  # Leave as string if conversion fails
    return value