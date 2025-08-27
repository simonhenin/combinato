# JN 2017-04-07 adding scaling factor for matfiles
# JN 2020-03-06 Python3 compatibility


from __future__ import division, print_function, absolute_import
import os
from argparse import ArgumentParser, FileType
import tables
from .mp_extract import mp_extract
from .. import NcsFile
import numpy as np


def get_nrecs(filename):
    fid = NcsFile(filename)
    return fid.num_recs

def get_h5size(filename):
    fid = tables.open_file(filename, 'r')
    n = fid.root.data.shape[0]
    fid.close()
    return n

def get_binsize(filename):
    """get number of samples in a bin file"""
    fid = open(filename, 'rb')
    fid.seek(0)
    fid.seek(0, 2)  # Seek to end
    file_size = fid.tell()
    data_size = file_size - 8
    bytes_per_sample = np.dtype(np.int32).itemsize
    n_samples = data_size // bytes_per_sample
    fid.close()
    return n_samples


def main():
    """standard main function"""
    # standard options
    nWorkers = 5
    blocksize = 10000

    parser = ArgumentParser(prog='css-extract',
                            description='spike extraction from .ncs files',
                            epilog='Johannes Niediek (jonied@posteo.de)')
    parser.add_argument('--files', nargs='+',
                        help='.ncs files to be extracted', default=['/Users/seh223/Data/Liulab/microwire/NY941/2024-Aug-16_144340.sdma/2024-Aug-16_144340.csc017.bin'])
    parser.add_argument('--start', type=int,
                        help='start index for extraction')
    parser.add_argument('--stop', type=int,
                        help='stop index for extraction')
    parser.add_argument('--jobs', nargs=1,
                        help='job file contains one filename per row')
    parser.add_argument('--matfile', nargs=1,
                        help='extract data from a matlab file')
    parser.add_argument('--h5', action='store_true', default=False,
                        help='assume that files are h5 files')
    parser.add_argument('--matfile-scale-factor', nargs='?', type=float,
                        help='rescale matfile data by this factor'
                             ' (to obtain microvolts)', default=1)
    parser.add_argument('--destination', nargs=1,
                        help='folder where spikes should be saved')
    parser.add_argument('--refscheme', nargs=1, type=FileType(mode='r'),
                        help='scheme for re-referencing')
    args = parser.parse_args()

    if ((args.files is None) and 
        (args.matfile is None) and 
        (args.jobs is None)):

        parser.print_help()
        print('Supply either files or jobs or matfile.')
        return

    if args.destination is not None:
        destination = args.destination[0]
    else:
        destination = ''

    # special case for a matlab file
    if args.matfile is not None:
        jname = os.path.splitext(os.path.basename(args.matfile[0]))[0]
        jobs = [{'name': jname,
                 'filename': args.matfile[0],
                 'is_matfile': True,
                 'count': 0,
                 'destination': destination,
                 'scale_factor': args.matfile_scale_factor}]
        mp_extract(jobs, 1)
        return


    if args.jobs:
        with open(args.jobs[0], 'r') as f:
            files = [a.strip() for a in f.readlines()]
        f.close()
        print('Read jobs from ' + args.jobs[0])
    else:
        files = args.files

    if args.h5:
        jobs = []
        for f in files:
            size = get_h5size(f)
            starts = list(range(0, size, 32000*5*60))
            stops = starts[1:] + [size]
            name = os.path.splitext(os.path.basename(f))[0]

            for i in range(len(starts)):

                jdict = {'name': name,
                     'filename': f,
                     'start': starts[i],
                     'stop': stops[i],
                     'is_h5file': True,
                     'count': i,
                     'destination': destination}

                jobs.append(jdict)

        mp_extract(jobs, nWorkers)
        return


    if files[0] is None:
        print('Specify files!')
        return

    # construct the jobs
    jobs = []

    references = None
    if args.refscheme:
        import csv
        reader = csv.reader(args.refscheme[0], delimiter=';')
        references = {line[0]: line[1] for line in reader}

    for f in files:
        if args.start:
            start = args.start
        else:
            start = 0

        if f.endswith('.bin'):
            blocksize = 100000
            stop = get_binsize(f)
        else:
            nrecs = get_nrecs(f)
            if args.stop:
                stop = min(args.stop, nrecs)
            else:
                stop = nrecs

        if stop % blocksize > blocksize/2:
            laststart = stop-blocksize
        else:
            laststart = stop

        starts = list(range(start, laststart, blocksize))
        stops = starts[1:] + [stop]
        name = os.path.splitext(os.path.basename(f))[0]
        if references is not None:
            reference = references[f]
            print('{} (re-referenced to {})'.format(f, reference))

        else:
            reference = None
            print(name)

        for i in range(len(starts)):
            jdict = {'name': name,
                     'filename': f,
                     'start': starts[i],
                     'stop': stops[i],
                     'count': i,
                     'destination': destination,
                     'reference': reference}

            jobs.append(jdict)


    mp_extract(jobs, nWorkers)
