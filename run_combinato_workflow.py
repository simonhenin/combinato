#!/usr/bin/env python3
import os
import sys
import subprocess
import time
from argparse import ArgumentParser, FileType

from combinato.extract.extract import main as extract_main
from combinato.artifacts.concurrent import main as concurrent_main
from combinato.artifacts.mask_artifacts import parse_args as mask_main
from combinato.cluster.prepare import main as prepare_main
from combinato.cluster.cluster import main as cluster_main
from combinato.cluster.concatenate import main as combine_main
from combinato.cluster.create_groups import main as groups_main

# Usage:
#   python run_combinato_workflow.py --jobs jobs.txt --destination /path/to/output/folder
#
# Multiple job files, same destination:
#   python run_combinato_workflow.py --jobs job1.txt job2.txt job3.txt --destination /out
#
# Multiple job files, different destinations:
#   python run_combinato_workflow.py --jobs job1.txt job2.txt job3.txt --destination /out1 /out2 /out3
#
# Process only one polarity (default: both pos and neg):
#   python run_combinato_workflow.py --jobs jobs.txt --destination /out --sign neg
#   python run_combinato_workflow.py --jobs jobs.txt --destination /out --sign pos
#
# example job1.txt:
#   /path/to/electrode.bin


def main(job_file, out_folder, signs=('pos', 'neg')):
    print('Using job file: {}'.format(job_file))

    if not os.path.exists(os.path.expanduser(out_folder)):
        os.makedirs(os.path.expanduser(out_folder))


    # extract spikes
    sys.argv = ['~',
        '--jobs', job_file,
        '--destination', out_folder]
    extract_main()

    # find concurrent spikes
    os.chdir(out_folder)
    concurrent_main()

    # mask artifacts using concurrent spikes
    os.chdir(out_folder)
    sys.argv = ['~'] # no arguments to pass here
    mask_main()

    # find all extracted h5 files
    from combinato import h5files
    h5files = h5files(out_folder)
    print('Found {} h5 files'.format(len(h5files)))


    # run through each extracted spike file, and do sorting
    for fname in h5files:

        print('Processing file: {}'.format(fname))

        sign = 'neg'
        label_ = 'basic'
        for sign_ in signs:
            sessions = prepare_main([fname], sign_, 'index', 0,
                                        None, 20000, label_, False, False)
            if (sessions) :
                for name, sign, ses in sessions:
                    cluster_main(name, ses, sign)
                label = 'sort_{}_{}'.format(sign, label_)
                outfname = combine_main(fname,
                                        [os.path.basename(ses[2]) for ses in sessions],
                                        label)
                groups_main(fname, outfname)


if __name__ == "__main__":

    parser = ArgumentParser('run_combinato_workflow.py',
                            description='runs combinato workflow using jobs')
    parser.add_argument('--jobs', type=FileType('r'), nargs='+')
    parser.add_argument('--destination', nargs='+')
    parser.add_argument('--sign', choices=['pos', 'neg'], nargs='+', default=['pos', 'neg'])
    args = parser.parse_args()

    if args.destination is None:
        print('Supply destination folder using --destination')
        sys.exit(1)
    if args.jobs is None:
        print('Supply job file using --jobs')
        sys.exit(1)

    job_files = [f.name for f in args.jobs]
    destinations = args.destination

    if len(destinations) == 1:
        destinations = destinations * len(job_files)
    elif len(destinations) != len(job_files):
        print('Number of destinations must be 1 or match the number of job files')
        sys.exit(1)

    for job_file, dest in zip(job_files, destinations):
        main(job_file, dest, args.sign)
