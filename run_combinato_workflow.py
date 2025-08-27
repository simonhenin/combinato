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

# job_file = '/Users/seh223/combinato_jobs.txt'
# out_folder = '/Users/seh223/combinato_out/'



def main(job_file, out_folder):
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

        sessions = prepare_main([fname], sign, 'index', 0,
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
    parser.add_argument('--jobs', type=FileType('r'))
    parser.add_argument('--destination', nargs=1)
    args = parser.parse_args()

    if args.destination is None:
        print('Supply destination folder using --destination')
        sys.exit(1)
    if args.jobs is None:
        print('Supply job file using --jobs')
        sys.exit(1)

    main(args.jobs.name, args.destination[0])
