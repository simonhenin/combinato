#!/usr/bin/env python3
import os
import subprocess

from combinato.cluster.prepare import main as prepare_main
from combinato.cluster.cluster import main as cluster_main
from combinato.cluster.concatenate import main as combine_main
from combinato.cluster.create_groups import main as groups_main

job_file = '/Users/seh223/combinato_jobs.txt'
out_folder = '/Users/seh223/combinato_out/'

print('Using job file: {}'.format(job_file))

if not os.path.exists(os.path.expanduser(out_folder)):
    os.makedirs(os.path.expanduser(out_folder))



# # extract spikes
# cmd = ['/Users/seh223/combinato/css-extract',
#        '--jobs', job_file,
#        '--destination', out_folder]
# subprocess.run(cmd)


# # find concurrent spikes
# os.chdir(out_folder)
# cmd = ['/Users/seh223/combinato/css-find-concurrent']
# subprocess.run(cmd)


# #mask artifacts
# cmd = ['/Users/seh223/combinato/css-mask-artifacts']      
# subprocess.run(cmd)

# find all extracted h5 files
from combinato import h5files
h5files = h5files(out_folder)
print('Found {} h5 files'.format(len(h5files)))




# # create sorting preparation jobs
# job_file = os.path.join(out_folder, 'combinato_neg_jobs.txt')
# with open(job_file, 'w') as f:
#     for fff in h5files:
#         f.write('{}\n'.
#                 format(fff))
# f.close()
# print('Wrote sorting job file: {}'.format(job_file))

# # prepare clustering jobs using job file
# os.chdir(out_folder)
# sorting_job_file = os.path.join(out_folder, 'sort_neg_jobs.txt')
# if os.path.exists(sorting_job_file):
#     os.remove(sorting_job_file)
# cmd = ['/Users/seh223/combinato/css-prepare-sorting',
#        '--jobs', job_file, '--label', 'jobs'] 
# subprocess.run(cmd)

# # run clustering using sorting job file
# cmd = ['/Users/seh223/combinato/css-cluster',
#        '--jobs', sorting_job_file] 
# subprocess.run(cmd)

# # combine clustered files
# cmd = ['/Users/seh223/combinato/css-combine',
#        '--jobs', sorting_job_file, 
#        '--single', True] 
# subprocess.run(cmd)



for fname in h5files:

    print('Processing file: {}'.format(fname))

    sign = 'neg'
    label = 'test'

    sessions = prepare_main([fname], sign, 'index', 0,
                                None, 20000, label, False, False)
    if (sessions) :

        for name, sign, ses in sessions:
            cluster_main(name, ses, sign)

    
        outfname = combine_main(fname,
                                [os.path.basename(ses[2]) for ses in sessions],
                                label)

        groups_main(fname, outfname)

