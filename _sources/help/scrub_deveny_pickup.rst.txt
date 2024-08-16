.. code-block:: console

    $ scrub_deveny_pickup -h
    usage: scrub_deveny_pickup [-h] [--proc_dir PROC_DIR] [--overwrite_raw] [-d]
                               [-n]
                               file [file ...]
    
    Clean RF pickup noise from DeVeny raw frames
    
    positional arguments:
      file                 File(s) to clean
    
    options:
      -h, --help           show this help message and exit
      --proc_dir PROC_DIR  Path to the directory above that which contains the
                           .pypeit file used to process `file` (i.e., the directory
                           above ldt_deveny_?) -- use only if `-r` was used in the
                           call to `pypeit_setup`. (default: current working
                           directory)
      --overwrite_raw      Overwrite the raw file rather than create a new file with
                           the '_scrub' suffix (default: False)
      -d, --diagnostics    Output additional information and plots during the
                           analysis for debugging purposes (default: False)
      -n, --no_refit       Force no refit of 'bad' RMS values (default: False)
    