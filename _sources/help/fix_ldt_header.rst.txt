.. code-block:: console

    $ fix_ldt_header -h
    usage: fix_ldt_header [-h] file [file ...] keyword new_value
    
    Fix a keyword in LDT FITS headers
    
    positional arguments:
      file        File(s) on which to operate
      keyword     FITS keyword to change
      new_value   New header keyword value to insert
    
    options:
      -h, --help  show this help message and exit
    