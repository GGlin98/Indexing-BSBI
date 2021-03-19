# Indexing-BSBI
 Blocked Sort-Based Indexing Python Implementation
# Usage
```
main.py [-h] [-s SIZE] [-u {K,M,G,k,m,g}] [-d DIR] [-o OUTPUT] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  Block size (an integer)
  -u {K,M,G,k,m,g}, --unit {K,M,G,k,m,g}
                        Block size unit, in [K, M, G]
  -d DIR, --dir DIR     The directory path for input documents
  -o OUTPUT, --output OUTPUT
                        The output directory path
  -v, --verbose         Track and display memory usage (will degrade the performance)
  ```
  For example:
  ```
  python main.py --size 2 --unit m --dir Documents --output Output --verbose
  ```