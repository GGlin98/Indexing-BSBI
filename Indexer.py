import gc
import os
import sys
import time
from collections import deque
from string import punctuation

from guppy import hpy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pympler import asizeof


class IndexerBSBI:
    def __init__(self, dir_name, block_size, output_dir, verbose):
        self.block_size = block_size
        self.dir_name = dir_name
        self.output_dir = output_dir
        self.docId_to_doc = {}
        self.number_of_docs = 0
        self.current_block = 0
        self.total_docs_size = 0
        self.current_docId = 0
        self.current_file = 0
        self.punctuation = punctuation + '-—'
        self.sorting_times = []
        self.term_to_docIds = {}
        self.term_to_docIds_sorted = []
        self.docId_to_terms = None
        self.h = hpy()
        self.memory_track = []
        self.current_termId = 0
        self.verbose = verbose
        self.total_bytes = 0  # Current block size
        self.eof = False
        self.porter = PorterStemmer()

    def construct_index(self):
        if self.verbose:
            self.memory_track.append(self.h.heap().size)
        timer_start = time.process_time()
        self.total_docs_size = 0

        # Create doc-docID mapping and calculate total doc size
        doc_index = 0
        for dirpath, dirnames, filenames in os.walk(self.dir_name):
            filenames.sort()
            for f in filenames:
                self.docId_to_doc[doc_index] = f
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    sz = os.path.getsize(fp)
                    if sz >= self.block_size:
                        # Reach the block size limit
                        print('Error: {} size exceeds block size limit: {} KB > {} KB'.format(
                            self.docId_to_doc[self.current_docId],
                            sz / 1024,
                            self.block_size / 1024))
                        sys.exit(1)
                    self.total_docs_size += sz
                doc_index += 1

        self.number_of_docs = len(self.docId_to_doc)

        self.print_info()

        # Create and clean output directory
        if os.path.exists(self.output_dir):
            if not os.path.isdir(self.output_dir):
                os.remove(self.output_dir)
                os.mkdir(self.output_dir)
            else:
                for dirpath, dirnames, filenames in os.walk(self.output_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        os.remove(fp)
        else:
            os.mkdir(self.output_dir)

        # Construct index
        print('Current block: {}  Current doc ID: {}'.format(self.current_block, self.current_docId), end='')
        while True:
            # Parse and invert every doc
            self.parse_next_doc()
            self.invert_doc()

            sz = sys.getsizeof(self.term_to_docIds)
            current_block_size = self.total_bytes + sz
            if self.eof or current_block_size >= self.block_size:
                # Last block or reaches block size limit, sort the term-doc pairs based on terms and write to disk
                if self.verbose:
                    sz = asizeof.asizeof(self.term_to_docIds)
                    print(' Current block size: {} bytes ({:.2f} KB or {:.2f} MB)'.format(sz, sz / 1024,
                                                                                          sz / (1024 ** 2)))
                else:
                    print()

                # Sort the postings based on terms and start timing
                timer_start_sort = time.process_time()
                terms = list(self.term_to_docIds.keys())
                terms.sort()
                self.term_to_docIds_sorted = []
                for term in terms:
                    self.term_to_docIds_sorted.append((term, self.term_to_docIds[term]))
                    del self.term_to_docIds[term]
                self.term_to_docIds = sorted(self.term_to_docIds.items())
                timer_end_sort = time.process_time()
                self.sorting_times.append(timer_end_sort - timer_start_sort)
                self.write_block()
                self.current_block += 1
                self.term_to_docIds = {}
                self.total_bytes = 0
                gc.collect()
                if not self.eof:
                    print('Current block: {}  Current doc ID: {}'.format(self.current_block, self.current_docId),
                          end='')
                else:
                    # Last block, start merging
                    print()
                    break

        self.merge_blocks()

        timer_end = time.process_time()

        print('Processing time of indexing the whole dataset: {:.4f}s'.format(timer_end - timer_start))
        print('Average processing time of sorting a block: {:.4f}s'
              .format(sum(self.sorting_times) / self.current_block))

        self.clean_output_dir()

        if self.verbose:
            # Output memory increment & total memory usage
            self.memory_track.append(self.h.heap().size)
            high, low = self.memory_track[0], self.memory_track[0]
            for m in self.memory_track[1:]:
                if m > high:
                    high = m
                elif m < low:
                    low = m
            memory_usage = high - low

            print('Total memory increment: {} bytes ({:.4f} KB or {:.4f} MB)'.format(memory_usage, memory_usage / 1024,
                                                                                     memory_usage / (1024 ** 2)))
            print('Total memory usage: {} bytes ({:.4f} KB or {:.4f} MB)'.format(high, high / 1024,
                                                                                 high / (1024 ** 2)))

    def preprocess(self, terms):
        sz = len(terms)
        i = 0
        while i < sz:
            j = 0
            # Remove the terms that consist only punctuations
            for x in terms[i]:
                if x not in self.punctuation:
                    break
                else:
                    j += 1
            else:
                # All characters are punctuations, remove the term
                del terms[i]
                sz -= 1
                continue

            if j != 0:
                # Remove prefix punctuations
                terms[i] = terms[i][j:]

            terms[i] = terms[i].lower()
            terms[i] = self.porter.stem(terms[i])

            i += 1

    def parse_next_doc(self):
        self.docId_to_terms = None

        # Get next doc and its size
        if self.current_docId > self.number_of_docs - 1:
            self.eof = True
            return

        current_doc = self.docId_to_doc[self.current_docId]
        current_doc = os.path.join(self.dir_name, current_doc)
        self.current_file = open(current_doc, 'rt', encoding='utf-8')
        current_file_size = os.path.getsize(current_doc)

        if current_file_size <= self.block_size:
            # Parse all the doc
            self.docId_to_terms = [self.current_docId, self.current_file.read()]
            self.current_file.close()
            self.current_docId += 1
        else:
            # Reach the block size limit
            print('\nError: {} size exceeds block size limit: {} KB > {} KB'.format(
                self.docId_to_doc[self.current_docId],
                current_file_size / 1024,
                self.block_size / 1024))
            self.eof = True

        # Extract tokens from a doc and preprocess them into terms
        self.docId_to_terms[1] = word_tokenize(self.docId_to_terms[1])
        self.preprocess(self.docId_to_terms[1])

    def invert_doc(self):
        if self.eof:
            return

        # Store the terms into term-docs postings
        i = 0
        j = 0  # Number of duplicate terms
        doc_id = self.docId_to_terms[0]
        terms = self.docId_to_terms[1]
        while i < len(terms):
            term = terms[i]
            if term not in self.term_to_docIds:
                j += 1
                sz = sys.getsizeof(term)
                inc = sz % 8
                if inc != 0:
                    sz += 8 - inc  # real size of the term
                self.term_to_docIds[term] = []
                self.total_bytes += sz  # size of term
            self.term_to_docIds[term].append(doc_id)
            i += 1

        # Get the approximation of the block size
        sz = sys.getsizeof(doc_id)
        self.total_bytes += sz
        self.total_bytes += j * 56  # size of []
        self.total_bytes += j * + sz  # size of the doc ID elements
        self.total_bytes += 8 * i  # size of a reference in []

        del self.docId_to_terms, terms

    def write_block(self):
        if self.verbose:
            self.memory_track.append(self.h.heap().size)

        with open('{}/block{}.txt'.format(self.output_dir, self.current_block), 'wt', encoding='utf-8') as f:
            for term, docs in self.term_to_docIds_sorted:
                for doc in docs:
                    # Write term-docID pairs
                    f.write('{} {}\n'.format(term, doc))

        del self.term_to_docIds, f
        gc.collect()

    def merge_blocks(self):
        if self.verbose:
            self.memory_track.append(self.h.heap().size)
        print('Merging...')
        timer_start = time.process_time()
        queue = deque()
        i = 0  # Merged block ID

        # Get all runs
        for dirpath, _, filenames in os.walk(self.output_dir):
            for file in filenames:
                fp = os.path.join(dirpath, file)
                queue.append(fp)

        del filenames, file, dirpath

        while len(queue) > 1:
            # Still has blocks be merged
            filepath_a = queue.popleft()
            filepath_b = queue.popleft()
            merged_path = os.path.join(self.output_dir, 'merged{}.txt'.format(i))
            file_a = open(filepath_a, 'rt', encoding='utf-8')
            file_b = open(filepath_b, 'rt', encoding='utf-8')
            td_a = file_a.readline()  # Term-Doc pair a
            td_b = file_b.readline()  # Term-Doc pair b
            if td_a == '' or td_b == '':
                # Special treatment for empty files
                file_a.close()
                file_b.close()
                if td_a == '':
                    os.rename(filepath_b, merged_path)
                else:
                    os.rename(filepath_a, merged_path)
                queue.append(merged_path)
                i += 1
                continue
            file_merged = open(merged_path, 'wt', encoding='utf-8')
            td_a = td_a[:-1].split(' ')
            td_b = td_b[:-1].split(' ')
            while True:
                # Write the term-doc pairs line by line to the merged file until one of the file reaches EOF
                if td_a[0] < td_b[0] or (td_a[0] == td_b[0] and int(td_a[1]) < int(td_b[1])):
                    td = td_a
                    ptr = file_a
                else:
                    td = td_b
                    ptr = file_b
                file_merged.write(td[0] + ' ' + td[1] + '\n')
                if ptr == file_a:
                    td_a = file_a.readline()
                    if td_a == '':
                        break
                    td_a = td_a[:-1].split(' ')
                else:
                    td_b = file_b.readline()
                    if td_b == '':
                        break
                    td_b = td_b[:-1].split(' ')

            # Check which file has reaches EOF and close it
            if td_a == '':
                ptr = file_b
                file_a.close()
            else:
                ptr = file_a
                file_b.close()
            td = ptr.readline()
            while td != '':
                # Write remaining lines to the merged file
                td = td[:-1].split(' ')
                file_merged.write(td[0] + ' ' + td[1] + '\n')
                td = ptr.readline()
            ptr.close()
            file_merged.close()
            queue.append(merged_path)
            i += 1

        # Create the output file from the last merged file (Convert the doc ID each line into doc name)
        merged_file = open(queue.pop(), 'rt', encoding='utf-8')
        output_file = open(os.path.join(self.output_dir, 'output.txt'), 'wt', encoding='utf-8')
        td = merged_file.readline()
        while td != '':
            td = td[:-1].split(' ')
            output_file.write(td[0] + ' ' + self.docId_to_doc[int(td[1])] + '\n')
            td = merged_file.readline()
        merged_file.close()
        output_file.close()

        timer_end = time.process_time()
        if self.verbose:
            self.memory_track.append(self.h.heap().size)
            print('merge_blocks() memory usage: {} bytes ({:.4f} KB or {:.4f} MB)'
                  .format(self.memory_track[-1] - self.memory_track[-2],
                          (self.memory_track[-1] - self.memory_track[-2]) / 1024,
                          (self.memory_track[-1] - self.memory_track[-2]) / (1024 ** 2)))
        print('Processing time of merging blocks: {:.4f}s'.format(timer_end - timer_start))

    def clean_output_dir(self):
        for dirpath, dirnames, filenames in os.walk(self.output_dir):
            for file in filenames:
                if file == 'output.txt':
                    continue
                fp = os.path.join(dirpath, file)
                os.remove(fp)

    def print_info(self):
        print('Input directory: {}'.format(os.path.abspath(self.dir_name)))
        print('Output directory: {}'.format(os.path.abspath(self.output_dir)))
        print('Total number of documents: {}'.format(self.number_of_docs))
        print('Total size of documents: {} bytes ({:.0f} KB or {:.2f} MB)'.format(self.total_docs_size,
                                                                                  self.total_docs_size / 1024,
                                                                                  self.total_docs_size / (1024 ** 2)))
        print('Block size: {} bytes ({:.0f} KB or {:.2f} MB)'
              .format(self.block_size,
                      self.block_size / 1024,
                      self.block_size / (1024 ** 2)))
        print('Verbose mode: {}'.format(self.verbose))
