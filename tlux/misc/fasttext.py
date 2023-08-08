
# A memory mapped Fasttext model (slightly slower, but uses much less memory).
class MemmapFasttext:

    # Load a memory mapped fasttext model. Can be provided any of:
    #   path = "<fasttext-model>.bin"  -->  Will load, convert to memmap, save memmap in adjacent path, and load memmap.
    #   path = "<memmap-model-path>"  -->  Will load that memmamp model path directly.
    #   model = fasttext.Fasttext  -->  Will convert to memmap, save in a temporary directory (unless path provided), and load memmap.
    # 
    # Optional:
    #   overwrite  -->  Provide in the case that the memmap directory already exists and should be overwritten.
    #   max_cache_size  -->  Provide an integer or None to set a positive limit to functools.lru_cache around internal methods.
    # 
    def __init__(self, path="", model=None, overwrite=False, max_cache_size=0):
        from tlux.profiling import mem_use
        # If a ".bin" file was provided, then read it and generate a 
        #   MemmapFasttext of that pure Fasttext model in an adjacent path.
        if (path.endswith(".bin")):
            og_path = path
            path += ".mmap"
            import os
            if os.path.exists(path):
                model = None
            else:
                import sys
                if os.path.dirname(__file__) in sys.path:
                    sys.path.remove(os.path.dirname(__file__))
                import fasttext
                model = fasttext.load_model(og_path)
        # This might be an attempt to save a model and convert it to a MemmapFasttext.            
        if ((type(path) is not str) and (type(model) is None)):
            model = path
            path = ""
        if (model is not None):
            if (len(path) == 0):
                import tempfile
                self.temp_dir = tempfile.TemporaryDirectory()
                path = self.temp_dir.name
            self.save_memmap(path=path, model=model, overwrite=overwrite)
        # Store the path locally.
        self.path = path
        # Load the model from the provided path.
        ( 
            config_path, subword_vecs_path, word_vecs_path,
            word_strings_path, word_starts_path, word_ends_path,
            word_order_path
        ) = self.get_paths(path)
        # Read the configuration.
        with open(config_path) as f:
            import json
            for key, value in json.loads(f.read()).items():
                if (key not in {"bucket", "minn", "maxn", "wordsn", "subwordsn", "dim", "real_type", "char_type", "int_type"}):
                    raise(ValueError(f"Unrecognized key in model JSON, '{key}'."))
                setattr(self, key, value)
        # Load the various memory mapped files.
        import numpy as np
        self.subwords_matrix = np.memmap(subword_vecs_path, dtype=self.real_type, shape=(self.subwordsn, self.dim), mode="r")
        self.words_matrix = np.memmap(word_vecs_path, dtype=self.real_type, shape=(self.wordsn, self.dim), mode="r")
        self.word_strings = np.memmap(word_strings_path, dtype=self.char_type, mode="r")
        self.word_start = np.memmap(word_starts_path, dtype=self.int_type, mode="r")
        self.word_end = np.memmap(word_ends_path, dtype=self.int_type, mode="r")
        self.word_order = np.memmap(word_order_path, dtype=self.int_type, mode="r")
        # If caching should be enabled, then decorate internal methods.
        if max_cache_size is None:
            from functools import cache
            self.get_word = cache(self.get_word_at)
            self.get_word_id = cache(self.get_word_id)
            self.get_subword_id = cache(self.get_subword_id)
            self.get_subwords = cache(self.get_subwords)
            self.get_word_vector = cache(self.get_word_vector)
        elif max_cache_size > 0:
            from functools import lru_cache
            self.get_word = lru_cache(maxsize=max_cache_size)(self.get_word_at)
            self.get_word_id = lru_cache(maxsize=max_cache_size)(self.get_word_id)
            self.get_subword_id = lru_cache(maxsize=max_cache_size)(self.get_subword_id)
            self.get_subwords = lru_cache(maxsize=max_cache_size)(self.get_subwords)
            self.get_word_vector = lru_cache(maxsize=max_cache_size)(self.get_word_vector)


    # Return the list of all words (in the original fasttext order).
    @property
    def words(self):
        if not hasattr(self, "_words"):
            self._words = [
                self.get_word_at(i) for i in
                sorted(range(self.wordsn), key=lambda i: self.word_order[i])
            ]
        return self._words


    # Given an index, get the word at that index out of the sorted-order list of words.
    def get_word_at(self, i):
        return "".join(map(chr, self.word_strings[self.word_start[i]:self.word_end[i]]))


    # Get the index associated with a word, return None if there is not an exact match.
    def get_word_id(self, word):
        # Perform a binary search to try and find the word.
        low = 0
        high = self.wordsn
        index = (low + high) // 2
        while high != low:
            word_at_index = self.get_word_at(index)
            if (len(word_at_index) > len(word)):  high = index
            elif (len(word_at_index) < len(word)): low = index + 1
            else:
                if (word_at_index > word):  high = index
                elif (word_at_index < word): low = index + 1
                else: return self.word_order[index]
            index = (low + high) // 2
        # If the word is not found, then it does not exist in the vocabulary.
        return None


    # Mimic the Fasttext hash function as well as the dictionary key lookup
    #  function to produce the index of a subword given its string.
    def get_subword_id(self, string):
        h = 2166136261 # Use Fasttext initial value.
        for c in string:
            h = h ^ ord(c) # Use Fasttext XOR strategy (for bit-level diversity).
            h = h * 16777619 # Use Fasttext multiplier.
            h = h & 0xFFFFFFFF # Keep in uint32 range to mimic C++.
        return h % self.bucket + self.wordsn # Perform the mod and offset used by the Fasttext Dictionary.


    # Given a word, produce the list of subwords and subword IDs contained in the word.
    def get_subwords(self, word):
        subwords = []
        subword_ids = []
        # Add the index of the word if it is known.
        word_id = self.get_word_id(word)
        if word_id is not None:
            subwords.append(word)
            subword_ids.append(word_id)
        # Get all subwords.
        word = "<" + word + ">"
        for i in range(0, len(word)-self.minn+1):
            for j in range(i+self.minn,min(i+self.maxn,len(word))+1):
                subword = word[i:j]
                subwords.append(subword)
                subword_ids.append(self.get_subword_id(subword))
        # Return the list of subwords and IDs.
        return subwords, subword_ids


    # Get the word vector for this word.
    def get_word_vector(self, word):
        import numpy as np
        # First try and retrieve a known word.
        word_id = self.get_word_id(word)
        if (word_id is not None):
            return self.words_matrix[word_id]
        # Otherwise, get the subwords and compute the embedding.
        subwords, subword_ids = self.get_subwords(word)
        vec = np.zeros(self.dim, dtype=float)
        for i in subword_ids:
            vec += self.subwords_matrix[i]
        if (len(subword_ids) > 0):
            vec /= len(subword_ids)
        # Return the vector representation.
        return vec


    # Generate the names of all file paths given the MemmapFasttext directory path.
    @classmethod
    def get_paths(cls, dirname):
        import os
        config_path = os.path.join(dirname, "model_config.json")
        subword_vecs_path = os.path.join(dirname, "subword_vectors.memmap")
        word_vecs_path = os.path.join(dirname, "word_vectors.memmap")
        word_strings_path = os.path.join(dirname, "word_strings.memmap")
        word_starts_path = os.path.join(dirname, "word_start.memmap")
        word_ends_path = os.path.join(dirname, "word_end.memmap")
        word_order_path = os.path.join(dirname, "word_order.memmap")
        return config_path, subword_vecs_path, word_vecs_path, word_strings_path, word_starts_path, word_ends_path, word_order_path


    # Given a Fasttext model, save all necessary data to load it as an mmap.
    @classmethod
    def save_memmap(cls, path, model, overwrite=False, char_type="uint32", int_type="uint32"):
        import os, json
        import numpy as np
        # Make the directory.
        os.makedirs(path, exist_ok=overwrite)
        # Get all file paths.
        dirname = path
        ( 
            config_path, subword_vecs_path, word_vecs_path,
            word_strings_path, word_starts_path, word_ends_path,
            word_order_path
        ) = cls.get_paths(dirname)
        # Save the subwords matrix.
        matrix = model.get_input_matrix()
        real_type = matrix.dtype.name
        subwords_matrix = np.memmap(
            subword_vecs_path, dtype=matrix.dtype,
            mode="w+", shape=matrix.shape
        )
        subwordsn = matrix.shape[0]
        dim = matrix.shape[1]
        subwords_matrix[:,:] = matrix[:,:]
        # We are done with the source matrix.
        del matrix
        # Trigger a write by flushing the memmap array.
        subwords_matrix.flush()
        # Reloade the matrix in "read" mode.
        subwords_matrix = np.memmap(subword_vecs_path, dtype=real_type, mode="r")
        # Save the words matrix.
        wordsn = len(model.words)
        # Vectors.
        words_matrix = np.memmap(
            word_vecs_path, dtype=subwords_matrix.dtype,
            mode="w+", shape=(wordsn, dim)
        )
        all_words_len = 0
        for i,w in enumerate(model.words):
            all_words_len += len(w)
            words_matrix[i,:] = model.get_word_vector(w)
        words_matrix.flush()
        # Strings.
        word_strings = np.memmap(word_strings_path, dtype=char_type, mode="w+", shape=(all_words_len))
        word_start = np.memmap(word_starts_path, dtype=int_type, mode="w+", shape=(wordsn))
        word_end = np.memmap(word_ends_path, dtype=int_type, mode="w+", shape=(wordsn))
        word_order = np.memmap(word_order_path, dtype=int_type, mode="w+", shape=(wordsn))
        # Iterate over words, writing each one.
        start_index = 0
        word_order[:] = sorted(range(wordsn), key=lambda i: (len(model.words[i]), model.words[i]))
        for i,o in enumerate(word_order):
            w = model.words[o]
            end_index = start_index + len(w)
            word_start[i] = start_index
            word_end[i] = end_index
            word_strings[start_index:end_index] = list(map(ord, w))
            start_index = end_index
        word_strings.flush()
        word_start.flush()
        word_end.flush()
        word_order.flush()
        char_type = word_strings.dtype.name
        int_type = word_order.dtype.name
        # Collect and write the details about the model to a JSON.
        args = model.f.getArgs()
        bucket = args.bucket
        minn = args.minn
        maxn = args.maxn
        with open(config_path, "w") as f:
            f.write(json.dumps(dict(
                bucket=bucket, minn=minn, maxn=maxn, wordsn=wordsn, subwordsn=subwordsn,
                dim=dim, real_type=real_type, char_type=char_type, int_type=int_type
            )))



# Build a fasttext model over this file.
def _test_build_model():
    # Disable standard error to prevent fasttext C++ lib from printing.
    import os, sys
    with open(os.devnull, "w") as f:
        os.dup2(f.fileno(), sys.stderr.fileno())
    # Build a fasttext model over this file.
    import fasttext
    model = fasttext.train_unsupervised(__file__, model='skipgram')
    model.save_model("test.bin")


# Compare the memmap model and the fasttext model.
def _compare_mmap_fasttext():
    # Pop this file's directory from the path to avoid name issues with "fasttext" module.
    import os, sys
    if os.path.dirname(__file__) in sys.path:
        sys.path.remove(os.path.dirname(__file__))

    print(" testing memmap fasttext.. building model", end=".. ")
    import multiprocessing
    builder = multiprocessing.Process(target=_test_build_model)
    builder.start()
    builder.join()
    builder.close()

    # Use the profiling function to track memory usage.
    from tlux.profiling import mem_use

    # Get Memmap model memory usage.
    memory_1 = mem_use()
    mmap = MemmapFasttext("test.bin", max_cache_size=None)
    memory_1 = mem_use() - memory_1

    # Get Fasttext model memory usage.
    memory_2 = mem_use()
    import fasttext
    model = fasttext.load_model("test.bin")
    memory_2 = mem_use() - memory_2
    print("size", end=".. ")
    assert (memory_1 / memory_2 < 0.2), f"Memmap model had a larger than expected size ({memory_1}MB) relative to original Fasttext model ({memory_2}MB)."

    # Verify that the words lists are the same.
    print("words", end=".. ")
    assert (tuple(mmap.words) == tuple(model.words)), f"Memmap words did not match fasttext words."

    # Compute the sum total difference between the word vectors.
    total_diff = 0.0
    for w in model.words:
        total_diff += sum(abs(model.get_word_vector(w) - mmap.get_word_vector(w)))
    print("vecs", end=".. ")
    assert (total_diff < 1e-06), f"Total difference between fasttext and memmap was too large.\n  {total_diff}"

    # Compute and compare the embeddings for random words.
    with open(__file__) as f:
        words = f.read().split()

    # Compute difference when embedding strange words.
    max_diff = 0.0
    for w in words:
        max_diff = max(max_diff, max(abs(model.get_word_vector(w) - mmap.get_word_vector(w))))
    print("novel vecs", end=".. ")
    assert (max_diff < 1e-06), f"Maximum difference between fasttext and memmap was too large.\n  {max_diff}"

    # Remove the build models.
    print("cleaning up", end=".. ")
    os.remove("test.bin")
    import shutil
    shutil.rmtree(mmap.path)

    print("passed", end="!\n")

if __name__ == "__main__":
    _compare_mmap_fasttext()



# 2023-08-07 22:42:26
# 
########################################################################################################
# def _test_load_memmap():                                                                             #
#     from tlux.profiling import mem_use                                                               #
#     # Load a memory mapped version of the model.                                                     #
#     print()                                                                                          #
#     print("-"*70)                                                                                    #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     model = MemmapFasttext("memmap_model")                                                           #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     words = model.words                                                                              #
#     print(words)                                                                                     #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     print("len(words): ", len(words), flush=True)                                                    #
#     w = "the"                                                                                        #
#     print("model.get_word_id(w): ", model.get_word_id(w), flush=True)                                #
#     print("model.get_word_vector('"+w+"'): ")                                                        #
#     print(model.get_word_vector(w).round(2), flush=True)                                             #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     print()                                                                                          #
#                                                                                                      #
#                                                                                                      #
# def _test_original_usage():                                                                          #
#     from tlux.profiling import mem_use                                                               #
#     # Import the fasttext python library.                                                            #
#     import fasttext                                                                                  #
#     # Declare the model path.                                                                        #
#     model_path = "test.bin"                                                                          #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     # Load the model and get the number of buckets in the model.                                     #
#     model = fasttext.load_model(model_path)                                                          #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     args = model.f.getArgs()                                                                         #
#     # Get the model words.                                                                           #
#     words = model.words                                                                              #
#     subwords = {sw:i for w in model.words for (sw,i) in zip(*model.get_subwords(w))}                 #
#     subword_ids = set(subwords.values())                                                             #
#     subword_matrix = model.get_input_matrix()                                                        #
#     print("Num words:", len(words))                                                                  #
#     print("words[1]  ", words[1])                                                                    #
#     print("words[134]", words[134])                                                                  #
#     print()                                                                                          #
#     print("Num subwords:", len(subwords))                                                            #
#     print()                                                                                          #
#     print("Size of subword matrix:")                                                                 #
#     print(subword_matrix.shape)                                                                      #
#     print()                                                                                          #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     print()                                                                                          #
#     print("Size of word matrix:")                                                                    #
#     word_matrix = model.get_output_matrix()                                                          #
#     print("word_matrix: ", word_matrix.shape, flush=True)                                            #
#     print()                                                                                          #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     print()                                                                                          #
#                                                                                                      #
#     # Save a memory mapped version of the model.                                                     #
#     print()                                                                                          #
#     print("-"*70)                                                                                    #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     mmap_model = MemmapFasttext("memmap_model", model, overwrite=True)                               #
#     print("mem_use(): ", mem_use(), flush=True)                                                      #
#     print("-"*70)                                                                                    #
#     print()                                                                                          #
#                                                                                                      #
#     # Example word.                                                                                  #
#     word = "the"                                                                                     #
#     word_vec = model.get_word_vector(word)                                                           #
#     word_subwords, word_subword_ids = model.get_subwords(word)                                       #
#     print("word:     ", word, flush=True)                                                            #
#     print(word_vec.round(2))                                                                         #
#     print(subword_matrix[word_subword_ids].mean(axis=0).round(2))                                    #
#     print()                                                                                          #
#                                                                                                      #
#                                                                                                      #
#     min_len = args.minn # 3                                                                          #
#     max_len = args.maxn # 6                                                                          #
#     missing = {}                                                                                     #
#     acc = 0                                                                                          #
#     for wi,w in enumerate(words):                                                                    #
#         if (w == "</s>"): continue                                                                   #
#         l = []                                                                                       #
#         w = "<" + w + ">"                                                                            #
#         for i in range(0, len(w)-min_len+1):                                                         #
#             for j in range(i+min_len,min(i+max_len,len(w))+1):                                       #
#                 sw = w[i:j]                                                                          #
#                 l.append(sw)                                                                         #
#                 acc = max(acc, len(sw))                                                              #
#                 if sw not in subwords:                                                               #
#                     # print("", sw)                                                                  #
#                     missing[sw] = missing.get(sw,0) + 1                                              #
#         # Compute subwords here.                                                                     #
#         ids = [wi] + sorted(map(model.get_subword_id, l))                                            #
#         if (w[1:-1] in words): l.append(w[1:-1])                                                     #
#         #                                                                                            #
#         # For demonstration purposes.                                                                #
#         if (wi == 100):                                                                              #
#             print("w: ", w, flush=True)                                                              #
#             # Compute "true" subwords.                                                               #
#             word_subwords, word_subword_ids = model.get_subwords(w[1:-1])                            #
#             # Compute subwords with local function.                                                  #
#             local_subwords, local_subword_ids = mmap_model.get_subwords(w[1:-1])                     #
#             #                                                                                        #
#             #                                                                                        #
#             # Subword IDs.                                                                           #
#             print(" (true subword ids): ", sorted(word_subword_ids), flush=True)                     #
#             print(" (made subword ids): ", sorted(ids), flush=True)                                  #
#             print(" ( new subword ids): ", sorted(local_subword_ids), flush=True)                    #
#             # Subwords.                                                                              #
#             print(" (true subwords):    ", sorted(word_subwords))                                    #
#             print(" (made subwords):    ", sorted(l))                                                #
#             print(" ( new subwords):    ", sorted(local_subwords))                                   #
#             break                                                                                    #
#                                                                                                      #
#     print("max subword length (acc): ", acc, flush=True)                                             #
#     print("len(missing) subwords: ", len(missing), flush=True)                                       #
#     print()                                                                                          #
#     for kv in sorted(missing.items(), key=lambda kv: -kv[1])[:10]:                                   #
#         print(kv[0], kv[1])                                                                          #
#     print()                                                                                          #
#                                                                                                      #
#     for i in range(len(words)+1, len(subword_matrix)):                                               #
#         if i not in subword_ids:                                                                     #
#             print(i)                                                                                 #
#             print(subword_matrix[i].round(2))                                                        #
#             break                                                                                    #
#                                                                                                      #
#     print()                                                                                          #
#     fake_subword = "xkdo"                                                                            #
#     print("true ID", model.get_subword_id(fake_subword))                                             #
#     print("appr ID", mmap_model.get_subword_id(fake_subword))                                        #
#     print()                                                                                          #
#     print("fake subword in word subwords?", fake_subword in word_subwords)                           #
#     print()                                                                                          #
#     print(subword_matrix[model.get_subword_id(fake_subword)].round(2))                               #
#     print()                                                                                          #
#     print("max diff between memmap and true:",                                                       #
#           abs(mmap_model.get_word_vector(fake_subword) - model.get_word_vector(fake_subword)).max()) #
########################################################################################################

