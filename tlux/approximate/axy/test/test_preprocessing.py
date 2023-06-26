import os, random
import numpy as np
import ctypes
from tlux.approximate.axy.preprocessing import i_map, i_encode, to_array
from tlux.unique import unique, to_int
from tlux.profiling import profile



def _test_imap(num_words=10, max_word_length=16, leave_save=False, verbose=False):
    if (not verbose): print("test imap", ("load file " if leave_save == False else "save file "), end="...", flush=True)
    random.seed(0)
    path = f"words-{max_word_length}_{num_words}.npz"
    if (not os.path.exists(path)):
        if verbose: print(f"Generating {num_words} words..", flush=True)
        # Generate random words.
        words = [[
            "".join((chr(random.randint(ord('0'),ord('z')))
                     for _ in range(random.randint(0,max_word_length))))
              .replace("\\", "/")
              .replace("`", ".")
            , random.randint(0, max_word_length**2)] for _ in range(num_words)
        ]
        # Convert those words to a numpy array.
        xi = np.asarray(words, dtype=object)
        # Save the words to file (for later reuse).
        np.savez_compressed(path, xi=xi)
    else:
        if verbose: print(f"Loading '{path}'..", flush=True)
        xi = np.load(path, allow_pickle=True)['xi']

    # Delete the save file if "leave_save" is false.
    if (not leave_save):
        os.remove(path)

    # Modify the shape.
    if verbose: print("Modifying..", flush=True)
    xi = xi.reshape((-1,2))

    # @profile
    # def test(steps=3):
    #     for _ in range(steps):
    #         print("", _, end="\r")
    #         res = unique(xi)
    #         del(res)
    # for _ in range(10):
    #     test()
    #     test.show_profile()
    # print()
    # unique.show_profile()
    # exit()

    # Convert data into the xi_map.
    if verbose: print("Calling i_map..", flush=True)
    xi_map = i_map(xi, [])
    if verbose:
        if (hasattr(unique, "show_profile")): unique.show_profile()
        if (hasattr(i_map, "show_profile")): i_map.show_profile()
        for i,m in enumerate(xi_map):
            print(f" map {i} (len {len(m)})", m)

    # Assert that the maps have the expected contents in the expected order.
    #   order first by length, second by ordinal value
    assert (tuple(xi_map[0]) == ('@C', 'ub', 'VvU', ']gX', 'HG4Ql', 'Bui;:Xq', 'A<PtBW<9Z', 'uJvTh;aXyN', 'e5QqncVm]zKp', 'hrQ7v1;c0oZOY8H'))
    assert (tuple(xi_map[1]) == ('19', '35', '63', '71', '104', '113', '141', '148', '241', '250'))

    # Encode the data in integer format.
    if verbose: print("Calling i_encode..", flush=True)
    final_xi = i_encode(xi, xi_map)
    if verbose:
        if (hasattr(to_int, "show_profile")): to_int.show_profile()
        if (hasattr(i_encode, "show_profile")): i_encode.show_profile()
        # Print the source data and the final integer version.
        print(xi)
        print(final_xi)

    assert (final_xi == np.asarray(
        [[9, 14],
         [7, 19],
         [4, 15],
         [10, 16],
         [6, 20],
         [3, 13],
         [8, 18],
         [5, 12],
         [1, 11],
         [2, 17]]
    )).all()

    if (not verbose): print(" passed.", flush=True)

if __name__ == "__main__":
    _test_imap(verbose=False, leave_save=True)
    _test_imap(verbose=False, leave_save=False)
