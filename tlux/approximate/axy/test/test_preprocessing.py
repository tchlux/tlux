import os, random
import numpy as np
import ctypes
from tlux.approximate.axy.preprocessing import i_map, i_encode, to_array
from tlux.unique import unique, to_int
from tlux.profiling import profile



if __name__ == "__main__":
    random.seed()
    num_words = 10
    max_word_length = 16
    path = f"words-{max_word_length}_{num_words}.npz"
    if (not os.path.exists(path)):
        print(f"Generating {num_words} words..", flush=True)
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
        print(f"Loading '{path}'..", flush=True)
        xi = np.load(path, allow_pickle=True)['xi']

    print("Modifying..", flush=True)
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
    print("Calling i_map..", flush=True)
    xi_map = i_map(xi, [])
    if (hasattr(unique, "show_profile")): unique.show_profile()
    if (hasattr(i_map, "show_profile")): i_map.show_profile()

    for i,m in enumerate(xi_map):
        print(f" map {i} (len {len(m)})", m)

    # Encode the data in integer format.
    print("Calling i_encode..", flush=True)
    final_xi = i_encode(xi, xi_map)
    if (hasattr(to_int, "show_profile")): to_int.show_profile()
    if (hasattr(i_encode, "show_profile")): i_encode.show_profile()

    # Print the source data and the final integer version.
    print(xi)
    print(final_xi)
