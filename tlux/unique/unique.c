/*

This C library is designed to quick identify unique elements in an
array and then map subsequent (and similar) arrays to integer format
where each unique value is replaced by a unique integer in the range
[0, n] where 'n' is the number of unique values in total and 0 is
reserved for representing "unrecognized" values.

Where possible, it utilizes OpenMP shared-memory parallelism to accelerate
the loops of similar actions. Define the preprocessor variable _UNIQUE_SINGLE_THREADED
to disable all OpenMP parallelism (and disable the inclusion of 'omp.h').

Define the preprocessor variable _UNIQUE_DEBUG_ENABLED to enable
verbose logging of internal operations for debugging purposes to stderr.

Author: Thomas C.H. Lux
License: MIT
Modification history:
   April 2023 -- Created and tested. (704 lines total)

*/


// Debugging definitions.
/* #define _UNIQUE_DEBUG_ENABLED */

// Max to show in debug mode.
#define _UNIQUE_HALF_MAX_EXAMPLES 50

// Whether local functions are allowed (required for comparison of custom width objects).
// #define _UNIQUE_ALLOW_LOCAL_FUNCTIONS

// Use if OpenMP doesn't exist, or if only single threaded behavior is desired.
// #define _UNIQUE_SINGLE_THREADED

// Standard libraries.
#include <stdio.h>  // printf
#include <stdlib.h>  // malloc, free
#include <string.h>  // strlen, strcpy

// Handle parallelism (only use OpenMP if _UNIQUE_SINGLE_THREADED is not defined).
#ifdef _UNIQUE_SINGLE_THREADED
// Define the OpenMP library functions to return values that correspond to single threaded behavior.
int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }
int omp_get_num_threads() { return 1; }
#else
#include <omp.h> // omp_get_thread_num, omp_get_num_threads, omp_get_max_threads
#endif


// -------------------------------------------------------------------------------------------------
// Return 1 if the two provided arrays overlap with each other, 0 if they
//  do not overlap with each other.
const unsigned char arrays_overlap(const int element_width,
                                   const long n1, const void* array1,
                                   const long n2, const void* array2) {
  const long start_a1 = (long) array1;
  const long end_a1 = ((long) array1) + (element_width * n1) - 1;
  const long start_a2 = (long) array2;
  const long end_a2 = ((long) array2) + (element_width * n2) - 1;
  const unsigned char overlap = (
    ((start_a1 >= start_a2) && (start_a1 <= end_a2)) ||
    ((end_a1 >= start_a2) && (end_a1 <= end_a2))
  ) ? 1 : 0;
  return overlap;
}

// -------------------------------------------------------------------------------------------------
// Compare two strings.
int str_compare(const void* a_in, const void* b_in) {
  const char* a = *(const char**) a_in;
  const char* b = *(const char**) b_in;
  int a_len = strlen(a);
  int b_len = strlen(b);
  int comparison_result = 0;
  if (a_len < b_len) {
    comparison_result = -1;
  } else if (a_len > b_len) {
    comparison_result = 1;
  } else {
    // These strings have equal length, compare their characters.
    for (int i = 0; i < a_len; i++) {
      if (a[i] < b[i]) {
        comparison_result = -1;
        break;
      } else if (a[i] > b[i]) {
        comparison_result = 1;
        break;
      }
    }
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"  unique.c[str_cmp] -- '%s' <= '%s'  =  %d\n", a, b, comparison_result);
  #endif
  return comparison_result;
}

// -------------------------------------------------------------------------------------------------
// Compare two characters (or any 1 byte elements).
int char_compare(const void* a_in, const void* b_in) {
  const unsigned char* a = *((const unsigned char**) a_in);
  const unsigned char* b = *((const unsigned char**) b_in);
  int comparison_result = 0;
  if (*a < *b) {
    comparison_result = -1;
  } else if (*a > *b) {
    comparison_result = 1;
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"  unique.c[char_cmp] -- %d <= %d  =  %d\n", *a, *b, comparison_result);
  #endif
  return comparison_result;
}

// -------------------------------------------------------------------------------------------------
// Compare two integers (or any 4 byte elements).
int int_compare(const void* a_in, const void* b_in) {
  const int* a = *((const int**) a_in);
  const int* b = *((const int**) b_in);
  int comparison_result = 0;
  if (*a < *b) {
    comparison_result = -1;
  } else if (*a > *b) {
    comparison_result = 1;
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"  unique.c[int_cmp] -- %d <= %d  =  %d\n", *a, *b, comparison_result);
  #endif
  return comparison_result;
}

// -------------------------------------------------------------------------------------------------
// Compare two longs (or any 8 byte elements).
int long_compare(const void* a_in, const void* b_in) {
  const long* a = *((const long**) a_in);
  const long* b = *((const long**) b_in);
  int comparison_result = 0;
  if (*a < *b) {
    comparison_result = -1;
  } else if (*a > *b) {
    comparison_result = 1;
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"  unique.c[long_cmp] -- %ld <= %ld  =  %d\n", *a, *b, comparison_result);
  #endif
  return comparison_result;
}

// -------------------------------------------------------------------------------------------------
// Given an array of fixefd width objects (or strings), generate a sorted list of unique values.
void unique(char**restrict array_in, const long num_elements, const int width,
            long*restrict num_unique, char***restrict sorted_unique) {
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"=============> unique <---------------\n");
  fprintf(stderr,"unique.c[unique] Array at %lu with length %ld and byte width %d.\n",
          (unsigned long) array_in, num_elements, width);
  #endif
  // If the byte-size width is positive, then we need to allocate and fill an
  //  array of pointers assuming the provided pointer is to contiguous memory.
  char**restrict array;
  if (width == -1) {
    array = array_in;
  } else {
    array = (char**) malloc(num_elements * sizeof(char*)); // ALLOCATION: pointer variant of (flat) array_in
    // Check for memory overlap between 'array' and the provided 'array_in'.
    while (arrays_overlap(width, num_elements, array_in, num_elements, array)) {
      fprintf(stderr,"ERROR unique.c[unique] -- Overlap in allocated memory locations for array[%lu:%lu] and provided array_in[%lu:%lu].\n",
              (unsigned long) array, ((unsigned long)(&(array[num_elements]))) + (width-1), 
              (unsigned long) array_in, ((unsigned long)(&(array_in[num_elements]))) + (width-1));
      // This should NEVER happen if 'malloc' is working correctly, but since it DOES happen
      //  occasionally in testing (macOS 13.2, python 3.11.2, numpy 1.24.2, gfortran 12.2.0),
      //  we will allocate a new array until we create one that doesn't overlap.
      char** old_array = array;
      array = (char**) malloc(num_elements * sizeof(char*)); // ALLOCATION: pointer variant of (flat) array_in
      free(old_array); // DEALLOCATION: pointer variant of (flat) array_in
      fprintf(stderr,"unique.c Line 177: long at %lu (array in at %lu)\n", (unsigned long)array, (unsigned long)array_in);
    }

    /* fprintf(stderr,"unique.c[unique] -- Allocated memory locations for array[%lu:%lu], provided array_in[%lu:%lu].\n", */
    /*         (unsigned long) array, ((unsigned long)(&(array[num_elements]))) + (width-1),  */
    /*         (unsigned long) array_in, ((unsigned long)(&(array_in[num_elements]))) + (width-1)); */

    for (long i=0; i<num_elements; i++) {
      array[i] = &(((char*)array_in)[i*width]);
      #ifdef _UNIQUE_DEBUG_ENABLED
      if ((i <= _UNIQUE_HALF_MAX_EXAMPLES) || (i >= num_elements-_UNIQUE_HALF_MAX_EXAMPLES)) {
        if (width == -1) {
          fprintf(stderr," unique.c[unique] -- array[%ld] = '%s' is at %lu\n", i, array[i], (unsigned long)(array[i]));
        } else if (width == 1) {
          fprintf(stderr," unique.c[unique] -- array[%ld] = %u is at %lu\n", i, *((unsigned char*)(array[i])), (unsigned long)(array[i]));
        } else if (width == 4) {
          fprintf(stderr," unique.c[unique] -- array[%ld] = %d is at %lu\n", i, *((int*)(array[i])), (unsigned long)(array[i]));
        } else if (width == 8) {
          fprintf(stderr," unique.c[unique] -- array[%ld] = %lu is at %lu\n", i, *(((long**)array)[i]), (unsigned long)(array[i]));
        } else {
          fprintf(stderr," unique.c[unique] -- array[%ld] is at %lu\n", i, (unsigned long)(array[i]));
        }
      }
      #endif
    }
  }
  // Set the comparison function based on byte width.
  int (*byte_compare)(const void*, const void*);
  if (width == -1) {
    byte_compare = &str_compare;
  } else if (width == 1) {
    byte_compare = &char_compare;
  } else if (width == 4) {
    byte_compare = &int_compare;
  } else if (width == 8) {
    byte_compare = &long_compare;
  } else {
    #ifdef _UNIQUE_ALLOW_LOCAL_FUNCTIONS
    // Compare two fixed size arrays, treat as unsigned integers.
    int custom_compare(const void* a_in, const void* b_in) {
      const unsigned char* a = *(const unsigned char**) a_in;
      const unsigned char* b = *(const unsigned char**) b_in;
      int comparison_result = 0;
      // These strings have equal length, compare their characters.
      for (int i = 0; i < width; i++) {
        if (a[i] < b[i]) {
          comparison_result = -1;
          break;
        } else if (a[i] > b[i]) {
          comparison_result = 1;
          break;
        }
      }
      #ifdef _UNIQUE_DEBUG_ENABLED
      fprintf(stderr,"  unique.c[custom_cmp] -- %lu <= %lu  =  %d\n", a, b, comparison_result);
      #endif
      return comparison_result;
    }
    byte_compare = &custom_compare;
    #else
    fprintf(stderr,"ERROR unique.c[unique] There is no known comparison function for elements of %d bytes.\n", width);
    (*num_unique) = -1;
    (*sorted_unique) = NULL;
    return;
    #endif
  }
  // Set the number of parallel threads to use.
  int num_threads = omp_get_max_threads();
  if (num_elements < num_threads) num_threads = 1;
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"unique.c[unique] -- Array[0:%ld] (%d thread%s\n", num_elements, num_threads, (num_threads > 1 ? "s)" : ")"));
  #endif
  // Generate the start and end indices of chunks of the array for parallel sorting.
  long*restrict chunk_indices = (long*) malloc(num_threads * sizeof(long)); // ALLOCATION: chunk indices
  long*restrict chunk_ends = (long*) malloc(num_threads * sizeof(long)); // ALLOCATION: chunk ends
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"unique.c[unique] -- Allocated chunk_indices at %lu and chunk_ends at %lu.\n",
          (unsigned long) chunk_indices, (unsigned long) chunk_ends);
  #endif
  const long chunk_size = num_elements / num_threads;
  chunk_indices[0] = 0;
  for (int i=0; i < (num_elements%num_threads); i++) {
    chunk_ends[i] = chunk_indices[i] + chunk_size + 1;
    if (i+1 < num_threads) {
      chunk_indices[i+1] = chunk_ends[i];
    }
  }
  for (int i=(num_elements%num_threads); i < num_threads; i++) {
    chunk_ends[i] = chunk_indices[i] + chunk_size;
    if (i+1 < num_threads) {
      chunk_indices[i+1] = chunk_ends[i];
    }
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"unique.c[unique] -- Chunk size %ld\n", chunk_size);
  for (int i=0; i<num_threads; i++) {
    fprintf(stderr,"  unique.c[unique] -- Chunk %d covers [%ld, %ld)\n", i, chunk_indices[i], chunk_ends[i]);
  }
  #endif
  // Sort chunks of the array (in parallel).
  #pragma omp parallel num_threads(num_threads) if(num_threads > 1)
  {
    const int tid = omp_get_thread_num();  // [0, .., nt-1].
    const unsigned long start = chunk_indices[tid]; // Index of first element.
    const unsigned long end = chunk_ends[tid];      // EXCLUSIVE, end = last element + 1.
    #ifdef _UNIQUE_DEBUG_ENABLED
    fprintf(stderr,"unique.c[unique] -- Thread %d sorting Array[%lu:%lu]\n", tid, start, end);
    #endif
    qsort(&array[start], end - start, sizeof(char*), *byte_compare);
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  // Show a sample of the results.
  for (int i=0; i<num_elements; i += (num_elements < 10) ? 1 : (num_elements / 10)) {
    if (width == -1) {
      fprintf(stderr,"unique.c[unique] -- array[%d] = '%s'\n", i, array[i]);
    } else if (width == 1) {
      fprintf(stderr,"unique.c[unique] -- array[%d] = %u\n", i, *((unsigned char*)(array[i])));
    } else if (width == 4) {
      fprintf(stderr,"unique.c[unique] -- array[%d] = %d\n", i, *((int*)(array[i])));
    } else if (width == 8) {
      fprintf(stderr,"unique.c[unique] -- array[%d] = %ld\n", i, *((long*)(array[i])));
    } else {
      fprintf(stderr,"unique.c[unique] -- array[%d] at %lu\n", i, (unsigned long) array[i]);
    }
  }
  #endif
  // Initialize space and variables for computing unique elements.
  (*sorted_unique) = (char**) malloc(num_elements * sizeof(char*)); // ALLOCATION: tentative unique words holder
  while (arrays_overlap(width, num_elements, array_in, num_elements, (*sorted_unique))) {
    fprintf(stderr,"ERROR unique.c[unique] -- Overlap in allocated memory locations for sorted_unique[%lu:%lu] and provided array_in[%lu:%lu].\n",
            (unsigned long) (*sorted_unique), ((unsigned long)(&((*sorted_unique)[num_elements]))) + (width-1),
            (unsigned long) array_in, ((unsigned long)(&(array_in[num_elements]))) + (width-1));
    // This should NEVER happen if 'malloc' is working correctly, but since it DOES happen
    //  occasionally in testing (macOS 13.2, python 3.11.2, numpy 1.24.2, gfortran 12.2.0),
    //  we will allocate a new array until we create one that doesn't overlap.
    char** old_array = (*sorted_unique);
    (*sorted_unique) = (char**) malloc(num_elements * sizeof(char*)); // ALLOCATION: tentative unique words holder
    free(old_array); // DEALLOCATION: tentative unique words holder
  }
  (*num_unique) = 0;
  char* v_prev = NULL;
  char* v_next = "init"; // Initial value doesn't matter as long as it's not NULL.
  int i_next = -1;
  #ifdef _UNIQUE_DEBUG_ENABLED  
  fprintf(stderr,"unique.c[unique] -- Allocated space for %ld unique values at %lu, finding unique values..\n",
          num_elements, (unsigned long) (*sorted_unique));
  #endif
  // Loop and collect all the unique elements.
  while (v_next != NULL) {
    v_next = NULL;
    i_next = -1;
    // If there was a previous value added, then rotate all chunks until getting
    //  to a "new" value in each of the chunks arrays (or exhausting the array).
    if (v_prev != NULL) {
      for (int j = 0; j < num_threads; j++) {
        // Rotate to the next string in this chunk until we find a new value.
        while ((chunk_indices[j] < chunk_ends[j])
               && ((*byte_compare)(&v_prev, &array[chunk_indices[j]]) == 0)) {
          chunk_indices[j]++;
        }
      }
    }
    // Find the "least" string from each of the chunks to add next.
    for (int j = 0; j < num_threads; j++) {
      // If there is another value in this chunk to consider, then..
      if (chunk_indices[j] < chunk_ends[j]) {
        // If there is not a previous candidate.
        if (v_next == NULL) {
          v_next = array[chunk_indices[j]];
          i_next = j;
        } else {
          char* alternative = array[chunk_indices[j]];
          int comparison = (*byte_compare)(&alternative, &v_next);
          // If the new string is lesser, then store it as the next candidate.
          if (comparison < 0) {
            v_next = alternative;
            i_next = j;
          }
          // If the string was equal to the current candidate, iterate that chunk (duplicate).
          else if (comparison == 0) {
            chunk_indices[j]++;
          }
          // Otherwise, the string was greater, so it's not getting added this time.
        } // END if (v_next == NULL)
      } // END if (chunk_indices[j] < chunk_ends[j])
    } // END for (int j = 0; j < nt; n++) {
    // Store the next unique value and increment our index.
    if (v_next != NULL) {
      #ifdef _UNIQUE_DEBUG_ENABLED
      if (width == -1) {
        fprintf(stderr,"unique.c[unique] -- Found next '%s' at %lu.\n", v_next, (unsigned long) v_next);
      } else if (width == 1) {
        fprintf(stderr,"unique.c[unique] -- Found next %u at %lu.\n", *((unsigned char*)(v_next)), (unsigned long) v_next);
      } else if (width == 4) {
        fprintf(stderr,"unique.c[unique] -- Found next %d at %lu.\n", *((int*)(v_next)), (unsigned long) v_next);
      } else if (width == 8) {
        fprintf(stderr,"unique.c[unique] -- Found next %ld at %lu.\n", *((long*)(v_next)), (unsigned long) v_next);
      } else {
        fprintf(stderr,"unique.c[unique] -- Found next at %lu.\n", (unsigned long) v_next);
      }
      #endif
      // Increment the subgroup that we are pulling the next element from.
      chunk_indices[i_next]++;
      v_prev = v_next;
      // Store the unique value and increment the number of unique values found.
      (*sorted_unique)[*num_unique] = v_next;
      (*num_unique)++;
    }
  } // END while (checked > 0)
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"unique.c[unique] -- Freeing chunk_indices at %lu and chunk_ends at %lu.\n",
          (unsigned long) chunk_indices, (unsigned long) chunk_ends);
  #endif
  free(chunk_indices); // DEALLOCATION: chunk indices
  free(chunk_ends); // DEALLOCATION: chunk ends
  // Reallocate the space for all of the unique words (shrinking the array unless all were unique).
  if ((*num_unique) < num_elements) {
    char** old_unique = (*sorted_unique);
    (*sorted_unique) = (char**) malloc((*num_unique) * sizeof(char*)); // ALLOCATION: sorted unique pointers
    #ifdef _UNIQUE_DEBUG_ENABLED
    fprintf(stderr,"unique.c[unique] -- Allocating smaller sorted unique with size %ld and moving from %lu to %lu\n",
            (*num_unique), (unsigned long) old_unique, (unsigned long)(*sorted_unique));
    #endif
    for (long i=0; i<(*num_unique); i++) {
      (*sorted_unique)[i] = old_unique[i];
    }
    free(old_unique); // DEALLOCATION: tentative unique words holder
  }
  // Count the total number of characters (including null terminators).
  long total_chars = 0;
  if (width == -1) {
    for (long i=0; i<(*num_unique); i++) {
      total_chars += strlen((*sorted_unique)[i]) + 1;
    }
  } else {
    total_chars = width * (*num_unique);
  }
  // Allocate space for all words to be packed into one contiguous block of memory.
  char*restrict unique_words = (char*) malloc(total_chars * sizeof(char)); // ALLOCATION: packed unique words
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"unique.c[unique] -- Allocated space for %ld bytes at [%lu,%lu].\n",
          total_chars, (unsigned long) unique_words,
          ((unsigned long) unique_words)+(total_chars * sizeof(char)));
  #endif
  // Copy the string for each word into the contiguous block (and free old memory).
  long j = 0;
  for (long i=0; i<(*num_unique); i++) {
    char* old_word = (*sorted_unique)[i];
    char* new_word = &unique_words[j];
    if (width == -1) {
      strcpy(new_word, old_word);
    } else {
      for (int wi=0; wi<width; wi++) {
        new_word[wi] = old_word[wi];
      }
    }
    #ifdef _UNIQUE_DEBUG_ENABLED
    if (width == -1) {
      fprintf(stderr,"unique.c[unique] -- Moved '%s' at %lu to '%s' at %lu.\n",
              old_word, (unsigned long) old_word, new_word, (unsigned long) new_word);
    } else if (width == 1) {
      fprintf(stderr,"unique.c[unique] -- Moved %u at %lu to %u at %lu.\n",
              *((unsigned char*)(old_word)), (unsigned long) old_word, *((unsigned char*)(new_word)), (unsigned long) new_word);
    } else if (width == 4) {
      fprintf(stderr,"unique.c[unique] -- Moved %d at %lu to %d at %lu.\n",
              *((int*)(old_word)), (unsigned long) old_word, *((int*)(new_word)), (unsigned long) new_word);
    } else if (width == 8) {
      fprintf(stderr,"unique.c[unique] -- Moved %ld at %lu to %ld at %lu.\n",
              *((long*)(old_word)), (unsigned long) old_word, *((long*)(new_word)), (unsigned long) new_word);
    } else {
      fprintf(stderr,"unique.c[unique] -- Moved value at %lu to %lu.\n", (unsigned long) old_word, (unsigned long) new_word);
    }
    #endif
    (*sorted_unique)[i] = new_word;
    // Increment the starting position based on the size of the element.
    if (width == -1) {
      j += strlen(new_word) + 1;
    } else {
      j += width;
    }
  }
  // Deallocate temporary space for "array" if it was created.
  if (width != -1) {
    free(array); // DEALLOCATION: pointer variant of (flat) array_in
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"unique.c[unique] -- Final sorted %ld unique elements stored at %lu with array of pointers at %lu.\n",
          (*num_unique), (unsigned long) (**sorted_unique), (unsigned long) (*sorted_unique));
  fprintf(stderr,"<------------- unique ===============>\n");
  #endif
}

// -------------------------------------------------------------------------------------------------
// This function can be used to free the memory for the sorted unique words from python.
void free_unique(char***restrict sorted_unique, long*restrict num_unique, const int*restrict width) {
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"=============> free_unique <---------------\n");
  fprintf(stderr,"unique.c[free_unique] -- %ld word pointers (width of %d) are at %lu.\n", *num_unique, *width, (unsigned long)(*sorted_unique));
  for (long i=0; i<(*num_unique); i++) {
    if (*width == -1) {
      fprintf(stderr,"  unique.c[free_unique] -- Word %ld = '%s' at %lu.\n", i+1,
              (*sorted_unique)[i], (unsigned long)((*sorted_unique)[i]));
    } else if (*width == 1) {
      fprintf(stderr,"  unique.c[free_unique] -- Word %ld = %u at %lu.\n", i+1,
              *((unsigned char*)((*sorted_unique)[i])), (unsigned long)((*sorted_unique)[i]));
    } else if (*width == 4) {
      fprintf(stderr,"  unique.c[free_unique] -- Word %ld = %d at %lu.\n", i+1,
              *((int*)((*sorted_unique)[i])), (unsigned long)((*sorted_unique)[i]));
    } else if (*width == 8) {
      fprintf(stderr,"  unique.c[free_unique] -- Word %ld = %ld at %lu.\n", i+1,
              *((long*)((*sorted_unique)[i])), (unsigned long)((*sorted_unique)[i]));
    } else {
      fprintf(stderr,"  unique.c[free_unique] -- Word %ld at %lu.\n", i+1, (unsigned long)((*sorted_unique)[i]));
    }
  }
  fprintf(stderr,"unique.c[free_unique] -- Freeing all words at %lu.\n", (unsigned long)((*sorted_unique)[0]));
  #endif
  // Free all of the unique words (they were allocated as one big block).
  if ((*sorted_unique)[0] != NULL) {
    free((*sorted_unique)[0]); // DEALLOCATION: packed unique words
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"unique.c[free_unique] -- Freeing pointers to %ld words from %lu\n", *num_unique, (unsigned long)(*sorted_unique));
  #endif
  // Free the array of pointers to the words.
  if (sorted_unique != NULL) {
    free((*sorted_unique)); // DEALLOCATION: sorted unique pointers
  }
  // Set the total count to zero.
  (*num_unique) = 0;
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"<------------- free_unique ===============>\n");
  #endif
}

// -------------------------------------------------------------------------------------------------
// Given an array of Python objects and , generate a sorted list of unique elements.
void to_int(const int width, const long n, char**restrict array_in,
            const long num_unique, char**restrict sorted_unique,
            long*restrict int_arr) {
  #ifdef _UNIQUE_DEBUG_ENABLED  
  fprintf(stderr,"=============> to_int <---------------\n");
  #endif
  // Check for memory overlap between 'array_in' and 'sorted_unique'.
  if (arrays_overlap(width, n, array_in, num_unique, sorted_unique)) {
    fprintf(stderr,"ERROR unique.c[to_int] -- Overlap in provided memory locations for array_in[%lu:%lu] and provided sorted_unique[%lu:%lu].\n",
            (unsigned long) array_in, ((unsigned long)(&(array_in[n]))) + (width-1),
            (unsigned long) sorted_unique, ((unsigned long)(&(sorted_unique[num_unique]))) + (width-1));
  }
  // If the byte-size width is positive, then we need to allocate and fill an
  //  array of pointers assuming the provided pointer is to contiguous memory.
  char**restrict array;
  if (width == -1) {
    array = array_in;
  } else {
    array = (char**) malloc(n * sizeof(char*)); // ALLOCATION: pointer variant of (flat) array_in
    // Check for memory overlap between 'array' and the provided 'sorted_unique'.
    while (arrays_overlap(width, num_unique, sorted_unique, n, array)) {
      fprintf(stderr,"ERROR unique.c[to_int] -- Overlap in allocated memory locations for array[%lu:%lu] and provided sorted_unique[%lu:%lu].\n",
              (unsigned long) array, ((unsigned long)(&(array[n]))) + (width-1), 
              (unsigned long) sorted_unique, ((unsigned long)(&(sorted_unique[num_unique]))) + (width-1));
      char** old_array = array;
      array = (char**) malloc(n * sizeof(char*)); // ALLOCATION: pointer variant of (flat) array_in
      free(old_array); // DEALLOCATION: pointer variant of (flat) array_in
    }
    // Check for memory overlap between 'array' and the provided 'array_in'.
    while (arrays_overlap(width, n, array_in, n, array)) {
      fprintf(stderr,"ERROR unique.c[to_int] -- Overlap in allocated memory locations for array[%lu:%lu] and provided array_in[%lu:%lu].\n",
              (unsigned long) array, ((unsigned long)(&(array[n]))) + (width-1), 
              (unsigned long) array_in, ((unsigned long)(&(array_in[n]))) + (width-1));
      // This should NEVER happen if 'malloc' is working correctly, but since it DOES happen
      //  occasionally in testing (macOS 13.2, python 3.11.2, numpy 1.24.2, gfortran 12.2.0),
      //  we will allocate a new array until we create one that doesn't overlap.
      char** old_array = array;
      array = (char**) malloc(n * sizeof(char*)); // ALLOCATION: pointer variant of (flat) array_in
      free(old_array); // DEALLOCATION: pointer variant of (flat) array_in
    }
    #ifdef _UNIQUE_DEBUG_ENABLED
    fprintf(stderr," unique.c[to_int] -- Allocated pointer array[%lu:%lu]\n",
            (unsigned long) array, ((unsigned long) array) + (n*sizeof(char*)));
    #endif
    for (long i=0; i<n; i++) {
      array[i] = &(((char*)array_in)[i*width]);
      #ifdef _UNIQUE_DEBUG_ENABLED
      if ((i <= _UNIQUE_HALF_MAX_EXAMPLES) || (i >= n-_UNIQUE_HALF_MAX_EXAMPLES)) {
        if (width == -1) {
          fprintf(stderr," unique.c[to_int] -- array[%ld] = '%s' is at %lu\n", i, array[i], (unsigned long)(array[i]));
        } else if (width == 1) {
          fprintf(stderr," unique.c[to_int] -- array[%ld] = %u is at %lu\n", i, *((unsigned char*)(array[i])), (unsigned long)(array[i]));
        } else if (width == 4) {
          fprintf(stderr," unique.c[to_int] -- array[%ld] = %d is at %lu\n", i, *((int*)(array[i])), (unsigned long)(array[i]));
        } else if (width == 8) {
          fprintf(stderr," unique.c[to_int] -- array[%ld] = %ld is at %lu\n", i, *((long*)(array[i])), (unsigned long)(array[i]));
        } else {
          fprintf(stderr," unique.c[to_int] -- array[%ld] is at %lu\n", i, (unsigned long)(array[i]));
        }
      }
      #endif
    }
  }
  // Set the number of parallel threads to use.
  int nt = omp_get_max_threads();
  if (n < nt) nt = 1;
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"unique.c[to_int] -- given Array[0:%ld] at %lu and Unique[0:%ld] at %lu (%d thread%s\n",
          n, (unsigned long) array, num_unique, (unsigned long) sorted_unique, nt, (nt > 1 ? "s)" : ")"));
  #endif
  // Set the byte comparison function based on the element width.
  int (*byte_compare)(const void*, const void*);
  if (width == -1) {
    byte_compare = &str_compare;
  } else if (width == 1) {
    byte_compare = &char_compare;
  } else if (width == 4) {
    byte_compare = &int_compare;
  } else if (width == 8) {
    byte_compare = &long_compare;
  } else {
    #ifdef _UNIQUE_ALLOW_LOCAL_FUNCTIONS
    // Compare two fixed size arrays, treat as unsigned integers.
    int custom_compare(const void* a_in, const void* b_in) {
      const unsigned char* a = *(const unsigned char**) a_in;
      const unsigned char* b = *(const unsigned char**) b_in;
      int comparison_result = 0;
      // These strings have equal length, compare their characters.
      for (int i = 0; i < width; i++) {
        if (a[i] < b[i]) {
          comparison_result = -1;
          break;
        } else if (a[i] > b[i]) {
          comparison_result = 1;
          break;
        }
      }
      #ifdef _UNIQUE_DEBUG_ENABLED
      fprintf(stderr,"  unique.c[custom_cmp] -- %lu <= %lu  =  %d\n", a, b, comparison_result);
      #endif
      return comparison_result;
    }
    byte_compare = &custom_compare;
    #else
    fprintf(stderr,"ERROR unique.c[unique] There is no known comparison function for elements of %d bytes.\n", width);
    for (long i=0; i<n; i++) {
      int_arr[i] = -1;
    }
    return;
    #endif
  }
  // Parallel loop for mapping elements to integers.
  #pragma omp parallel for num_threads(nt) if(nt > 1)
  for (long i=0; i<n; i++) {
    // Get the local copy of the specific element.
    char* value = array[i];
    #ifdef _UNIQUE_DEBUG_ENABLED
    if ((i <= _UNIQUE_HALF_MAX_EXAMPLES) || (i >= n-_UNIQUE_HALF_MAX_EXAMPLES)) {
      if (width == -1) {
        fprintf(stderr,"\nunique.c[to_int] -- array[%ld] = '%s' at %lu\n", i, value, (unsigned long) value);
      } else if (width == 1) {
        fprintf(stderr,"\nunique.c[to_int] -- array[%ld] = %u at %lu\n", i, *((unsigned char*)(value)), (unsigned long) value);
      } else if (width == 4) {
        fprintf(stderr,"\nunique.c[to_int] -- array[%ld] = %d at %lu\n", i, *((int*)(value)), (unsigned long) value);
      } else if (width == 8) {
        fprintf(stderr,"\nunique.c[to_int] -- array[%ld] = %ld at %lu\n", i, *((long*)(value)), (unsigned long) value);
      } else {
        fprintf(stderr,"\nunique.c[to_int] -- array[%ld] at %lu\n", i, (unsigned long) value);
      }
    }
    #endif
    // Find the index of that value in the sorted list.
    long low = 0;
    long high = num_unique;
    long index = (low + high) / 2;
    int comparison = 0;
    while (high != low) {
      #ifdef _UNIQUE_DEBUG_ENABLED
      if ((i <= _UNIQUE_HALF_MAX_EXAMPLES) || (i >= n-_UNIQUE_HALF_MAX_EXAMPLES)) {
        fprintf(stderr," unique.c[to_int] (low %ld) (high %ld) (index %ld)\n", low, high, index);
        if (width == -1) {
          fprintf(stderr," unique.c[to_int] comparing with '%s' at %lu\n", sorted_unique[index], (unsigned long)(sorted_unique[index]));
        } else if (width == 1) {
          fprintf(stderr," unique.c[to_int] comparing with %u at %lu\n", *((unsigned char*)(sorted_unique[index])), (unsigned long)(sorted_unique[index]));
        } else if (width == 4) {
          fprintf(stderr," unique.c[to_int] comparing with %d at %lu\n", *((int*)(sorted_unique[index])), (unsigned long)(sorted_unique[index]));
        } else if (width == 8) {
          fprintf(stderr," unique.c[to_int] comparing with %ld at %lu\n", *((long*)(sorted_unique[index])), (unsigned long)(sorted_unique[index]));
        } else {
          fprintf(stderr," unique.c[to_int] comparing with value at %lu\n", (unsigned long)(sorted_unique[index]));
        }
      }
      #endif
      comparison = byte_compare(&sorted_unique[index], &value);
      if (comparison == 0) { break; } 
      else if (comparison > 0)    { high = index; }
      else /* (comparison < 0) */ { low = index + 1; }
      index = (low + high) / 2;
    }
    // Convert the final comparison result into a final index.
    if (comparison != 0) {
      index = 0;
    } else {
      index++;
    }
    // Place the (1's) index into the output integer array.
    int_arr[i] = index;
    #ifdef _UNIQUE_DEBUG_ENABLED
    if ((i <= _UNIQUE_HALF_MAX_EXAMPLES) || (i >= n-_UNIQUE_HALF_MAX_EXAMPLES)) {
      if (width == -1) {
        fprintf(stderr," unique.c[to_int] '%s' assigned %ld\n", value, index);
      } else if (width == 1) {
        fprintf(stderr," unique.c[to_int] %u assigned %ld\n", *((unsigned char*)(value)), index);
      } else if (width == 4) {
        fprintf(stderr," unique.c[to_int] %d assigned %ld\n", *((int*)(value)), index);
      } else if (width == 8) {
        fprintf(stderr," unique.c[to_int] %ld assigned %ld\n", *((long*)(value)), index);
      } else {
        fprintf(stderr," unique.c[to_int] value at %lu assigned %ld\n", (unsigned long) value, index);
      }
    }
    #endif 
  }
  // Deallocate space for pointers if that was created.
  if (width != -1) {
    free(array); // DEALLOCATION: pointer variant of (flat) array_in
  }
  #ifdef _UNIQUE_DEBUG_ENABLED
  fprintf(stderr,"<------------- to_int ===============>\n");
  #endif
}
