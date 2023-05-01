/*

This C extension to Python is designed to quick identify unique elements
in an array and then map subsequent (and similar) arrays to integer format
where each unique value is replaced by a unique integer in the range
[0, n) where 'n' is the number of unique values in total.

Where possible, it utilizes OpenMP shared-memory parallelism to accelerate
the loops of similar actions. The main barrier to parallelism is the
Global Interpreter Lock (GIL), which is required for all meaningful PyObject
actions (like converting any object to a string).

*/

// As per recommended on https://docs.python.org/3/c-api/intro.html#include-files we
//  are including Python.h FIRST, before standard libraries, and defining PY_SSIZE_T_CLEAN.
#define PY_SSIZE_T_CLEAN
#include <Python.h>  // PyObject, Py_None, PyUnicode_Check, PyUnicode_AsUTF8, PyObject_str, Py_DECREF

// Standard libraries.
#include <stdio.h>  // printf
#include <stdlib.h>  // malloc
#include <string.h>  // strdup

// Parallelism.
#include <omp.h> // omp_get_thread_num, omp_get_num_threads, omp_get_max_threads */

// Convert a Python object into a string.
//  WARNING: Result is ALLOCATED, must be FREED.
char* to_str(PyObject* obj) {
  char* cstr = NULL;
  // Handle C null objects.
  if (!obj) {
    cstr = strdup("NULL");
  }
  // Handle None objects.
  else if (obj == Py_None) {
    cstr = strdup("None");
  }
  // Check if the object is already a string.
  else if (PyUnicode_Check(obj)) {
    // Get the UTF8 character sequence from the string.
    const char* str = PyUnicode_AsUTF8(obj);
    if (!str) {
      cstr = strdup("NULL.UTF8");
    } else {
      cstr = strdup(str);
    }
  }
  // Otherwise, the object is not already a string, we must cast it.
  else {
    // The following section must be single-threaded to be GIL-safe.
    #pragma omp critical
    {
      // Ensure the GIL is obtained before converting an object to a string.
      PyGILState_STATE state = PyGILState_Ensure();
      PyObject* str_obj = PyObject_Str(obj);
      // Check for any python errors.
      if (PyErr_Occurred()) {
        PyErr_Print();
        cstr = strdup("ERROR: C Python exception encountered on converting object to string.");
      }
      // Check for an invalid string object (some serious error probably occurred).
      else if (!str_obj) {
        cstr = strdup("NULL.str");
      }
      // Get the UTF8 character sequence from the string form of the object.
      else {
        const char* str = PyUnicode_AsUTF8(str_obj);
        if (!str) {
          cstr = strdup("NULL.str.UTF8");
        } else {
          cstr = strdup(str);
        }
        Py_DECREF(str_obj);
      }
      // Release the GIL.
      PyGILState_Release(state);
    }
  }

  // Return the char* string.
  return cstr;
}


// Compare two strings.
int string_compare_function(const char* a, const char* b) {
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
  return comparison_result;
}


// Create a function for comparing strings that are stored as void pointers.
int void_compare_function(const void* a, const void* b) {
  const char* a_str = *(const char**)a;
  const char* b_str = *(const char**)b;
  return string_compare_function(a_str, b_str);
}


// Given an array of Python objects, generate a sorted list of unique strings.
void unique(const long n, PyObject **arr_obj, long * num_unique, char *** sorted_unique) {
  // Set the number of parallel threads to use.
  int nt = omp_get_max_threads();
  if (n < nt) nt = 1;
  // Allocate space for all python object strings.
  char **strings = malloc(n * sizeof(char*));

  // Convert all of the objects to strings (as parallel as possible, with Global Interpreter Lock).
  #pragma omp parallel for num_threads(nt) if(nt > 1)
  for (long i=0; i<n; i++) {
    strings[i] = to_str(arr_obj[i]);
  }

  // Generate the start and end indices of chunks of the array for parallel sorting.
  long * chunk_indices = malloc(nt * sizeof(long));
  long * chunk_ends = malloc(nt * sizeof(long));
  long chunk_size = n / nt;
  chunk_indices[0] = 0;
  for (int i=0; i < (n%nt); i++) {
    chunk_ends[i] = chunk_indices[i] + chunk_size + 1;
    if (i+1 < nt) {
      chunk_indices[i+1] = chunk_ends[i];
    }
  }
  for (int i=(n%nt); i < nt; i++) {
    chunk_ends[i] = chunk_indices[i] + chunk_size;
    if (i+1 < nt) {
      chunk_indices[i+1] = chunk_ends[i];
    }
  }

  // Sort chunks of the array (in parallel).
  #pragma omp parallel num_threads(nt) if(nt > 1)
  {
    int tid = omp_get_thread_num();  // [0, .., nt-1].
    long start = chunk_indices[tid]; // Index of first element.
    long end = chunk_ends[tid];      // EXCLUSIVE, end = last element + 1.
    qsort(&strings[start], end - start, sizeof(char*), void_compare_function);
  }

  // Initialize space and variables for computing unique strings.
  (*sorted_unique) = malloc(n * sizeof(char*));
  (*num_unique) = 0;
  char * c_prev = NULL;
  char * c_next = "init";
  int i_next = -1;

  // Loop and collect all the unique strings.
  while (c_next != NULL) {
    c_next = NULL;
    i_next = -1;

    // If there was a previous value added, then rotate all chunks until getting
    //  to a "new" value in each of the chunks arrays (or exhausting the array).
    if (c_prev != NULL) {
      for (int j = 0; j < nt; j++) {
        // Rotate to the next string in this chunk until we find a new value.
        while ((chunk_indices[j] < chunk_ends[j])
               && (string_compare_function(c_prev, strings[chunk_indices[j]]) == 0)) {
          chunk_indices[j]++;
        }
      }
    }

    // Find the "least" string from each of the chunks to add next.
    for (int j = 0; j < nt; j++) {
      // If there is another value in this chunk to consider, then..
      if (chunk_indices[j] < chunk_ends[j]) {
        // If there is not a previous candidate.
        if (c_next == NULL) {
          c_next = strings[chunk_indices[j]];
          i_next = j;
        } else {
          char * alternative = strings[chunk_indices[j]];
          int comparison = string_compare_function(alternative, c_next);
          // If the new string is lesser, then store it as the next candidate.
          if (comparison < 0) {
            c_next = alternative;
            i_next = j;
          }
          // If the string was equal to the current candidate, iterate that chunk (duplicate).
          else if (comparison == 0) {
            chunk_indices[j]++;
          }
          // Otherwise, the string was greater, so it's not getting added this time.
        } // END if (c_next == NULL)
      } // END if (chunk_indices[j] < chunk_ends[j])
    } // END for (int j = 0; j < nt; n++) {

    // Increment the subgroup that we are pulling the next element from.
    chunk_indices[i_next]++;
    // Store the next unique value and increment our index.
    if (c_next != NULL) {
      c_prev = c_next;
      (*sorted_unique)[*num_unique] = c_next;
      (*num_unique)++;
    }

  } // END while (checked > 0)

  free(chunk_indices);
  free(chunk_ends);

  // Reallocate the space for all of the unique words (shrinking the array unless all were unique).
  if ((*num_unique) < n) {
    char **old_unique = (*sorted_unique);
    (*sorted_unique) = malloc((*num_unique) * sizeof(char*));
    for (long i=0; i<(*num_unique); i++) {
      (*sorted_unique)[i] = old_unique[i];
    }
    free(old_unique);
  }
  // Count the total number of characters (including null terminators).
  long total_chars = 0;
  for (long i=0; i<(*num_unique); i++) {
    total_chars += strlen((*sorted_unique)[i]) + 1;
  }
  // Allocate space for all words to be packed into one contiguous block of memory.
  char* unique_words = malloc(total_chars * sizeof(char));
  // Copy the string for each word into the contiguous block (and free old memory).
  long j = 0;
  for (long i=0; i<(*num_unique); i++) {
    char* old_word = (*sorted_unique)[i];
    char* new_word = &unique_words[j];
    strcpy(new_word, old_word);
    (*sorted_unique)[i] = new_word;
    j += strlen(new_word) + 1;
  }
  // Free allocated words and top-level array that was allocated to hold the strings.
  for (long i=0; i<n; i++) {
    // Free the memory for this (duplicate) string.
    if (strings[i] != NULL) {
      free(strings[i]);
    }
  }
  free(strings);
}


// Given an array of Python objects and , generate a sorted list of unique strings.
void to_int(const long n, PyObject **obj_arr,
            const long num_unique, char ***sorted_unique,
            long *int_arr) {
  
  // Set the number of parallel threads to use.
  int nt = omp_get_max_threads();
  if (n < nt) nt = 1;

  // Convert all of the objects to strings (as parallel as possible, with Global Interpreter Lock).
  #pragma omp parallel for num_threads(nt) if(nt > 1)
  for (long i=0; i<n; i++) {
    // Get the string version of the specific element.
    char *str = to_str(obj_arr[i]);
    // Find the index of that string in the sorted list.
    long low = 0;
    long high = num_unique;
    long index = (low + high) / 2;
    int comparison = 0;
    while (high != low) {
      comparison = string_compare_function((*sorted_unique)[index], str);
      if (comparison == 0) { break; } 
      else if (comparison > 0)    { high = index; }
      else /* (comparison < 0) */ { low = index + 1; }
      index = (low + high) / 2;
    }
    if (comparison != 0) {
      comparison = string_compare_function((*sorted_unique)[index], str);
    }
    // Free the allocated string.
    free(str);
    // Convert the final comparison result into a final index.
    if (comparison != 0) {
      index = 0;
    } else {
      index++;
    }
    // Place the (1's) index into the output integer array.
    int_arr[i] = index;
  }
}


// This function can be used to free the memory for the sorted unique words from python.
void free_unique(long * num_unique, char *** sorted_unique) {
  // Free all of the unique words (they were allocated as one big block).
  if (sorted_unique[0] != NULL) {
    free(sorted_unique[0]);
  }

  // Free the array of pointers to the words.
  if (sorted_unique != NULL) {
    free(sorted_unique);
  }

  // Set the total count to zero.
  (*num_unique) = 0;
}
