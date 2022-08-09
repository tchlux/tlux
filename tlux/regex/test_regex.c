// cc -o test_regex test_regex.c && ./test_regex * ; rm -f ./test_regex

#define DEBUG

#ifdef DEBUG
#include <stdio.h>  // EOF
// Define a global flag for determining if interior prints should be done.
int DO_PRINT = 0;
// Define a global character array for safely printing escaped characters.
char CHAR3[3];
// Define a function for safely printing a character (escapes whitespace).
char* SAFE_CHAR(const char c) {
  CHAR3[0] = '\\';
  CHAR3[1] = '\0';
  CHAR3[2] = '\0';
  if (c == '\n') CHAR3[1] = 'n';
  else if (c == '\t') CHAR3[1] = 't';
  else if (c == '\r') CHAR3[1] = 'r';
  else if (c == '\0') CHAR3[1] = '0';
  else if (c == EOF)  CHAR3[1] = 'X';
  else CHAR3[0] = c;
  return (char *) CHAR3;
}
#endif

// Include the source code for regex here in the tests.
#include "regex.c"


// ===================================================================
//                  BEGIN   T E S T I N G   CODE
// ===================================================================


#ifdef DEBUG
int run_tests(); // <- actually declared later
// For testing purposes.
int main(int argc, char * argv[]) {
  // =================================================================
  // Manual test of `match`.. (use "if (1)" to run, "if (0)" to skip)
  if (0) {
    // ------------------------------------------
    // char * regex = "((\r\n)|\r|\n)";
    // char * string = "\r\n**** \n";
    // ------------------------------------------
    // char * regex = "$(({\n}\n?)|(\n?{\n}))*$";
    // char * string = "$\n  testing \n$";
    // ------------------------------------------
    DO_PRINT = 1;
    char * regex = ".*st{.}";
    char * string = "| test";
    int start, end;
    match(regex, string, &start, &end);
    printf("==================================================\n\n");
    // Handle errors.
    if (start < 0) {
      if (end < 0) {
        printf("\nERROR: invalid regular expression, code %d", -end);
        if (start < -1) {
          printf(" error at position %d.\n", -start-1);
          printf("  %s\n", regex);
          printf("  %*c\n", -start, '^');
        } else {
          printf(".\n");
        }
        // Mark the failure in the match with return code.
        return (1);
        // No matches found in the search string.
      } else {
        printf("no match found\n");
      }
      // Matched.
    } else {
      printf("match at (%d -> %d)\n", start, end);
    }
    // Print out the matched expression.
    if (start >= 0) {
      printf("\n\"");
      for (int j = start; j < end; j++)
        printf("%c",string[j]);
      printf("\"\n");
    }
    return 0;
    // =================================================================
    // Manual test of `fmatcha`.. (use "if (1)" to run, "if (0)" to skip)
  } else if (0) {
    DO_PRINT = 1;
    char * regex = ".*hehe";
    // char * path = "regex.so";
    char * path = "test.txt";
    int n_matches;
    int * starts;
    int * ends;
    int * lines;
    float min_ascii_ratio = 0.5;
    fmatcha(regex, path, &n_matches, &starts, &ends, &lines, min_ascii_ratio);
    printf("==================================================\n\n");
    // Handle errors.
    if (n_matches == -3) {
      printf("\nERROR: too many non-ASCII characters in file");
      return(3);
    } else if (n_matches == -2) {
      printf("\nERROR: failed to load file");
      return(2);
    } else if (n_matches == -1) {
      if (starts[0] > 0) {
        if (ends[0] > 0) {
          printf("\nERROR: invalid regular expression, code %d", ends[0]);
          if ((long) starts[0] > 1) {
            printf(" error at position %d.\n", starts[0]-1);
            printf("  ");
            for (int i = 0; regex[i] != '\0'; i++)
              printf("%s", SAFE_CHAR(regex[i]));
            printf("\n");
            printf("  %*c\n", starts[0], '^');
          } else {
            printf(".\n");
          }
          // Mark the failure in the match with return code.
          return (1);
          // No matches found in the search string.
        } else {
          printf("ERROR: unexpected execution flow, (n_matches = -1) and match at (%d -> %d)\n", starts[0], ends[0]);
          return(4);
        }
      }
      printf("ERROR: unexpected execution flow, (n_matches = -1) and match at (%ld -> %ld)\n", (long) starts, (long) ends);
      return(4);
    } else {
      // Print out the matched expression.
      printf("\n");
      for (int i = 0; i < n_matches; i++) {
        printf("  %d -> %d\n", starts[i], ends[i]);
      }
      return 0;
    }
    // =================================================================
  } else {
    return(run_tests());
  }
}

int run_tests() {
  // test data array
  char * regexes[] = {
    // Invalid regular expressions.
    "*abc",
    "?abc",
    "|abc",
    ")abc",
    "}abc",
    "]abc",
    "abc|",
    "abc|*",
    "abc|?",
    "abc|)",
    "abc|]",
    "abc|}",
    "abc**",
    "abc*?",
    "abc?*",
    "abc??",
    "abc(*",
    "abc(?",
    "abc{*",
    "abc{?",
    "abc(",
    "abc{",
    "abc()",
    "abc{}",
    "abc[]",
    // Valid regular expressions.
    ".",
    ".*",
    "..",
    " (.|.)*d",
    ".* .*ad",
    "abc",
    ".*abc",
    ".((a*)|(b*))*.",
    "(abc)",
    "[abc]",
    "{abc}",
    "{[abc]}",
    "{{[abc]}}",
    "[ab][ab]",
    "{[ab][ab]}",
    "a*bc",
    "(ab)*c",
    "[ab]*c",
    "{ab}*c",
    "[a][b]*{[c]}",
    "{{a}[bcd]}",
    "a{[bcd]}e",
    "{{a}[bcd]{e}}",
    "(a(bc)?)*(d)",
    "(a(bc*)?)|d",
    "{a(bc*)?}|d",
    "{(a(bc*)?)}|d",
    "(a(bc)?)|(de)",
    "(a(z.)*)[bc]*d*",
    "(a(z.)*)[bc]*d*{e}f?g",
    "(a(z.)*)[bc]*d*{e}f?g|h",
    "({({ab}c?)*d}|(e(fg)?))",
    "({({[ab]}c?)*d}|(e(fg)?))",
    "({(a)({[bc]}d?e)*(f)}|g(hi)?)",
    "[*][*]*{[*]}",
    "[[][[]",
    ".*[)][)]",
    ".*end{.}",
    "[|]",
    // Last test regular expression must be empty!
    ""
  };

  // test data array
  int true_n_tokens[] = {
    // Invalid regular expressions.
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -4,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    // Valid regular expressions    
    1,
    2,
    2,
    6,
    7,
    3,
    5,
    8,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    6,
    7,
    7,
    7,
    7,
    9,
    13,
    15,
    11,
    11,
    13,
    4,
    2,
    4,
    6,
    1,
    // Last test regular expression must be empty!
    0
  };

  // test data array
  int true_n_groups[] = {
    // Invalid regular expressions.
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_UNCLOSED_GROUP_ERROR,
    REGEX_UNCLOSED_GROUP_ERROR,
    REGEX_EMPTY_GROUP_ERROR,
    REGEX_EMPTY_GROUP_ERROR,
    REGEX_EMPTY_GROUP_ERROR,
    // Valid regular expressions    
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    3,
    1,
    1,
    1,
    2,
    3,
    2,
    3,
    0,
    1,
    1,
    1,
    4,
    3,
    2,
    4,
    3,
    2,
    2,
    3,
    3,
    3,
    4,
    4,
    6,
    7,
    8,
    4,
    2,
    2,
    1,
    1,
    // Last test regular expression must be empty!
    0
  };

  // test data array
  char * true_tokens[] = {
    // Invalid regular expressions.
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    // Valid regular expressions.
    ".",
    "*.",
    "..",
    " *|..d",
    "*. *.ad",
    "abc",
    "*.abc",
    ".*|*a*b.",
    "abc",
    "abc",
    "abc",
    "abc",
    "abc",
    "abab",
    "abab",
    "*abc",
    "*abc",
    "*abc",
    "*abc",
    "a*bc",
    "abcd",
    "abcde",
    "abcde",
    "*a?bcd",
    "|a?b*cd",
    "|a?b*cd",
    "|a?b*cd",
    "|a?bcde",
    "a*z.*bc*d",
    "a*z.*bc*de?fg",
    "a*z.*bc*de?f|gh",
    "|*ab?cde?fg",
    "|*ab?cde?fg",
    "|a*bc?defg?hi",
    "****",
    "[[",
    "*.))",
    "*.end.",
    "|",
    // Last test regular expression must be empty!
    ""
  };

  // test data array
  int true_jumps[] = {
    // Invalid regular expressions.
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // Valid regular expressions.
    1,
    1,0,
    1,2,
    1,2,3,1,1,6,
    1,0,3,4,3,6,7,
    1,2,3,
    1,0,3,4,5,
    1,2,3,4,3,6,5,8,
    1,2,3,
    3,3,3,
    -1,-1,-1,
    -1,-1,-1,
    3,3,3,
    2,2,4,4,
    -1,-1,-1,-1,
    1,0,3,4,
    1,2,0,4,
    1,0,0,4,
    1,-1,-1,4,
    1,2,1,-1,
    1,-1,-1,-1,
    1,-1,-1,-1,5,
    1,-1,-1,-1,5,
    1,2,3,4,0,6,
    1,2,3,4,5,4,7,
    1,-1,3,-1,5,-1,7,
    1,-1,3,-1,5,-1,7,
    1,2,3,4,7,6,7,
    1,2,3,1,5,4,4,8,7,
    1,2,3,1,5,4,4,8,7,-1,11,12,13,
    1,2,3,1,5,4,4,8,7,-1,11,12,13,15,15,
    1,2,3,4,5,-1,-1,8,9,10,11,
    1,2,4,4,5,-1,-1,8,9,10,11,
    1,-1,3,5,5,6,-1,-1,-1,10,11,12,13,
    1,2,1,-1,
    1,2,
    1,0,3,4,
    1,0,3,4,5,-1,
    1,
    // Last test regular expression must be empty!
    // {}
  };

  // test data array
  int true_jumpf[] = {
    // Invalid regular expressions.
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // Valid regular expressions.
    -1,
    2,-1,
    -1,-1,
    -1,5,4,-1,-1,-1,
    2,-1,-1,5,-1,-1,-1,
    -1,-1,-1,
    2,-1,-1,-1,-1,
    -1,7,5,7,-1,1,-1,-1,
    -1,-1,-1,
    1,2,-1,
    1,2,3,
    1,2,3,
    1,2,-1,
    1,-1,3,-1,
    1,2,3,4,
    2,-1,-1,-1,
    3,-1,-1,-1,
    3,2,-1,-1,
    3,2,0,-1,
    -1,3,-1,4,
    -1,2,3,4,
    -1,2,3,4,-1,
    -1,2,3,4,-1,
    5,-1,0,-1,-1,-1,
    6,-1,7,-1,7,-1,-1,
    6,2,7,4,7,4,-1,
    6,2,7,4,7,4,-1,
    5,-1,7,-1,-1,-1,-1,
    -1,4,-1,-1,7,6,-1,9,-1,
    -1,4,-1,-1,7,6,-1,9,-1,10,12,-1,-1,
    -1,4,-1,-1,7,6,-1,9,-1,10,12,-1,14,-1,-1,
    7,6,-1,-1,1,1,11,-1,11,-1,-1,
    7,6,3,-1,1,1,11,-1,11,-1,-1,
    9,2,8,4,-1,7,7,2,10,-1,13,-1,-1,
    -1,3,-1,4,
    -1,-1,
    2,-1,-1,-1,
    2,-1,-1,-1,-1,6,
    -1,
    // Last test regular expression must be empty!
    // {}
  };

  char true_jumpi[] = {
    // Invalid regular expressions.
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // 
    // Valid regular expressions.
    0,
    0,0,
    0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,
    0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,
    1,1,2,
    0,0,0,
    1,1,2,
    1,1,2,
    1,2,1,2,
    1,2,1,2,
    0,0,0,0,
    0,0,0,0,
    0,1,2,0,
    0,0,0,0,
    2,0,2,2,
    0,1,1,2,
    0,1,1,2,0,
    0,1,1,2,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,1,2,0,0,
    0,0,0,0,0,1,2,0,0,0,0,0,0,
    0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,2,0,0,0,0,0,0,0,
    0,0,0,1,2,0,0,0,0,0,0,0,0,
    2,0,2,2,
    2,2,
    0,0,2,2,
    0,0,0,0,0,0,
    2,
    // Last test regular expression must be empty!
    //
  };

  // test data array
  char * strings[] = {
    // Invalid regular expressions.
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    // Valid regular expressions.
    " abc",
    ".*",
    "..",
    " (.|.)*d",
    ".* .*ad",
    " abc",
    "      abc",
    " aabbb ",
    "abc",
    "c",
    "ddd",
    "d",
    "c",
    "ba",
    "cd",
    "aabc",
    "ababc",
    "baabc",
    "zzdc",
    "ad",
    "azw",
    "afe",
    "age",
    "abcabcd",
    "d",
    "zdb",
    "d",
    "abc",
    "az.bcd",
    "aztzsbcdfg",
    "aztzsbcdh",
    "abdabc",
    "efg",
    "gf",
    "*** test",
    "[[ test",
    "test ))",
    " does it ever end",
    "| test",
    // Last test regular expression must be empty!
    ""
  };

  // test data array
  int match_starts[] = {
    // Invalid regular expressions.
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -4,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    -5,
    // Valid regular expressions.
    0,
    0,
    0,
    0,
    2,
    -1,
    6,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    4,
    4,
    -1,
    0,
    0,
    0,
    0,
    6,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    5,
    14,
    0,
    //
    -1
  };

  // test data array
  int match_ends[] = {
    // Invalid regular expressions.
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_SYNTAX_ERROR,
    REGEX_UNCLOSED_GROUP_ERROR,
    REGEX_UNCLOSED_GROUP_ERROR,
    REGEX_EMPTY_GROUP_ERROR,
    REGEX_EMPTY_GROUP_ERROR,
    REGEX_EMPTY_GROUP_ERROR,
    // Valid regular expressions.
    1,
    0,
    2,
    8,
    7,
    0,
    9,
    2,
    3,
    1,
    3,
    1,
    1,
    2,
    2,
    4,
    5,
    5,
    0,
    2,
    2,
    3,
    3,
    7,
    1,
    1,
    1,
    1,
    1,
    10,
    9,
    1,
    1,
    1,
    4,
    2,
    2,
    18,
    1,
    //
    STRING_EMPTY_ERROR
  };


  int done = 0;
  int i = -1;  // test index
  int ji = -1; // index in jumps / jumpf / jumpi
  while (done == 0) {
    i++; // increment test index counter

    // ===============================================================
    //                          _count     
    // 
    // Count the number of tokens and groups in this regular expression.
    int n_tokens, n_groups;
    _count(regexes[i], &n_tokens, &n_groups);

    // Verify the number of tokens and the number of groups..
    if (n_tokens != true_n_tokens[i]) {
      printf("\nRegex: '");
      for (int j = 0; regexes[i][j] != '\0'; j++) {
        printf("%s", SAFE_CHAR(regexes[i][j]));
      }
      printf("'\n\n");
      printf("ERROR: Wrong number of tokens returned by _count.\n");
      printf(" expected %d\n", true_n_tokens[i]);
      printf(" received %d\n", n_tokens);
      return(1);
    } else if (n_groups != true_n_groups[i]) {
      printf("\nRegex: '");
      for (int j = 0; regexes[i][j] != '\0'; j++) {
        printf("%s", SAFE_CHAR(regexes[i][j]));
      }
      printf("'\n\n");
      printf("ERROR: Wrong number of groups returned by _count.\n");
      printf(" expected %d\n", true_n_groups[i]);
      printf(" received %d\n", n_groups);
      return(2);
    }
    // ---------------------------------------------------------------

    if (n_tokens > 0) {
      // ===============================================================
      //                          _set_jump     
      // 
      // Initialize storage for tracking the current active tokens and
      // where to jump based on the string being parsed.
      const int mem_bytes = (2*n_tokens*sizeof(int) + 2*(n_tokens+1)*sizeof(char));
      int * jumps = malloc(mem_bytes); // jump-to location after success
      int * jumpf = jumps + n_tokens; // jump-to location after failure
      char * tokens = (char*) (jumpf + n_tokens); // regex index of each token (character)
      char * jumpi = tokens + n_tokens + 1; // immediately check next on failure
      // Terminate the two character arrays with the null character.
      tokens[n_tokens] = '\0';
      jumpi[n_tokens] = '\0';

      // Determine the jump-to tokens upon successful match and failed
      // match at each token in the regular expression.
      _set_jump(regexes[i], n_tokens, n_groups, tokens, jumps, jumpf, jumpi);

      // Verify the the tokens, jumps, jumpf, and jumpi..
      for (int j = 0; j < n_tokens; j++) {
        ji++; // (increment the test-wide counter for jumps)
        if (tokens[j] != true_tokens[i][j]) {
          printf("\nRegex: '");
          for (int j = 0; regexes[i][j] != '\0'; j++) {
            printf("%s", SAFE_CHAR(regexes[i][j]));
          }
          printf("'\n\n");
          // Re-run the code with debug printing enabled.
          DO_PRINT = 1;
          _set_jump(regexes[i], n_tokens, n_groups, tokens, jumps, jumpf, jumpi);
          printf("\n");
          printf("ERROR: Wrong TOKEN returned by _set_jump.\n");
          printf(" expected '%s' as token %d\n", SAFE_CHAR(true_tokens[i][j]), j);
          printf(" received '%s'\n", SAFE_CHAR(tokens[j]));
          return(3);
        } else if (jumps[j] != true_jumps[ji]) {
          printf("\nRegex: '");
          for (int j = 0; regexes[i][j] != '\0'; j++) {
            printf("%s", SAFE_CHAR(regexes[i][j]));
          }
          printf("'\n");
          // Re-run the code with debug printing enabled.
          DO_PRINT = 1;
          _set_jump(regexes[i], n_tokens, n_groups, tokens, jumps, jumpf, jumpi);
          printf("\n");
          printf("ERROR: Wrong JUMP S returned by _set_jump.\n");
          printf(" expected %d in col 0, row %d\n", true_jumps[ji], j);
          printf(" received %d\n", jumps[j]);
          return(4);
        } else if (jumpf[j] != true_jumpf[ji]) {
          printf("\nRegex: '");
          for (int j = 0; regexes[i][j] != '\0'; j++) {
            printf("%s", SAFE_CHAR(regexes[i][j]));
          }
          printf("'\n");
          // Re-run the code with debug printing enabled.
          DO_PRINT = 1;
          _set_jump(regexes[i], n_tokens, n_groups, tokens, jumps, jumpf, jumpi);
          printf("\n");
          printf("ERROR: Wrong JUMP F returned by _set_jump.\n");
          printf(" expected %d in col 1, row %d\n", true_jumpf[ji], j);
          printf(" received %d\n", jumpf[j]);
          return(5);
        } else if (jumpi[j] != true_jumpi[ji]) {
          printf("\nRegex: '");
          for (int j = 0; regexes[i][j] != '\0'; j++) {
            printf("%s", SAFE_CHAR(regexes[i][j]));
          }
          printf("'\n");
          // Re-run the code with debug printing enabled.
          DO_PRINT = 1;
          _set_jump(regexes[i], n_tokens, n_groups, tokens, jumps, jumpf, jumpi);
          printf("\n");
          printf("ERROR: Wrong JUMP I returned by _set_jump.\n");
          printf(" expected %d in col 2, row %d\n", true_jumpi[ji], j);
          printf(" received %d\n", jumpi[j]);
          return(6);
        }
      }
      free(jumps); // free the allocated memory
    }
    // -------------------------------------------------------------


    // =============================================================
    //                          match     
    // 
    int start;
    int end;
    match(regexes[i], strings[i], &start, &end);

    if (start != match_starts[i]) {
      DO_PRINT = 1;
      match(regexes[i], strings[i], &start, &end);
      printf("\nString: '");
      for (int j = 0; strings[i][j] != '\0'; j++) {
        printf("%s", SAFE_CHAR(strings[i][j]));
      }
      printf("'\n\n");
      printf("ERROR: Bad match START returned by match.\n");
      printf(" expected %d\n", match_starts[i]);
      printf(" received %d\n", start);
      return(7);
    } else if (end != match_ends[i]) {
      DO_PRINT = 1;
      match(regexes[i], strings[i], &start, &end);      
      printf("\nString: '");
      for (int j = 0; strings[i][j] != '\0'; j++) {
        printf("%s", SAFE_CHAR(strings[i][j]));
      }
      printf("'\n\n");
      printf("ERROR: Bad match END returned by match.\n");
      printf(" expected %d\n", match_ends[i]);
      printf(" received %d\n", end);
      return(8);
    }

    // -------------------------------------------------------------

    // Exit once the empty regex has been verified.
    if (regexes[i][0] == '\0') done++;
  }
  
  printf("\n All tests PASSED.\n");
  // Successful return.
  return(0);
}

#endif
