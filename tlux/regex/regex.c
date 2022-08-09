// ___________________________________________________________________
//                            regex.c
// 
// DESCRIPTION
//  This file provides code for very fast regular expression matching
//  over character arrays for regular expressions. Only accepts
//  regular expressions of a reduced form, with syntax:
//
//     .   any character
//     *   0 or more repetitions of the preceding token (group)
//     ?   0 or 1 occurrence of the preceding token (group)
//     |   logical OR, either match the preceding token (group) or 
//         match the next token (group)
//     ()  define a token group (designate order of operations)
//     []  define a token set of character literals (single token
//         observed as any character from a collection), use this to
//         capture any (special) characters except ']'
//     {}  logical NOT, reverse behavior of successful and failed matches
//
//  The regular expression matcher is accessible through the function:
// 
//   void match(regex, string, start, end)
//     (const char *) regex -- A null-terminated simple regular expression.
//     (const char *) string -- A null-terminated character array.
//     (int *) start -- The (pass by reference) start of the first match.
//                      If there is no match this value is -1.
//     (int *) end -- The (pass by reference) end (noninclusive) of the
//                    first match. If *start is -1, contains error code.
// 
// 
// ERROR CODES
//  These codes are returned in "end" when "start<0".
//   0  Successful execution, no match found.
//   1  Regex error, no tokens.
//   2  Regex error, unterminated token set of character literals.
//   3  Regex error, bad syntax (i.e., starts with ')', '*', '?',
//        or '|', empty second argument to '|', or bad token
//        preceding '*' or '?').
//   4  Regex error, empty group, contains "()", "[]", or "{}".
//   5  Regex error, a group starting with '(', '[', or '{' is
//        not terminated.
// 
//
// COMPILATION
//  Compile shared object (importable by Python) with:
//    cc -O3 -fPIC -shared -o regex.so regex.c
// 
//  Compile and run test (including debugging print statements) with:
//    cc -o test_regex regex.c && ./test_regex
// 
//
// EXAMPLES:
//  Match any text contained within square brackets.
//    ".*[[].*].*" 
//
//  Match any dates of the form YYYY-MM-DD or YYYY/MM/DD.
//    ".*[0123456789][0123456789][0123456789][0123456789][-/][0123456789][0123456789][-/][0123456789][0123456789].*"
//
//  Match any comments that only have horizontal whitespace before them in a C code file.
//    ".*\n[ \t\r]*//.*"
// 
//  Match this entire documentation block in this C file.
//    "/.*\n{/}"
// 
//  Match "$TEXT$" where TEXT does not contain "\n\n" (two sequential new lines).
//    "$(({\n}\n?)|(\n?{\n}))*$"
//
//
// NOTES:
// 
//  A more verbose format is required to match common character
//  collections. These substitutions can easily be placed in regular
//  expression pre-processing code. Some substitution examples:
// 
//    +     replace with one explicit copy of preceding token (group),
//            followed by a copy with '*'
//    ^     this is implicitly at the beginning of all regex'es,
//            disable by including ".*" at the beginning of the regex
//    $     include "{.}" at the end of the regex
//    [~ab] replace with {[ab]}
//    \d    replace with "[0123456789]"
//    \D    replace with "{[0123456789]}"
//    \s    replace with "[ \t\n\r]"
//    {n}   replace with n occurrences of the preceding group
//    {n,}  replace with n occurrences of the preceding group,
//            followed by a copy with '*'
//    {n,m} replace with n occurrences of the preceding group, then
//            m-n repetitions of the group all with '?'
// 
// ___________________________________________________________________

#include <stdio.h>  // printf, EOF
#include <stdlib.h> // malloc, free

#define EXIT_TOKEN -1
#define REGEX_NO_TOKENS_ERROR -1
#define REGEX_UNCLOSED_GROUP_ERROR -2
#define REGEX_SYNTAX_ERROR -3
#define REGEX_EMPTY_GROUP_ERROR -4
#define STRING_EMPTY_ERROR -5
#define DEFAULT_GROUP_MOD ' '
#define MIN_SAMPLE_SIZE 100
//      ^^ minimum number of bytes read before checking ASCII ratio
#define FILE_BUFFER_SIZE 33554432
//      ^^ 2^25 = 32MB = 33554432 bytes
#define INITIAL_FOUND_SIZE 4
//      ^^ default size of the arrays that store all regex matches

//  Name:
//    frex  -- fast regular expressions (frexi for case insensitive)

// Count the number of tokens and groups in a regular expression.
void _count(const char * regex, int * tokens, int * groups) {
  // Initialize the number of tokens and groups to 0.
  (*tokens) = 0;
  (*groups) = 0;
  int i = 0;             // regex character index
  char token = regex[i]; // current character in regex
  int gc = 0;            // groups closed
  char pt = '\0';        // previous token
  // Count tokens and groups.
  while (token != '\0') {
    // Count a character set as a single character.
    if (token == '[') {
      // Increment the count of the number of groups.
      (*groups)++;
      int tokens_in_group = 0;
      // Loop until this character set is complete.
      i++;
      pt = token;
      token = regex[i];
      while ((token != '\0') && (token != ']')) {
        // One token for this character in the group
        (*tokens)++;
        tokens_in_group++;
        // Increment to next character.
        i++;
        pt = token;
        token = regex[i];
      }
      // Check for an error in the regular expression.
      if (token == '\0') {
        (*tokens) = -i-1;
        (*groups) = REGEX_UNCLOSED_GROUP_ERROR;
        gc = REGEX_UNCLOSED_GROUP_ERROR;
        break;
        // Check for an empty group error.
      } else if (tokens_in_group == 0) {
        (*tokens) = -i-1;
        (*groups) = REGEX_EMPTY_GROUP_ERROR;
        gc = REGEX_EMPTY_GROUP_ERROR;
        break;
        // This group successfully closed.
      } else {
        gc++;
      }
      // If this is the beginning of another type of group, count it.
    } else if ((token == '(') || (token == '{')) {
      (*groups)++;
      // Check for invalid regular expressions
    } else if (
               // starts with a special character
               ((i == 0) && ((token == ')') || (token == ']') || (token == '}') || 
                             (token == '*') || (token == '?') || (token == '|'))) ||
               // illegally placed * or ?
               ((i > 0) && ((token == '*') || (token == '?')) &&
                ((pt == '*') || (pt == '?') || (pt == '(') || (pt == '{') || (pt == '|'))) ||
               // illegally placed ), ], or } after a |
               ((i > 0) && (pt == '|') && ((token == ')') || (token == ']') || (token == '}'))) ||
               // | at the end of the regex
               ((token == '|') && (regex[i+1] == '\0'))
               ) {
      (*tokens) = -i-1;
      (*groups) = REGEX_SYNTAX_ERROR;
      gc = REGEX_SYNTAX_ERROR;
      break;
      // Close opened groups
    } else if ((token == ')') || (token == '}')) {
      gc++;
      // too many closed groups, or an empty group.
      if ( (gc > (*groups)) ||
           ((token == ')') && (pt == '(')) ||
           ((token == '}') && (pt == '{')) ) {
        (*tokens) = -i-1;
        (*groups) = REGEX_EMPTY_GROUP_ERROR;
        gc = REGEX_EMPTY_GROUP_ERROR;
        break;
      }
      // If the character is counted (not special), count it as one.
    } else {
      (*tokens)++;
    }
    // Increment to the next step.
    i++;
    pt = token;
    token = regex[i];
  }
  // Error if there are unclosed groups.
  if (gc != (*groups)) {
    (*tokens) = -i-1;
    (*groups) = REGEX_UNCLOSED_GROUP_ERROR;
  }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DEBUG: Show regex and number of tokens computed.
#ifdef DEBUG
  if (DO_PRINT) {
    printf("\nRegex: '");
    for (int j = 0; regex[j] != '\0'; j++) {
      printf("%s", SAFE_CHAR(regex[j]));
    }
    printf("'\n tokens: %d\n groups: %d\n", (*tokens), (*groups));
  }
#endif
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Return the total number of tokens and groups.
  return;
}


// Read through the regular expression with the (already counted)
// number of tokens + groups and set the jump conditions.
void _set_jump(const char * regex, const int n_tokens, int n_groups,
               char * tokens, int * jumps, int * jumpf, char * jumpi) {

  // Initialize storage for the first and first proceding token of groups.
  int * group_starts = malloc((4*n_groups+n_tokens+2)*sizeof(int) + 2*n_groups*sizeof(char));
  int * group_nexts = group_starts + n_groups;
  int * gi_stack = group_nexts + n_groups; // active group stack
  int * gc_stack = gi_stack + n_groups; // closed group stack
  int * redirect = 1 + gc_stack + n_groups; // jump redirection (might access -1 or n_tokens)
  char * s_stack = (char*) (redirect + n_tokens + 1); // active group start character stack
  char * g_mods = s_stack + n_groups; // track the modifiers on each group

  // Initialize all the group pointers to a known value.
  for (int j = 0; j < n_groups; j++) {
    group_starts[j] = EXIT_TOKEN;
    group_nexts[j] = EXIT_TOKEN;
    gi_stack[j] = EXIT_TOKEN;
    gc_stack[j] = EXIT_TOKEN;
    s_stack[j] = DEFAULT_GROUP_MOD;
    g_mods[j] = DEFAULT_GROUP_MOD;
    redirect[j] = j;
  }
  // declare the rest of redirect.
  for (int j = n_groups; j <= n_tokens; j++) redirect[j] = j;
  // (declare the -1 to point to -1)
  redirect[EXIT_TOKEN] = EXIT_TOKEN;

  // =================================================================
  //                          FIRST PASS
  //  Identify the first token and first "next token" for each group
  //  as well as the modifiers placed on each group (*, ?, |).
  // 
  int i = 0;    // regex index
  int nt = 0;   // total tokens seen
  int ng = 0;   // total groups seen
  int gi = -1;  // group index
  int iga = -1; // index of active group (in stack)
  int igc = -1; // group closed index (in stack)
  char cgs = '\0' ; // character for active group start
  char token = regex[i]; // current character

  // Loop until all tokens in the regex have been processed.
  while (token != '\0') {
    // Look for the beginning of a new group.
    if (((token == '(') || (token == '[') || (token == '{')) && (cgs != '[')) {
      gi = ng;              // set current group index
      cgs = token;          // set current group start character
      ng++;                 // increment the number of groups seen
      // Push group index and start character into stack.
      iga++;                // increase active group index
      gi_stack[iga] = gi;   // push group index to stack
      s_stack[iga] = token; // push group start character to stack
      group_starts[gi] = nt; // set the start token for this group
      // Set the end of a group.
    } else if ((iga >= 0) &&
               (((cgs == '(') && (token == ')')) ||
                ((cgs == '[') && (token == ']')) ||
                ((cgs == '{') && (token == '}')))) {
      // Close the group, place it into the closed group stack.
      igc++;
      gc_stack[igc] = gi;
      // Check to see if the next character is a modifier.
      token = regex[i+1]; // (can reuse this, it will be overwritten after later i++)
      if ((token == '*') || (token == '?') || (token == '|')) {
        g_mods[gi] = token;
      }
      // Pop previous group index and start character from stack.
      iga--;
      if (iga >= 0) {
        gi = gi_stack[iga];
        cgs = s_stack[iga];
      } else {
        gi = -1;
        cgs = '\0';
      }
      // Handle normal tokens.
    } else {
      // if (not special)
      if ((cgs == '[') || ((token != '*') && (token != '?') && (token != '|'))) {
        // Set the "next" token for the recently closed groups.
        for (int j = 0; j <= igc; j++) {
          group_nexts[gc_stack[j]] = nt;
        }
        igc = -1; // now no groups are waiting to be closed
      }
      tokens[nt] = token; // store the token
      jumps[nt] = nt+1; // initialize jump on successful match to next token
      jumpf[nt] = EXIT_TOKEN; // initialize jump on failed match to exit
      jumpi[nt] = 0; // initialize immediately check next to 0 (off)
      nt++; // increment to next token
    }
    // Cycle to the next token.
    i++;
    token = regex[i];
  }

  // Set the "next" token for the closed groups at the end.
  for (int j = 0; j <= igc; j++)
    group_nexts[gc_stack[j]] = nt;
  igc = -1;
  
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DEBUG: Print out the tokens in the order given in the regex.
#ifdef DEBUG
  if (DO_PRINT) {
    printf("\nTokens (before prefixing modifiers)\n  ");
    for (int j = 0; j < n_tokens; j++) printf("%-2s  ", SAFE_CHAR(tokens[j]));
    printf("\n");
    if (n_groups > 0) {
      printf("\n");
      printf("Groups (before prefixing modifiers)\n");
      for (int j = 0; j < n_groups; j++) {
        printf(" %d: %c (%-2d %s)  -->", j, g_mods[j], group_starts[j],
               SAFE_CHAR(tokens[group_starts[j]]));
        printf("  (%-2d %s) \n", group_nexts[j]-1, 
               SAFE_CHAR(tokens[group_nexts[j]-1]));
      }
    }
  }
#endif
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  // =================================================================
  //                        SECOND PASS
  // Shift group modifier tokens to be prefixes (at front of group).
  // Assign all tokens to their final location (specials prefixed).
  // 
  i = 0;    // regex index
  nt = 0;   // total tokens seen
  ng = 0;   // total groups seen
  gi = -1;  // group index
  iga = -1; // index of active group (in stack)
  cgs = '\0'; // current group start
  token = regex[i]; // current character
  
  int gx = 0; // current group shift (number of tokens moved to front)
  while (token != '\0') {
    // Look for the beginning of a new group.
    if (((token == '(') || (token == '[') || (token == '{')) && (cgs != '[')) {
      // -------------------------------------------------------------
      // Shift this group start by any active modifiers on containing groups.
      if (gx > 0) group_starts[ng] += gx;
      // -------------------------------------------------------------
      gi = ng;              // set current group index
      cgs = token;          // set current group start character
      ng++;                 // increment the number of groups seen
      // Push group index and start character into stack.
      iga++;                // increase active group index
      gi_stack[iga] = gi;   // push group index to stack
      s_stack[iga] = token; // push group start character to stack
      // -------------------------------------------------------------
      // Push a modifier onto the stack if it exists.
      if (g_mods[gi] != DEFAULT_GROUP_MOD) {      
        gx++; // increase count of prepended modifiers
        // push starts of all affected groups back by one character
        tokens[nt] = g_mods[gi]; // store this modifier at the front
        nt++; // increment the token counter
      }
      // -------------------------------------------------------------
      // Set the end of a group.
    } else if ((iga >= 0) &&
               (((cgs == '(') && (token == ')')) ||
                ((cgs == '[') && (token == ']')) ||
                ((cgs == '{') && (token == '}')))) {
      // decrement the number of shifts that have occurred for prefix
      if (g_mods[gi] != DEFAULT_GROUP_MOD) {
        gx--;
        const int last_in_group = nt-1;
        // Shift all "group_nexts" that are contained in this group.
        //  WARNING: This makes the algorithmic complexity quadratic
        //           with the number of nested groups. Unavoidable?
        for (int j = gi; j < ng; j++)
          if (group_nexts[j] < last_in_group)
            group_nexts[j]++;
      }
      // Pop group index from the stack.
      iga--;
      if (iga >= 0) {
        gi = gi_stack[iga];
        cgs = s_stack[iga];
      } else {
        gi = -1;
        cgs = '\0';
      }
      // ---------------------------------------------------------------
      // Handle tokens.
    } else if (nt < n_tokens) {
      // If the next token is special, put it in front.
      const char nx_token = regex[i+1]; // temporary storage
      // if in token set
      if (cgs == '[') { // do nothing
        // Not in token set, handle special looping modifiers on single tokens
      } else if ((nx_token == '*') || (nx_token == '?') || (nx_token == '|')) {
        tokens[nt] = nx_token; // store this modifier at the front
        nt++; // increment the token counter.
        i++; // increment the regex index counter
        tokens[nt+1] = token; // move the original token back one
      }
      // store the token, skip specials that were already stored earlier
      if ((cgs == '[') || ((token != '*') && (token != '?') && (token != '|'))) {
        tokens[nt] = token; // store this token.
        nt++; // increment the token counter.
      }
    }
    // ---------------------------------------------------------------
    // Cycle to the next token.
    i++;
    token = regex[i];
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DEBUG: Print out the tokens with prefixed modifiers
#ifdef DEBUG
  if (DO_PRINT) {
    printf("\nTokens: (token / token index)\n  ");
    for (int j = 0; j < n_tokens; j++) printf("%-2s  ", SAFE_CHAR(tokens[j]));
    printf("\n  ");
    for (int j = 0; j < n_tokens; j++) printf("%-2d  ", j);
    printf("\n\n");
    // DEBUG: Print out the groups (new start and end with prefixes)
    if (n_groups > 0) {
      printf("Groups:  (group: mod, start token --> last token)\n");
      for (int j = 0; j < n_groups; j++) {
        printf(" %d: %c (%-2d %s)  -->", j, g_mods[j], group_starts[j],
               SAFE_CHAR(tokens[group_starts[j]]));
        printf("  (%-2d %s) \n", group_nexts[j]-1, 
               SAFE_CHAR(tokens[group_nexts[j]-1]));
      }
      printf("\n");
    }
  }
#endif
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Define an in-line function (fast) that correctly redirects any
  // token jump. This assumes that negative redirects are contained
  // within a {} group and should logically flip "success" and "fail".
  //   i  --  integer index of the token to redirect
  //   s  --  integer intended "success" destination
  //   f  --  integer intended "fail" destination
#define SET_JUMP(i, s, f)                       \
  if (neg) {                                    \
    jumps[i] = redirect[f];                     \
    jumpf[i] = redirect[s];                     \
  } else {                                      \
    jumps[i] = redirect[s];                     \
    jumpf[i] = redirect[f];                     \
  }

  // =================================================================
  //                        THIRD PASS
  // Assign jump-to success / failed / immediate for all tokens.
  // 
  i = 0;    // regex index
  nt = 0;   // total tokens seen
  ng = 0;   // total groups seen
  gi = -1;  // group index
  iga = -1; // index of active group (in stack)
  cgs = '\0'; // current group start
  token = regex[i]; // current character
  
  char neg = 0; // whether or not current section is negated
  while (token != '\0') {
    // Look for the beginning of a new group.
    if (((token == '(') || (token == '[') || (token == '{')) && (cgs != '[')) {
      // -------------------------------------------------------------
      gi = ng;              // set current group index
      cgs = token;          // set current group start character
      ng++;                 // increment the number of groups seen
      // Push group index and start character into stack.
      iga++;                // increase active group index
      gi_stack[iga] = gi;   // push group index to stack
      s_stack[iga] = token; // push group start character to stack
      // -------------------------------------------------------------
      // Push a modifier onto the stack if it exists.
      if (g_mods[gi] != DEFAULT_GROUP_MOD) {      
        // set jump conditions for modifier (prevent reversal from negation)
        if (neg) { SET_JUMP(nt, group_nexts[gi], nt+1); }
        else { SET_JUMP(nt, nt+1, group_nexts[gi]); }
        redirect[nt] = nt; // reset redirect to completed token
        nt++; // increment the token counter
        // assign 'redirect' based on the modifier.
        if (g_mods[gi] == '*') {
          redirect[group_nexts[gi]] = nt-1; // -1 because of nt++ above
        } else if (g_mods[gi] == '|') {
          // search for a group that starts at the first token after this
          int j = gi+1;
          while ((j < n_groups) && (group_starts[j] < group_nexts[gi])) j++;
          // if a group immediately follows (after the |)..
          if ((j < n_groups) && (group_starts[j] == group_nexts[gi]))
            redirect[group_nexts[gi]] = group_nexts[j];
          // otherwise a single token follows (after the |)..
          else
            redirect[group_nexts[gi]] = group_nexts[gi]+1;
        }
      }
      // Toggle "negated" for negated groups.
      if (cgs == '{') neg = (neg + 1) % 2;
      // -------------------------------------------------------------
      // Set the end of a group.
    } else if ((iga >= 0) &&
               (((cgs == '(') && (token == ')')) ||
                ((cgs == '[') && (token == ']')) ||
                ((cgs == '{') && (token == '}')))) {
      // Toggle "negated" for negated groups.
      if (token == '}') neg = (neg + 1) % 2;
      // Pop group index from the stack.
      iga--;
      if (iga >= 0) {
        gi = gi_stack[iga];
        cgs = s_stack[iga];
      } else {
        gi = -1;
        cgs = '\0';
      }
      // ---------------------------------------------------------------
      // Handle tokens.
    } else if (nt < n_tokens) {
      const char nx_token = regex[i+1]; // temporary storage
      // if in token set
      if (cgs == '[') {
        jumpi[nt] = 1; // set an immediate jump on failure
        // if this is the last token in the token set..
        if (nx_token == ']') {
          jumpi[nt] = 2; // set this as an "end of token set" element
          SET_JUMP(nt, group_nexts[gi], EXIT_TOKEN);
          // otherwise this is not the last token in the token set..
        } else {
          // this group is negated, so exit on "success" (compensate for flip)
          if (neg) {
            SET_JUMP(nt, nt+1, EXIT_TOKEN);
            // this is not negated nor last, success exits set, failure steps to next
          } else {
            SET_JUMP(nt, group_nexts[gi], nt+1);
          }
        }
        // Not in token set, handle special looping modifiers on single tokens
      } else if ((nx_token == '*') || (nx_token == '?') || (nx_token == '|')) {
        SET_JUMP(nt, nt+(neg?2:1), nt+(neg?1:2)); // set jump conditions for special token
        redirect[nt] = nt; // reset redirect to completed token
        nt++; // increment the token counter.
        i++; // increment the regex index counter
        // make tokens followed by * loop back to *
        if (nx_token == '*') {
          SET_JUMP(nt, nt-1, EXIT_TOKEN);
          // make tokens followed by | correctly jump
        } else if (nx_token == '|'){  
          const char nxnx_token = regex[i+1]; // guaranteed to exist
          if ((nxnx_token == '(') || (nxnx_token == '[') || (nxnx_token == '{')) {
            SET_JUMP(nt, group_nexts[ng+1], EXIT_TOKEN); // last token jumps after next group
          } else {
            SET_JUMP(nt, nt+2, EXIT_TOKEN); // last token jumps after next token
          }
          // make tokens followed by ? correctly jump
        } else if (nx_token == '?') {
          SET_JUMP(nt, nt+1, EXIT_TOKEN);
        }
        // if this is a standard token..
      } else {
        SET_JUMP(nt, nt+1, EXIT_TOKEN);
      }
      redirect[nt] = nt; // reset redirect to completed token
      // store the token, skip specials that were already stored earlier
      if ((cgs == '[') || ((token != '*') && (token != '?') && (token != '|'))) {
        nt++; // increment the token counter.
      }
    }
    // ---------------------------------------------------------------
    // Cycle to the next token.
    i++;
    token = regex[i];
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DEBUG: Print out the group starts and ends.
#ifdef DEBUG
  if (DO_PRINT) {
    printf("Jumps/f/i:  (token: jump on match, jump on failed match, jump immediately on fail)\n");
    for (int i = 0; i < n_tokens; i++) {
      printf(" (%-2d%2s):  % -3d % -3d  %1d\n", i, SAFE_CHAR(tokens[i]), jumps[i], jumpf[i], jumpi[i]);
    }
    printf("\n");
  }
#endif
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Free memory used for tracking the start and ends of groups.
  free(group_starts);
  return;
}


// Do a simple regular experession match.
void match(const char * regex, const char * string, int * start, int * end) {

  // Check for an empty string.
  if (string[0] == '\0') {
    (*start) = EXIT_TOKEN;
    (*end) = STRING_EMPTY_ERROR;
    return;
  }

  // Count the number of tokens and groups in this regular expression.
  int n_tokens, n_groups;
  _count(regex, &n_tokens, &n_groups);
  // Error mode, fewer than one token (no possible match).
  if (n_tokens <= 0) {
    // Set the error flag and return.
    if (n_tokens == 0) {
      (*start) = EXIT_TOKEN;
      (*end) = REGEX_NO_TOKENS_ERROR;
    } else {
      (*start) = n_tokens;
      (*end) = n_groups;
    }
    return;
  }

  // Initialize storage for tracking the current active tokens and
  // where to jump based on the string being parsed.
  const int mem_bytes = ((5*n_tokens+1)*sizeof(int) + (4*n_tokens+2)*sizeof(char));
  int * jumps = malloc(mem_bytes); // jump-to location after success
  int * jumpf = jumps + n_tokens; // jump-to location after failure
  int * active = jumpf + n_tokens; // presently active tokens in regex
  int * cstack = active + n_tokens+1; // current stack of active tokens
  int * nstack = cstack + n_tokens; // next stack of active tokens
  char * tokens = (char*) (nstack + n_tokens); // regex index of each token (character)
  char * jumpi = tokens + n_tokens+1; // immediately check next on failure
  char * incs = jumpi + n_tokens+1; // token flags for "in current stack"
  char * inns = incs + n_tokens; // token flags for "in next stack"
  // Terminate the two character arrays with the null character.
  tokens[n_tokens] = '\0';
  jumpi[n_tokens] = '\0';

  // Determine the jump-to tokens upon successful match and failed
  // match at each token in the regular expression.
  _set_jump(regex, n_tokens, n_groups, tokens, jumps, jumpf, jumpi);
  // Set all tokens to be inactive, convert ? to * for simplicity.
  active[n_tokens] = EXIT_TOKEN;
  for (int j = 0; j < n_tokens; j++) {
    // convert all "special tokens" to * for speed, exclude all
    // tokens with jumpi = 1 because those are inside token sets
    if ((! jumpi[j]) && ((tokens[j] == '?') || (tokens[j] == '|')))
      tokens[j] = '*';
    active[j] = EXIT_TOKEN; // token is inactive
    incs[j] = 0; // token is not in current stack
    inns[j] = 0; // token is not in next stack
  }
  
  // Set the current index in the string.
  int i = 0; // current index in string
  char c = string[i]; // current character in string
  int ics = 0; // index in current stack
  int ins = -1; // index in next stack
  int dest; // index of next token (for jump)
  void * temp; // temporary pointer (used for transferring nstack to cstack)
  cstack[ics] = 0; // set the first element in stack to '0'
  active[0] = 0; // set the start index of the first token
  incs[0] = 1; // the first token is in the current stack

  // Initialize to "no match".
  (*start) = EXIT_TOKEN;
  (*end) = 0;

  // Define an in-line substitution that will be used repeatedly in
  // a following while loop.
  // 
  // If the destination is valid, and the current start index (val)
  // is newer than the one to be overwritten, then stack the
  // new destination, assign active, and mark as set.
  // 
  // If the destination is the "done" token, then:
  //   free all memory that was allocated,
  //   set the start and end of the match
  //   return
#define MATCH_STACK_NEXT_TOKEN(stack, si, in_stack)         \
  if ((dest >= 0) && (val >= active[dest])) {               \
    if (in_stack[dest] == 0) {                              \
      si++;                                                 \
      stack[si] = dest;                                     \
    }                                                       \
    if (dest == n_tokens) {                                 \
      free(jumps);                                          \
      (*start) = val;                                       \
      (*end) = i;                                           \
      if ((jumpi[j]) || (ct != '*')) (*end)++;              \
      return;                                               \
    } else {                                                \
      in_stack[dest] = 1;                                   \
      if (active[dest] == EXIT_TOKEN) active[dest] = val;   \
    }                                                       \
  }

  // Start searching for a regular expression match. (the character
  // 'c' is checked for null value at the end of the loop.
  do {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifdef DEBUG
    if (DO_PRINT) {
      printf("--------------------------------------------------\n");
      printf("i = %d   c = '%s'\n\n", i, SAFE_CHAR(c));
      printf("stack:\n");
      for (int j = ics;  j >= 0; j--) {
        printf(" '%s' (at %2d) %d\n", SAFE_CHAR(tokens[cstack[j]]), cstack[j], active[cstack[j]]);
      }
      printf("\n");
      printf("active: (search token / index of match start)\n");
      for (int j = 0; j <= n_tokens; j++) {
        printf("  %-3s", SAFE_CHAR(tokens[j]));
      }
      printf("\n");
      for (int j = 0; j <= n_tokens; j++) {
        printf("  %-3d", active[j]);
      }
      printf("\n\n");
    }
#endif
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Continue popping active elements from the current stack and
    // checking them for a match and jump conditions, add next tokens
    // to the next stack.
    while (ics >= 0) {
      // Pop next token to check from the stack, skip if already done.
      const int j = cstack[ics];
      ics--;
      incs[j] = 0;
      // Get the token and the "start index" for the match that led here.
      const char ct = tokens[j];
      int val = active[j];
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifdef DEBUG
      if (DO_PRINT) {
        printf("    j = %d   ct = '%s'  %2d %2d  (%d)\n", j, SAFE_CHAR(ct), jumps[j], jumpf[j], val);
      }
#endif
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // If this is a special character, add its tokens immediately to
      // the current stack (to be checked before next charactrer).
      if ((ct == '*') && (! jumpi[j])) {
        if (j == 0) val = i; // ignore leading tokens where possible
        dest = jumps[j];
        MATCH_STACK_NEXT_TOKEN(cstack, ics, incs);
        dest = jumpf[j];
        MATCH_STACK_NEXT_TOKEN(cstack, ics, incs);
        // Check to see if this token matches the current character.
      } else if ((c == ct) || ((ct == '.') && (! jumpi[j]) && (c != '\0'))) {
        dest = jumps[j];
        MATCH_STACK_NEXT_TOKEN(nstack, ins, inns);
        // This token did not match, trigger a jump fail.
      } else {
        dest = jumpf[j];
        // jump immediately on fail if this is not the last token in a token set
        if (jumpi[j] == 1) { 
          MATCH_STACK_NEXT_TOKEN(cstack, ics, incs);
          // otherwise, put into the "next" stack
        } else { 
          MATCH_STACK_NEXT_TOKEN(nstack, ins, inns);
        }
      }
    }
    // Switch out the current stack with the next stack.
    //   switch stack of token indices
    temp = (void*) cstack; // store "current stack"
    cstack = nstack; // set "current stack"
    ics = ins; // set "index in current stack"
    nstack = (int*) temp; // set "next stack"
    //   switch flag arrays of "token in stack"
    temp = (void*) incs; // store "in current stack"
    incs = inns; // set "in current stack"
    inns = (char*) temp; // set "in next stack"
    ins = -1; // reset the count of elements in "next stack"

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifdef DEBUG
    if (DO_PRINT) {
      printf("\n");
    }
#endif
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // If the just-parsed character was the end of the string, then break.
    if (c == '\0') {
      break;
      // Get the next character in the string (assuming it's null terminated).
    } else {
      i++;
      c = string[i];
    }
  } while (ics >= 0) ; // loop until the active stack is empty
  free(jumps); // free all memory that was allocated
  return;
} 

// Find all nonoverlapping matches of a regular expression in a string.
// Return arrays of the starts and ends of matches.
void matcha(const char * regex, const char * string,
            int * n, int ** starts, int ** ends) {

  // Check for an empty string.
  if (string[0] == '\0') {
    (*n) = -2;
    return;
  }

  // Count the number of tokens and groups in this regular expression.
  (*n) = -1;
  int n_tokens, n_groups;
  _count(regex, &n_tokens, &n_groups);
  // Error mode, fewer than one token (no possible match).
  if (n_tokens <= 0) {
    (*starts) = malloc(2 * sizeof(int));
    (*ends) = (*starts) + 1;
    // Set the error flag and return.
    if (n_tokens == 0) {
      (*starts)[0] = EXIT_TOKEN;
      (*ends)[0] = REGEX_NO_TOKENS_ERROR;
    } else {
      (*starts)[0] = n_tokens;
      (*ends)[0] = n_groups;
    }
    // WARNING: returning pointers to allocated memory for two integers!
    return;
  }

  // Set there to be 0 matches, initially.
  (*n) = 0;
  // Initialize storage for tracking the current active tokens and
  // where to jump based on the string being parsed.
  const int mem_bytes = ((5*n_tokens+1)*sizeof(int) + (4*n_tokens+2)*sizeof(char));
  int * jumps = malloc(mem_bytes); // jump-to location after success
  int * jumpf = jumps + n_tokens; // jump-to location after failure
  int * active = jumpf + n_tokens; // presently active tokens in regex
  int * cstack = active + n_tokens + 1; // current stack of active tokens
  int * nstack = cstack + n_tokens; // next stack of active tokens
  char * tokens = (char*) (nstack + n_tokens); // regex index of each token (character)
  char * jumpi = tokens + n_tokens + 1; // immediately check next on failure
  char * incs = jumpi + n_tokens; // token flags for "in current stack"
  char * inns = incs + n_tokens; // token flags for "in next stack"
  // Terminate the two character arrays with the null character.
  tokens[n_tokens] = '\0';
  jumpi[n_tokens] = '\0';

  // Determine the jump-to tokens upon successful match and failed
  // match at each token in the regular expression.
  _set_jump(regex, n_tokens, n_groups, tokens, jumps, jumpf, jumpi);
  // Set all tokens to be inactive, convert ? to * for simplicity.
  active[n_tokens] = EXIT_TOKEN;
  for (int j = 0; j < n_tokens; j++) {
    // convert both "special tokens" to * for speed, exclude all
    // tokens with jumpi = 1 because those are inside token sets
    if ((! jumpi[j]) && ((tokens[j] == '?') || (tokens[j] == '|')))
      tokens[j] = '*';
    active[j] = EXIT_TOKEN; // token is inactive
    incs[j] = 0; // token is not in current stack
    inns[j] = 0; // token is not in next stack
  }

  // Set the current index in the file.
  int i = 0; // current index in file
  int c = string[i]; // get current character in file
  int ics = 0; // index in current stack
  int ins = -1; // index in next stack
  int dest; // index of next token (for jump)
  void * temp; // temporary pointer (used for transferring nstack to cstack)
  cstack[ics] = 0; // set the first element in stack to '0'
  active[0] = 0; // set the start index of the first token
  incs[0] = 1; // the first token is in the current stack

  // Initialize to "no match".
  (*starts) = NULL;
  (*ends) = NULL;
  int n_found = 0; // number found
  int s_found = 0; // size of "found" arrays

  // Define an in-line substitution that will be used repeatedly in
  // a following while loop.
#define MATCHA_STACK_NEXT_TOKEN(stack, si, in_stack)              \
  if ((dest >= 0) && (val >= active[dest])) {                     \
    if (dest == n_tokens) {                                       \
      (*n)++;                                                     \
      if (n_found >= s_found) {                                   \
        if (s_found == 0) s_found = INITIAL_FOUND_SIZE;           \
        else s_found = 2*s_found;                                 \
        int * new_starts = malloc(3 * s_found * sizeof(int));     \
        int * new_ends = new_starts + s_found;                    \
        for (int index = 0; index < n_found; index++)      {      \
          new_starts[index] = (*starts)[index];                   \
          new_ends[index] = (*ends)[index];                       \
        }                                                         \
        if ((*starts) != NULL) free(*starts);                     \
        (*starts) = new_starts;                                   \
        (*ends) = new_ends;                                       \
      }                                                           \
      (*starts)[n_found] = val;                                   \
      (*ends)[n_found] = (((jumpi[j]) || (ct != '*')) ? i+1 : i); \
      n_found++;                                                  \
    } else {                                                      \
      if (in_stack[dest] == 0) {                                  \
        si++;                                                     \
        stack[si] = dest;                                         \
        in_stack[dest] = 1;                                       \
      }                                                           \
      active[dest] = val;                                         \
    }                                                             \
  }                                

  // Start searching for a regular expression match. (the character
  // 'c' is checked for null value at the end of the loop.
  do {
    // Continue popping active elements from the current stack and
    // checking them for a match and jump conditions, add next tokens
    // to the next stack.
    while (ics >= 0) {
      // Pop next token to check from the stack.
      const int j = cstack[ics];
      ics--;
      incs[j] = 0;
      // Get the token and the "match start index" for the match that led here.
      const char ct = tokens[j];
      int val = active[j];
      // If this is a special character, add its tokens immediately to
      // the current stack (to be checked before next charactrer).
      if ((ct == '*') && (! jumpi[j])) {
        if (j == 0) val = i; // ignore leading tokens where possible
        dest = jumps[j];
        MATCHA_STACK_NEXT_TOKEN(cstack, ics, incs);
        dest = jumpf[j];
        MATCHA_STACK_NEXT_TOKEN(cstack, ics, incs);
        // Check to see if this token matches the current character.
      } else if ((c == ct) || ((ct == '.') && (! jumpi[j]) && (c != '\0'))) {
        dest = jumps[j];
        MATCHA_STACK_NEXT_TOKEN(nstack, ins, inns);
        // This token did not match, trigger a jump fail.
      } else {
        dest = jumpf[j];
        // jump immediately on fail if this is not the last token in a token set
        if (jumpi[j] == 1) { 
          MATCHA_STACK_NEXT_TOKEN(cstack, ics, incs);
          // otherwise, put into the "next" stack
        } else { 
          MATCHA_STACK_NEXT_TOKEN(nstack, ins, inns);
        }
      }
    }
    // Switch out the current stack with the next stack.
    //   switch stack of token indices
    temp = (void*) cstack; // store "current stack"
    cstack = nstack; // set "current stack"
    ics = ins; // set "index in current stack"
    nstack = (int*) temp; // set "next stack"
    //   switch flag arrays of "token in stack"
    temp = (void*) incs; // store "in current stack"
    incs = inns; // set "in current stack"
    inns = (char*) temp; // set "in next stack"
    ins = -1; // reset the count of elements in "next stack"

    // If the just-parsed character was the end of the string, then break.
    if (c == '\0') {
      break;
      // Get the next character in the string (assuming it's null terminated).
    } else {
      i++;
      c = string[i];
    }
  } while (ics >= 0) ; // loop until the active stack is empty
  free(jumps); // free all memory that was allocated
  // Check for errors, deallocate 'ends', 'starts', and 'lines' if there are errors.
  if ((*n) < 0) {
    if ((*starts) != NULL) free(*starts);
    // Re-allocate the output arrays to be the exact size of the number of matches.
  } else {
    if (n_found < s_found) {
      s_found = n_found;
      int * new_starts = malloc(3 * s_found * sizeof(int));
      int * new_ends = new_starts + s_found;
      for (int index = 0; index < n_found; index++) {
        new_starts[index] = (*starts)[index];
        new_ends[index] = (*ends)[index];
      }
      if ((*starts) != NULL) free(*starts);
      (*starts) = new_starts;
      (*ends) = new_ends;
    }
  }
  return;
}



// Find all nonoverlapping matches of a regular expression in a file
// at a given path. Return arrays of the starts and ends of matches.
void fmatcha(const char * regex, const char * path,
             int * n, int ** starts, int ** ends, int ** lines,
             float min_ascii_ratio) {

  // Open the file and handle any errors.
  FILE * file = fopen(path, "r");
  if (file == NULL) {
    (*n) = -2;
    return;
  }

  // Count the number of tokens and groups in this regular expression.
  (*n) = -1;
  int n_tokens, n_groups;
  _count(regex, &n_tokens, &n_groups);
  // Error mode, fewer than one token (no possible match).
  if (n_tokens <= 0) {
    (*starts) = malloc(2 * sizeof(int));
    (*ends) = (*starts) + 1;
    // Set the error flag and return.
    if (n_tokens == 0) {
      (*starts)[0] = EXIT_TOKEN;
      (*ends)[0] = REGEX_NO_TOKENS_ERROR;
    } else {
      (*starts)[0] = n_tokens;
      (*ends)[0] = n_groups;
    }
    return;
  }

  // Set there to be 0 matches, initially.
  (*n) = 0;
  // Initialize storage for tracking the current active tokens and
  // where to jump based on the string being parsed.
  const int mem_bytes = ((5*n_tokens+1)*sizeof(int) + (4*n_tokens+2)*sizeof(char));
  int * jumps = malloc(mem_bytes); // jump-to location after success
  int * jumpf = jumps + n_tokens; // jump-to location after failure
  int * active = jumpf + n_tokens; // presently active tokens in regex
  int * cstack = active + n_tokens + 1; // current stack of active tokens
  int * nstack = cstack + n_tokens; // next stack of active tokens
  char * tokens = (char*) (nstack + n_tokens); // regex index of each token (character)
  char * jumpi = tokens + n_tokens + 1; // immediately check next on failure
  char * incs = jumpi + n_tokens; // token flags for "in current stack"
  char * inns = incs + n_tokens; // token flags for "in next stack"
  // Terminate the two character arrays with the null character.
  tokens[n_tokens] = '\0';
  jumpi[n_tokens] = '\0';

  // Create a character buffer for reading data from the file.
  size_t buffer_size = FILE_BUFFER_SIZE;
  fseek(file, 0, SEEK_END); // go to the end of the file
  size_t file_size = ftell(file); // size of file in bytes
  rewind(file); // go back to the beginning of the file
  if (file_size < buffer_size)
    buffer_size = file_size+1; // set the buffer size smaller if appropriate
  char * file_buff = malloc(sizeof(char)*buffer_size); // storage for the file contents
  int bytes_buffered = fread(file_buff, sizeof(char), buffer_size, file);
  int ib = 0; // index in buffer.
  if (bytes_buffered < buffer_size) file_buff[bytes_buffered] = EOF;

  // Determine the jump-to tokens upon successful match and failed
  // match at each token in the regular expression.
  _set_jump(regex, n_tokens, n_groups, tokens, jumps, jumpf, jumpi);
  // Set all tokens to be inactive, convert ? to * for simplicity.
  active[n_tokens] = EXIT_TOKEN;
  for (int j = 0; j < n_tokens; j++) {
    // convert both "special tokens" to * for speed, exclude all
    // tokens with jumpi = 1 because those are inside token sets
    if ((! jumpi[j]) && ((tokens[j] == '?') || (tokens[j] == '|')))
      tokens[j] = '*';
    active[j] = EXIT_TOKEN; // token is inactive
    incs[j] = 0; // token is not in current stack
    inns[j] = 0; // token is not in next stack
  }

  // Set the current index in the file.
  int i = 0; // current index in file
  int c = file_buff[ib]; // get current character in file
  int ics = 0; // index in current stack
  int ins = -1; // index in next stack
  int dest; // index of next token (for jump)
  void * temp; // temporary pointer (used for transferring nstack to cstack)
  cstack[ics] = 0; // set the first element in stack to '0'
  active[0] = 0; // set the start index of the first token
  incs[0] = 1; // the first token is in the current stack

  // Initialize to "no match".
  (*starts) = NULL;
  (*ends) = NULL;
  (*lines) = NULL;
  int n_found = 0; // number found
  int s_found = 0; // size of "found" arrays
  int lines_read = 1;
  // Track some file statistics for early exit conditions.
  float ascii_count = 0.0;
  float bytes_read = 0.0;

  // Define an in-line substitution that will be used repeatedly in
  // a following while loop.
#define FMATCHA_STACK_NEXT_TOKEN(stack, si, in_stack)             \
  if ((dest >= 0) && (val >= active[dest])) {                     \
    if (dest == n_tokens) {                                       \
      (*n)++;                                                     \
      if (n_found >= s_found) {                                   \
        if (s_found == 0) s_found = INITIAL_FOUND_SIZE;           \
        else s_found = 2*s_found;                                 \
        int * new_starts = malloc(3 * s_found * sizeof(int));     \
        int * new_ends = new_starts + s_found;                    \
        int * new_lines = new_ends + s_found;                     \
        for (int index = 0; index < n_found; index++)      {      \
          new_starts[index] = (*starts)[index];                   \
          new_ends[index] = (*ends)[index];                       \
          new_lines[index] = (*lines)[index];                     \
        }                                                         \
        if ((*starts) != NULL) free(*starts);                     \
        (*starts) = new_starts;                                   \
        (*ends) = new_ends;                                       \
        (*lines) = new_lines;                                     \
      }                                                           \
      (*starts)[n_found] = val;                                   \
      (*ends)[n_found] = (((jumpi[j]) || (ct != '*')) ? i+1 : i); \
      (*lines)[n_found] = lines_read;                             \
      n_found++;                                                  \
    } else {                                                      \
      if (in_stack[dest] == 0) {                                  \
        si++;                                                     \
        stack[si] = dest;                                         \
        in_stack[dest] = 1;                                       \
      }                                                           \
      active[dest] = val;                                         \
    }                                                             \
  }                                

  // Start searching for a regular expression match. (the character
  // 'c' is checked for null value at the end of the loop.
  do {
    // Continue popping active elements from the current stack and
    // checking them for a match and jump conditions, add next tokens
    // to the next stack.
    while (ics >= 0) {
      // Record the ASCII fraction, exit if file is too non-ASCII.
      if ((c < 128) && (c != '\0')) {
        ascii_count++;
      } else if ((i >= MIN_SAMPLE_SIZE) &&
                 ((ascii_count / i) < min_ascii_ratio)) {
        (*n) = -3;
        break;
      }
      // Pop next token to check from the stack.
      const int j = cstack[ics];
      ics--;
      incs[j] = 0;
      // Get the token and the "match start index" for the match that led here.
      const char ct = tokens[j];
      int val = active[j];
      // If this is a special character, add its tokens immediately to
      // the current stack (to be checked before next charactrer).
      if ((ct == '*') && (! jumpi[j])) {
        if (j == 0) val = i; // ignore leading tokens where possible
        dest = jumps[j];
        FMATCHA_STACK_NEXT_TOKEN(cstack, ics, incs);
        dest = jumpf[j];
        FMATCHA_STACK_NEXT_TOKEN(cstack, ics, incs);
        // Check to see if this token matches the current character.
      } else if ((c == ct) || ((ct == '.') && (! jumpi[j]) && (c != EOF))) {
        dest = jumps[j];
        FMATCHA_STACK_NEXT_TOKEN(nstack, ins, inns);
        // This token did not match, trigger a jump fail.
      } else {
        dest = jumpf[j];
        // jump immediately on fail if this is not the last token in a token set
        if (jumpi[j] == 1) { 
          FMATCHA_STACK_NEXT_TOKEN(cstack, ics, incs);
          // otherwise, put into the "next" stack
        } else { 
          FMATCHA_STACK_NEXT_TOKEN(nstack, ins, inns);
        }
      }
    }
    // Switch out the current stack with the next stack.
    //   switch stack of token indices
    temp = (void*) cstack; // store "current stack"
    cstack = nstack; // set "current stack"
    ics = ins; // set "index in current stack"
    nstack = (int*) temp; // set "next stack"
    //   switch flag arrays of "token in stack"
    temp = (void*) incs; // store "in current stack"
    incs = inns; // set "in current stack"
    inns = (char*) temp; // set "in next stack"
    ins = -1; // reset the count of elements in "next stack"

    // If the just-parsed character was the end of the string, then break.
    if (c == EOF) {
      break;
      // Get the next character in the string (assuming it's null terminated).
    } else {
      i++;
      ib++;
      // Read another chunk of the file.
      if (ib >= buffer_size) {
        // Reset the index in the buffer and read next character set.
        ib = 0;
        bytes_buffered = fread(file_buff, sizeof(char), buffer_size, file);
        if (bytes_buffered < buffer_size) file_buff[bytes_buffered] = EOF;
      }
      c = file_buff[ib];
      // Increment the number of lines read.
      if (c == '\n') lines_read++;

    }
  } while (ics >= 0) ; // loop until the active stack is empty
  free(file_buff); // free the file buffer
  free(jumps); // free all memory that was allocated
  if (ferror(file)) (*n) = -2; // check for error while reading file
  fclose(file); // close the file
  // Check for errors, deallocate 'ends', 'starts', and 'lines' if there are errors.
  if ((*n) < 0) {
    if ((*starts) != NULL) free(*starts);
  } else {
    // Re-allocate the output arrays to be the exact size of the number of matches.
    if (n_found < s_found) {
      s_found = n_found;
      int * new_starts = malloc(3 * s_found * sizeof(int));
      int * new_ends = new_starts + s_found;
      int * new_lines = new_ends + s_found;
      for (int index = 0; index < n_found; index++) {
        new_starts[index] = (*starts)[index];
        new_ends[index] = (*ends)[index];
        new_lines[index] = (*lines)[index];
      }
      if ((*starts) != NULL) free(*starts);
      (*starts) = new_starts;
      (*ends) = new_ends;
      (*lines) = new_lines;
    }
  }
  return;
}


// If DEBUG is not define, make the main of this program be a command line interface.
#ifndef DEBUG
int main(int argc, char * argv[]) {
  printf("\n");
  printf("The main program has not been implemented yet.\n");
  printf("It should behave somewhat like `grep` once done.\n");
  printf("\n");
  printf("argc = %d\n", argc);
  for (int i = 0; i < argc; i++) {
    printf("argv[%d] = \"%s\"\n", i, argv[i]);
  }
  printf("\n");
}
#endif
