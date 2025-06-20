import os, sys
from . import DEFAULT_WAIT, DEFAULT_MAX_DISPLAY, \
    DEFAULT_MAX_STR_LEN, FILE_SAMPLE_SIZE, EXTRA_CATEGORY_KEY, \
    GENERATOR_TYPE, MISSING_SAMPLE_SIZE, NP_TYPES, SEPARATORS


# TODO:  Read data that has new lines in the column values.
# TODO:  Instead of just checking file size, check number of columns too,
#        use a sample by default if the #columns is large (speed!).
# TODO:  Warnings or versioning for when the source data is modified
#        in a way that invalidates a view. Need to track unpropagated
#        changes somehow. Or make permanent changes disallowed? Or copy?
#        E.g. view is made, columns are rearranged, column indices are now invalid.
# TODO:  Make "to_matrix" automatically consider 'None' a unique value
#        for categorical columns and add a dimension 'is present' otherwise.
# TODO:  Make "to_matrix" automatically flatten elements of columns
#        that contain iterables into their base types. If the types
#        are numeric, use float, otherwise use string. Use "is numeric".
# TODO:  Check for duplicate column names, handle duplicate column
#        names? Do names *have* to be Python strings?
# TODO:  Support in-place addition with *some* duplicate columns and
#        the same number of rows? That or block duplicate col names.
# TODO:  Change printout style for data frames with lots of columns,
#        use variable max_print_width
# TODO:  Is it possible (or relevant) for a data object to become
#        *empty* after it has been correctly initialized?
# TODO:  Test the index of [int,str] on a view where the str is a
#        column name in the data object, but not in the view.
# TODO:  Add test case for loading a CSV that only has a header line.
# 
# 
# TODO:  Implement "parallel read" that reads the file object 
#        simultaneously into multiple Data objects, then merges the
#        Data objects together (efficiently, fewest copies possible).
# TODO:  Implement "C read" that gets the types of all columns using
#        a C program.
# 
# 
# TODO:  Implmenent 'IndexedAudio' reader that mimics the
#        'IndexedVideo' reader and allows for access to peaks.
# TODO:  Implemenet 'IndexedText' reader that mimics the
#        'IndexedVideo' reader and allows for access to characters.
# TODO:  Implement 'IndexedCSV' reader that reads rows out of a very
#        large CSV file.
# TODO:  Implement 'CachedData' whose underlying '.data' object caches
#        groups of rows to keep the memory footprint low. OR,
#        Implement 'LazyData' whose underlying '.data' object is a
#        lazy-evaluated 'IndexedCSV' object. 



# ======================================================
#      Python 'Data' with named and typed columns     
# ======================================================

# Return 'True' if an object is iterable.
def is_iterable(obj):
    try:    return (iter(obj) and True)
    except: return False

# Given an object of (nested) iterable(s), expand it in one list. Try
# using a built-in flatten method for the object, if that fails, do it
# the (slow) python way.
def flatten(obj, depth=1):
    if (hasattr(obj, "flatten")):
        output = obj.flatten()
    elif (hasattr(obj, "__len__") and (len(obj) == 1)):
        output = [obj[0]]
    elif (is_iterable(obj)):
        output = []
        for v in obj: output += flatten(v, depth=depth+1)
    else:
        output = [obj]
    return output

# Given two data objects, merge all their contents on a given column.
def merge_on_column(d1, d2, column, d1_name="left", d2_name="right"):
    from tlux.data import Data
    from itertools import combinations
    # Get the list of unique values that will be merged into the output.
    unique_values = sorted(set(d1[column]) | set(d2[column]))
    # Get the list of columns that will exist from both data objects.
    d1_column_set = set(d1.columns)
    d2_column_set = set(d2.columns)
    d1_columns = [(c,c + ("_"+d1_name if (c in d2_column_set) and (len(d1_name) > 0) else ""))
                  for c in d1.columns if (c != column)]
    d2_columns = [(c,c + ("_"+d2_name if (c in d1_column_set) and (len(d2_name) > 0) else ""))
                  for c in d2.columns if (c != column)]
    # Initialize storage for the merged data.
    merged_data = Data(names=[column]
                       + [c for (_,c) in d1_columns]
                       + [c for (_,c) in d2_columns])
    # Get the set of all rows from source data objects with each unique value.
    rows_with_value = {}
    for i,row in enumerate(d1):
        v = row[column]
        rows = rows_with_value.get(v, [[],[]])
        rows[0].append(i)
        rows_with_value[v] = rows
    for i,row in enumerate(d2):
        v = row[column]
        rows = rows_with_value.get(v, [[],[]])
        rows[1].append(i)
        rows_with_value[v] = rows
    # Create the merged data object.
    for v in unique_values:
        d1_rows, d2_rows = rows_with_value[v]
        if (len(d1_rows) == 0): d1_rows.append(None)
        if (len(d2_rows) == 0): d2_rows.append(None)
        for d1_i in d1_rows:
            for d2_i in d2_rows:
                # Get the corresponding row from d1.
                if (d1_i is None): d1_row = [None]*len(d1_columns)
                else: d1_row = [d1[d1_i,c] for (c,_) in d1_columns]
                # Get the corresponding row from d2.
                if (d2_i is None): d2_row = [None]*len(d2_columns)
                else: d2_row = [d2[d2_i,c] for (c,_) in d2_columns]
                # Merge the two rows and add it to the output data.
                merged_data.append([v] + d1_row + d2_row)
    # Return the merged data.
    return merged_data


# Define a python data matrix structure (named and typed columns).
class Data:
    # Holder for data.
    data = None
    # Boolean for declaring if this is a "view" or actual data.
    view = False
    # Storage for the set of rows and columns being "viewed".
    _row_indices = None
    _col_indices = None
    # Default values for 'self.names' and 'self.types'
    _names = None
    _types = None
    # Default maximum printed rows of this data.
    max_display = DEFAULT_MAX_DISPLAY
    # Default maximum string length of printed column values.
    max_str_len = DEFAULT_MAX_STR_LEN
    # Define the time waited before displaying progress.
    max_wait = DEFAULT_WAIT
    print_kwargs = dict(end="\r", flush=True)

    # Import all of the exceptions.
    from .exceptions import NameTypeMismatch, \
        BadSpecifiedType, BadSpecifiedName, NoNamesSpecified, \
        ImproperUsage, UnknownName, Unsupported, BadAssignment, \
        BadElement, BadTarget, BadValue, BadIndex, BadData, Empty, \
        MissingModule

    # Import the classes used to protect this data object from users.
    from .internals import Descriptor, Column, Row

    # Protect the "names" and "types" attributes by using properties
    # to control the access and setting of these values.
    @property
    def names(self): return self._names
    @names.setter
    def names(self, values): self._names = Data.Descriptor(self, list(map(str,values)))
    @property
    def types(self): return self._types
    @types.setter
    def types(self, values): self._types = Data.Descriptor(self, list(values))
    # Alias "names" to also be referenced as "columns".
    @property
    def columns(self): return self._names
    @columns.setter
    def columns(self, values): self._names = Data.Descriptor(self, list(map(str,values)))
    # Return True if empty.
    @property
    def empty(self): return ((self.names is None) or (self.types is None) or (len(self) == 0))
    # Declare the shape of this Data object.
    @property
    def shape(self):
        if (len(self) == 0):
            if   (self.types is not None): return (0, len(self.types.values))
            elif (self.names is not None): return (0, len(self.names.values))
            else: return (0,0)
        elif (self.view): return (len(self._row_indices), len(self._col_indices))
        else:             return (len(self), len(self.names.values)) # <- access values to save time

    # Declare a "summary" property that returns the results of
    # "summarize" as a string (instead of printing them to output).
    @property
    def summary(self):
        import tempfile
        with tempfile.TemporaryFile(mode="w+t") as f:
            self.summarize(file=f)
            f.seek(0)
            output = f.read()
        return output

    # Given an index, return the index in "self.data" that holds that row. 
    def row(self, idx):
        if self.view: return self._row_indices[idx]
        else:         return idx

    # Given an index, return the index in "self.data" that holds that row. 
    def col(self, idx):
        if self.view: return self._col_indices[idx]
        else:         return idx


    #      Overwritten list Methods     
    # ==================================

    def __init__(self, data=None, names=None, types=None, rows=None, cols=None):
        import time
        start = time.time()
        # Initialize the contents of this Data to be an empty list.
        self.data = []
        # Verify the length of the types and names
        if ((types is not None) and
            (names is not None) and
            (len(types) != len(names))):
            raise(Data.NameTypeMismatch(f"Length of provided names {len(names)} does not match length of provided types {len(types)}."))
        # Get the names (must be strings)
        if (names is not None):
            if any(type(n) != str for n in names):
                raise(Data.BadSpecifiedName(f"An entry of provided names was not of {str} class."))
            self.names = names # <- performs copy
            # Construct default types for each column if names were not provided.
            if (types is None): types = [type(None)] * len(names)
        # Store the types if provided
        if (types is not None):
            if any(type(t) != type for t in types):
                raise(Data.BadSpecifiedType(f"An entry of provided types was not of {type(type)} class."))
            self.types = types # <- performs copy
            # Construct default names for each column if names were not provided.
            if (self.names is None): self.names = [str(i) for i in range(len(types))]
        # Check if this is a view or a full data object.
        self.view = ((rows is not None) or (cols is not None))
        # If this is a "view" object, store a pointer.
        if self.view:
            if (type(None) in map(type,(data,rows,cols))):
                raise(ImproperUsage("Data, column indices, and row indices must be provided for a view."))
            self.data = data
            self._row_indices = rows
            self._col_indices = cols
        # If provided, build this data as a deep copy of provided data.
        elif (data is not None):
            has_len = hasattr(data, "__len__")
            for i,row in enumerate(data):
                # Update user on progress if too much time has elapsed..
                if (time.time() - start) > self.max_wait:
                    print(f" {(str(round(100.*i/len(data),2))+'%' if has_len else str(i))} init", **self.print_kwargs)
                    start = time.time()
                self.append(row)


    # Overwrite the standard "[]" get operator to accept strings as well.
    # Given local-specific index, convert to Data index and return value.
    def __getitem__(self, index):
        # Special case for being empty.
        if (self.empty): raise(Data.Empty("Cannot get item from empty data."))
        # This index is accessing a (row,col) entry, both integers.
        if ((type(index) == tuple) and (len(index) == 2) and 
            (type(index[0]) == int) and (type(index[1]) == int)):
            return self.data[self.row(index[0])][self.col(index[1])]
        # This index is accessing a (row,col) entry where col is a string.
        elif ((type(index) == tuple) and (len(index) == 2) and 
              (type(index[0]) == int) and (type(index[1]) == str)):
            if (index[1] not in self.names):
                raise(Data.BadIndex(f"Column index {index[1]} is not a recognized column name."))
            return self.data[self.row(index[0])][self.names.index(index[1])]
        # This index is accessing a row by integer index.
        elif (type(index) == int):
            if (self.view): return Data.Row(self, self.data[self.row(index)])
            else:           return self.data[index]
        # This index is accessing a column
        elif (type(index) == str):
            if (index not in self.names):
                raise(Data.UnknownName(f"This data does not have a column named '{index}'."))
            # If this is a view, make that numerical index relative to this.
            if (self.view):
                col_names = list(self.names) # <- use the iterator because it grabs relevant columns
                if index not in col_names:
                    raise(Data.UnknownName(f"This data view does not have a column named '{index}'."))
                num_index = col_names.index(index)
            # Otherwise, retreive the exact numerical column index.
            else:
                num_index = self.names.index(index)
            # Return a mutable "Column" object
            return Data.Column(self, num_index)
        # Short-circuit for recognizing a "Column" object.
        if ((type(index) == tuple) and (len(index) == 2)
            and (type(index[0]) != str) and (type(index[1]) in {int, str})):
            col_index = index[1]
            if (type(col_index) == str):
                # If this is a view, make that numerical index relative to this.
                if (self.view):
                    col_names = list(self.names) # <- use the iterator because it grabs relevant columns
                    if col_index not in col_names:
                        raise(Data.UnknownName(f"This data view does not have a column named '{col_index}'."))
                    col_index = col_names.index(col_index)
                # Otherwise, retreive the exact numerical column index.
                else:
                    col_index = self.names.index(col_index)
            return Data.Column(self, col_index)[index[0]]
        # If they are extracting the full data then assume a copy is desired.
        elif ((type(index) == slice) and (index == slice(None))):
            return self.copy()
        elif ((type(index) == tuple) and (len(index) == 2) and
              (type(index[0]) == slice) and (index[0] == slice(None)) and 
              (type(index[1]) == slice) and (index[1] == slice(None))):
            return self.copy()
        # Othwerwise this is a special index and we need to convert it.
        else:
            # This index is retrieving a sliced subset of self.
            rows, cols = self._index_to_rows_cols(index)
            # Otherwise, this is a view.
            view = type(self)(data=self.data, names=self.names.values,
                              types=self.types.values, rows=rows, cols=cols)
            # If this is only accessing a single row, return a Row object
            # from a Data view that only looks at the selected columns.
            if (type(index) == tuple) and (len(index) == 2) and (type(index[0]) == int): 
                return view[0]
            # Otherwise return the full view.
            return view


    # Overwrite the standard "[]" get operator to accept strings as well.
    # Given local-specific index, convert to Data index and return value.
    def __setitem__(self, index, value):
        # Special case for being empty.
        if (self.empty):
            if (type(index) == str):
                return self.add_column(value, name=index)
            elif (type(index) == int):
                raise(Data.Empty("Cannot set row for empty data."))
            elif (type(index) == slice):
                raise(Data.Empty("Cannot set slice of empty data."))                
            else:
                raise(Data.Unsupported(f"Data is empty. No support for setting empty Data with index of type {type(index)}."))
        # Special case assignment of new column (*RETURN* to exit function).
        if ((type(index) == str) and (index not in self.names)):
            # Check to see if this is a data view object.
            if (self.view and (len(self) != len(self.data))):
                raise(Data.Unsupported("This data view does not observe all rows and hence does not support column assignment."))
            elif (self.view and (index in self.names.values)):
                raise(Data.ImproperUsage("This is a data view and the provided index conflicts with a column in the original data."))
            elif (self.view):
                raise(Data.Unsupported("\n  This is a view of a Data object.\n  Did you expect this new column to be added to the original data?"))
            # Check to see if the provided value was a singleton.
            singleton = (not is_iterable(value)) or (type(value) == str)
            if (not singleton):
                try:    singleton = len(value) != len(self)
                except: pass
            # If singleton was provided, set the whole column with that value.
            if singleton:
                v = value
                value = (v for i in range(len(self)))
            # Otherwise perform the normal column addition.
            return self.add_column(value, name=index)
        # Get the list of rows and list of columns being assigned.
        rows, cols = self._index_to_rows_cols(index)
        # ------------------------------------------------------------
        #    Determine whether a singleton was provided or not.
        # 
        # Assume if it has an iterator that is not a singleton.
        # Assume that if the value type matches all assigned types it is a singleton.
        singleton = (not is_iterable(value)) or \
                    (all(self.types[c] in {type(value), type(None)} for c in cols))
        # If it has a "length" then use that to verify singleton status.
        if singleton:
            try:    singleton = len(value) != (len(rows)*len(cols))
            except: pass
        # Assume if the "value" is a string, then this is a singleton.
        if (not singleton) and (type(value) == str): singleton = True
        # Assume if the value is a generator and the column(s) it is
        # assigning to are not *all* generators, it is not a singleton.
        if ((type(value) == GENERATOR_TYPE) and
            any(self.types[c] != GENERATOR_TYPE for c in cols)): singleton = False
        # ------------------------------------------------------------
        # Assume singleton status is correct.
        if (not singleton): value_iter = iter(value)
        # Reset type entirely if the new data assigns to the full column,
        # only if this data views the entire column.
        if (len(rows) == len(self.data)):
            for col in cols: self.types[col] = type(None)
        # Iterate over rows and perform assignment.
        for step, row in enumerate(rows):
            if type(row) != int:
                raise(Data.BadIndex(f"The provided row index of type {type(row)}, {row}, is not understood."))
            for col in cols:
                # Retreive next value from iterator if necessary.
                if not singleton:
                    try:    value = next(value_iter)
                    except: raise(Data.BadValue(f"Provided iterable only contained {step+1} elements, expected more."))
                # The existing column doesn't have a type, assign it.
                if (self.types[col] == type(None)):
                    self.types[col] = type(value)
                # Ignore missing values.
                elif (value is None): pass
                # Check against the existing column type (for verification).
                elif (type(value) != self.types[col]):
                    message = f"Provided value {value} of type {type(value)} does not match expected type {self.types[col]}."
                    if (self.view): message = "This Data is a view. " + message
                    raise(Data.BadAssignment(message))
                # Make the assignment
                self.data[row][col] = value
        # Verify that the iterable has been exhausted.
        if (not singleton):
            try:
                value = next(value_iter)
                raise(Data.BadAssignment(f"Column assignment requires a column of length {len(self)}, provided values iterable was too long."))
            except StopIteration: pass


    # Return a short string representation of this object.
    def __repr__(self):
        return self.__str__(short=True)


    # Printout a brief table-format summary of this data.
    def __str__(self, short=False):
        # Special case for being empty.
        if (self.empty): return "This Data has no contents.\n"
        # Custom short string for a 'data' type object.
        if short: return f"Data ({self.shape[0]} x {self.shape[1]}) -- {self.names}"
        # Make a pretty table formatted output
        rows = []
        # ------------------------------------------------------------
        #          MAKE THE FIRST TWO LINES (NAMES AND TYPES)
        # 
        half_max = self.max_display // 2
        # Add the string values of "types" and "names" to the display.
        if (self.shape[1] <= self.max_display):
            rows += [ list(self.names) ]
            rows += [ list(map(lambda t: str(t)[8:-2],self.types)) ]
        else:
            names = []
            types = []
            for c in range(self.max_display):
                if (c == half_max):
                    names.append("...")
                    types.append("...")
                if (c >= half_max): c -= self.max_display
                names.append( str(self.names[c]) )
                types.append( str(self.types[c])[8:-2] )
            rows += [names, types]
        # ------------------------------------------------------------
        #          PRODUCE STRINGS OF STORED DATA TO SHOW
        # 
        # Add some rows to display to the printout.
        for i in range(min(len(self), self.max_display)):
            # If this is more than halfway through max_display, insert
            # a truncation sequence and look at back of this Data.
            if (len(self) > self.max_display) and (i >= half_max):
                if (i == half_max): rows += [["..."]*len(rows[0])]
                i -= self.max_display
            # Get the row.
            row = self[i]
            # Convert values in the row into strings for printing.
            str_row = []
            for c in range(len(row)):
                if c >= self.max_display: break
                elif (self.shape[1] > self.max_display) and (c >= half_max):
                    if (c == half_max): str_row.append("...")
                    c -= self.max_display
                str_row.append( self._val_to_str(row[c]) )
            rows += [ str_row ]
        # ------------------------------------------------------------
        #               MAKE COLUMN WIDTHS UNIFORM
        # 
        # Compute length of each column.
        lens = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        # Redefine all columns by adding spaces to short-columns.
        rows = [[v + " "*(lens[i]-len(v)) for i,v in enumerate(row)]
                for row in rows]
        # Add dividers between the data values (skip truncation rows).
        for r in range(len(rows)):
            if (len(self) > self.max_display) and ((r-2) == half_max):
                rows[r] = " " + ("   ".join(rows[r]))
            else:
                rows[r] = " " + (" | ".join(rows[r]))
        # Add a divider between the first two rows and the data.
        rows.insert(2, "-"*len(rows[0]))
        # Construct the strings for the "view" and "size" descriptions
        view_string = "This data is a view." if self.view else ""
        size_string = f"Size: ({self.shape[0]} x {self.shape[1]})"
        # Construct the string describing missing values.
        missing_string = []
        missing_string_len = 0
        # ------------------------------------------------------------
        #           COMPUTE THE AMOUNT OF MISSING DATA
        # 
        # For small data, everything is checked (quickly).
        if (self.shape[0] * self.shape[1]) <= MISSING_SAMPLE_SIZE:
            sample = self
        # For larger data, use a sample of reasonably small size.
        else:
            from .utilities import random_range
            # Get a sample of rows to check for missing values.
            indices_for_sample = sorted(random_range(len(self),count=MISSING_SAMPLE_SIZE//self.shape[1]))
            sample = self[indices_for_sample,:]
        is_none = lambda v: (v is None)
        missing_rows = {r+1:sum(map(is_none,v)) for (r,v) in enumerate(sample)}
        no_none = [r for r in missing_rows if missing_rows[r] == 0]
        for r in no_none: missing_rows.pop(r)
        # Print some info about missing values if they exist.
        if len(missing_rows) > 0:
            missing_cols = {}
            # Get the row and column indices.
            if (sample._col_indices is None): col_ids = range(self.shape[1])
            else:                             col_ids = sample._col_indices
            if (sample._row_indices is None): row_ids = range(len(self))
            else:                             row_ids = sample._row_indices
            # Cycle through and count the missing values in each column.
            for c_idx, c in enumerate(col_ids):
                missing = 0
                for r in row_ids:
                    if (sample.data[r][c] is None): missing += 1
                if (missing > 0): missing_cols[c_idx] = missing
            # Get some statistics on the missing values.
            row_indices = sorted(missing_rows)
            col_indices = [sample.names[i] for i in sorted(missing_cols)]
            # Extrapolate total missing by using the row as an estimate.
            total_missing_values = sum(missing_rows.values())
            is_full = id(sample) == id(self)
            if (not is_full): total_missing_values = max(1,int(total_missing_values * (len(self)/len(sample))+1/2))
            total_values = self.shape[0] * self.shape[1]
            # --------------------------------------------------------
            #       MAKE A PRETTY PRINTOUT OF MISSING ENTRIES
            # 
            # Print out the indices of the rows with missing values
            # Add elipses if there are a lot of missing values.
            missing_string += [f" missing {'estimated ' if (not is_full) else ''}{total_missing_values} of {total_values} entries"]
            if (not is_full): missing_string[-1] += f" ({100*total_missing_values/total_values:.1f}%)"
            print_indices = ', '.join(map(str, row_indices[:min(self.max_display,len(row_indices))]))
            missing_string += [f"      at rows: [{print_indices}"]
            if (len(row_indices) > self.max_display): missing_string[-1] += ", ..."
            missing_string[-1] += "]"
            # Repeat same information for columns..
            print_indices = ', '.join(map(str, col_indices[:min(self.max_display, len(col_indices))]))
            missing_string += [f"   at columns: [{print_indices}"]
            if (len(col_indices) > self.max_display): missing_string[-1] += ", ..."
            missing_string[-1] += "]"
            missing_string_len = max(map(len, missing_string))
        # ------------------------------------------------------------
        #        ADD HEADER, FOOTER, AND GENERATE FINAL STRING
        # 
        # Construct a header bar out of "=".
        width = max(len(rows[0]), len(size_string),
                    len(view_string), missing_string_len)
        horizontal_bar = "="*width
        header = [horizontal_bar]
        if len(view_string) > 0:
            header += [view_string]
        header += [size_string, ""]
        footer = [""] + missing_string + [horizontal_bar]
        # Print out all of the rows with appropriate whitespace padding.
        string = "\n"
        for row in (header + rows + footer):
            string += row  +  " "*(width - len(row))  +  "\n"
        # Return the final string.
        return string


    # Define a convenience funciton for concatenating another
    # similarly typed and named data to self.
    def __iadd__(self, data):
        import time
        start = time.time()
        # Check to see if this is a data view object.
        if (self.view): raise(Data.Unsupported("This is a data alias and does not support in-place addition."))
        # Check for improper usage
        if (id(self) == id(data)): raise(Data.Unsupported("A Data object cannot be added to itself in-place."))
        if type(data) != Data:
            raise(Data.Unsupported(f"In-place addition only supports type {Data}, but {type(data)} was given."))
        # Special case for being empty.
        if data.empty: return self
        # Special case for being empty.
        elif self.empty:
            self.names = data.names
            self.types = data.types
            # Load in the data
            for i,row in enumerate(data):
                # Update user on progress if too much time has elapsed..
                if (time.time() - start) > self.max_wait:
                    print(f" {100.*i/len(data):.2f}% in-place add", **self.print_kwargs)
                    start = time.time()
                self.append(row)
        # Add rows to this from the provided Data
        elif (set(self.names) == set(data.names)):
            # If the names are not in the same order, create a view
            # on the provided data with the correct order.
            if (tuple(self.names) != tuple(data.names)): data = data[self.names]
            # Load in the data
            for i,row in enumerate(data):
                # Update user on progress if too much time has elapsed..
                if (time.time() - start) > self.max_wait:
                    print(f" {100.*i/len(data):.2f}% in-place add", **self.print_kwargs)
                    start = time.time()
                self.append(row)
        # Add columns to this from the provided Data (if number of rows is same).
        elif (len(self) == len(data)) and (set(self.names).isdisjoint(data.names)):
            for col in data.names: self[col] = data[col]
        # Otherwise if a subset of columns are matching, add rows (and new columns).
        elif (set(self.names).issubset(data.names) or set(data.names).issubset(self.names)):
            # First add rows for all the same-named columns.
            other_names = set(data.names)
            shared_names = [n for n in self.names if n in other_names]
            my_col_indices = [self.names.index(n) for n in shared_names]
            for i,row in enumerate(data[:,shared_names]):
                # Update user on progress if too much time has elapsed..
                if (time.time() - start) > self.max_wait:
                    print(f" {100.*i/len(data):.2f}% in-place add (new rows)", **self.print_kwargs)
                    start = time.time()
                # Fill missing values with "None".
                if (len(shared_names) < len(self.names)):
                    # Construct a full row, start with None, fill
                    # values from provided data row.
                    full_row = [None] * len(self.names)
                    for j,v in zip(my_col_indices,row): full_row[j] = v
                    row = full_row
                self.append(row)
            # Second add new columns for all the new names.
            my_names = set(self.names)
            new_names = [n for n in data.names if n not in my_names]
            for i,name in enumerate(new_names):
                # Update user on progress if too much time has elapsed..
                if (time.time() - start) > self.max_wait:
                    print(f" {100.*i/len(new_names):.2f}% in-place add (init cols)", **self.print_kwargs)
                    start = time.time()
                self[name] = None
            for r in range(len(data)):
                # Update user on progress if too much time has elapsed..
                if (time.time() - start) > self.max_wait:
                    print(f" {100.*i/len(data):.2f}% in-place add (fill cols)", **self.print_kwargs)
                    start = time.time()
                self[r-len(data),new_names] = data[r][new_names]
        else: raise(Data.Unsupported("Not sure how to add given data, no shape or name precedents match."))
        # Return self
        return self


    # Define a convenience funciton for concatenating another
    # similarly typed and named data to self.
    def __add__(self, data):
        if self.empty:
            return data
        d = self.copy()
        d += data
        return d


    # Check whether or not the values match a row in data.
    def __contains__(self, row):
        # Use the "equality" operator to determine if something is contained.
        for i in (self == row): return True
        return False


    # Given a row, generate the list of indices in this data that equal the row.
    def __eq__(self, row):
        # Make sure the row object provided is not a generator.
        if (type(row) is type((_ for _ in 'generator'))): row = list(row)
        # Cycle through rows in self.
        for row_idx in range(len(self)):
            # Check to see that all values match.
            for v in (self[row_idx] == row):
                if (v is NotImplemented): break
                elif (v is False): break
            # If the entire row is looped and no problems are encountered, it matches.
            else: yield row_idx


    # Overwrite the "len" operator.
    def __len__(self):
        if (self.view): return len(self._row_indices)
        else:           return len(self.data)


    # Return the index of a 'row' based on its contents.
    def index(self, search_row):
        for i in range(len(self)):
            if all(((v == True) or (v is None))
                   for v in (self[i] == search_row)): return i
        else: raise(ValueError(f"{search_row} not found in this data."))


    # Define the "append" operator using the insert operation. On a
    # list, the  "insert" operation takes about twice as long when
    # placing values at the end, but that speed is sacrificed to
    # remove the extra code.
    def append(self, index, value=None):
        if (value is None): index, value = len(self), index
        # Call the standard append operator, adding the element to self
        return self.insert(index, value)


    # Redefine the "insert" method for this data.
    def insert(self, index, element):
        # Check to see if this is a data view object.
        if (self.view):
            raise(Data.Unsupported("This is a data alias and does not support assignment."))
        # Convert the element into a python list
        try: element = list(element)
        except: raise(Data.BadValue(f"Invalid element, failed conversion to list."))
        # Check length in case types already exists.
        if (self.types is not None):
            if ((self.shape[1] > 0) and (len(element) != self.shape[1])):
                raise(Data.BadElement(f"Only elements of length {self.shape[1]} can be added to this data, received length {len(element)}."))
            # Try type casing the new element
            for i, (val, typ) in enumerate(zip(element, self.types)):
                # Ignore 'None' type values.
                if (val is None): pass
                elif (typ == type(None)):
                    # Update unassigned types with the new values' type
                    self.types[i] = type(val)
                elif (type(val) != typ):
                    # If not missing, then check the type
                    try:
                        # Try casting the element as the expected type
                        element[i] = typ(val)
                    except ValueError:
                        # Otherwise raise an error to the user
                        raise(Data.BadValue(f"Value '{val}' of type '{type(val)}' could not successfully be cast as '{typ}' to match column {i+1}."))
                    except TypeError:
                        print("Data error:", typ, typ==type(None))
        else:
            # Construct the expected "types" of this data
            self.types = [type(val) for val in element]
        if (self.names is None):
            # Construct default names for each column
            self.names = [str(i) for i in range(len(element))]
        # Call the standard insert operator, adding the element to self
        self.data.insert(index, Data.Row(self, element))


    # Make sure the copy of this data is a deep copy.
    def copy(self):
        import time
        start = time.time()
        # Construct a new data set that is a copy of the data in this object.
        if self.view: rows, cols = self._row_indices, self._col_indices
        else:         rows, cols = map(list,map(range,self.shape))        
        data = type(self)(names=self.names, types=self.types)
        for i,row in enumerate(rows):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(self):.2f}% copy", **self.print_kwargs)
                start = time.time()
            # Save time and skip data validation (since we already
            # know that the data was valid inside this object).
            row = Data.Row(data, [self.data[row][col] for col in cols])
            data.data.insert( len(data.data), row )
        # Return a new data object.
        return data


    # Overwrite the 'pop' method.
    def pop(self, index=-1):
        import time
        start = time.time()
        # Special case for being empty.
        if (self.empty): raise(Data.Empty("Cannot pop from empty data."))
        if self.view: raise(Data.Unsupported("Cannot 'pop' from a data view. Copy this object to remove items."))
        # Popping of a row
        if type(index) == int:
            return self.data.pop(index)
        # Popping of a column
        elif (type(index) == str):
            if index not in self.names:
                raise(Data.BadSpecifiedName(f"There is no column named '{index}' in this Data."))
            col = self.names.index(index)
            # Pop the column from "names", from "types" and then
            # return values from the "pop" operation.
            self.names.pop(col)
            self.types.pop(col)
            values = []
            for i,row in enumerate(self):
                # Update user on progress if too much time has elapsed..
                if (time.time() - start) > self.max_wait:
                    print(f" {100.*i/len(self):.2f}% pop", **self.print_kwargs)
                    start = time.time()
                values.append( row.pop(col) )
            return values
        else:
            raise(Data.BadIndex(f"Index {index} of type {type(index)} is not recognized."))


    # Overwrite the "sort" method to behave as expected.
    def sort(self, key=None):
        # If this is a view, sort the row indices.
        if self.view:
            if (key is None): key = lambda row: tuple(row)
            order = list(range(len(self)))
            order.sort(key=lambda i: key(self[i]))
            self._row_indices = [self._row_indices[i] for i in order]
        # Otherwise sort the actual list of rows inside this data.
        else:
            if (key is None): key = lambda row: row.values
            self.data.sort(key=key)


    # ========================
    #      Custom Methods     
    # ========================


    # Custom function for mapping values to strings in the printout of self.
    def _val_to_str(self, v): 
        # Get the string version of the value.
        if not ((type(v) == Data) or (issubclass(type(v),Data))):
            if (type(v) == str): v = '"' + v + '"'
            string = str(v).replace("\n"," ")
        else:
            # Custom string for a 'data' type object.
            string = v.__str__(short=True)
        # Shorten the string if it needs to be done.
        if len(string) > self.max_str_len: 
            string = string[:self.max_str_len]+'..'
        return string


    # Given an index (of any acceptable type), convert it into an
    # iterable of integer rows and a list of integer columns.
    # Input index is in terms of this data, output is in true data.
    def _index_to_rows_cols(self, index):
        # Special case for being empty.
        if (self.empty): raise(Data.Empty("Cannot get rows and cols from empty data."))
        # Standard usage.
        if (type(index) == int):
            rows = [self.row(index)]
            cols = [self.col(i) for i in range(self.shape[1])]
        # Case for column index access.
        elif (type(index) == str):
            if index not in self.names:
                raise(Data.UnknownName(f"This data does not have a column named '{index}'."))
            rows = [self.row(i) for i in range(len(self))]
            cols = [self.names.index(index)]
        # Case for slice-index access.
        elif (type(index) == slice):
            # Construct a new "data" with just the specified rows (deep copy)
            rows = [self.row(i) for i in range(len(self))[index]]
            cols = [self.col(i) for i in range(self.shape[1])]
        # Case for tuple-string index access where thera are two selected columns.
        elif ((type(index) == tuple) and (len(index) == 2) and
              (type(index[0]) == str) and (type(index[1]) == str)):
            rows = [self.row(i) for i in range(len(self))]
            cols = []
            for i in index:
                if i not in self.names:
                    raise(Data.UnknownName(f"This data does not have a column named '{i}'."))
                cols.append( self.names.index(i) )
        # Case where (row indices, col indices) were provided.
        elif ((type(index) == tuple) and (len(index) == 2)):
            if (type(index[0]) == int) and (type(index[1]) == int):
                # This index is accessing a (row, col) entry.
                rows = [self.row(index[0])]
                cols = [self.col(index[1])]
            elif ((type(index[0]) == int) and (type(index[1]) == str)):
                # This index is accessing a (row, col) entry with a named column.
                rows = [self.row(index[0])]
                cols = [self.names.index(index[1])]
            else:
                rows, cols = index

                # Handling the "rows"
                if (type(rows) == int):     rows = [self.row(rows)]
                elif (type(rows) == slice): rows = [self.row(i) for i in range(len(self))[rows]]
                elif is_iterable(rows):
                    rows = list(rows)
                    if   all(type(i)==int for i in rows):  rows = [self.row(i) for i in rows]
                    elif all(type(i)==bool for i in rows):
                        rows = [self.row(i) for (i,b) in enumerate(rows) if b]
                    else:
                        type_printout = sorted(map(str,set(map(type,rows))))
                        raise(Data.BadIndex(f"The provided row index, {index}, is not understood.\n It has types {type_printout}."))
                else:
                    type_printout = sorted(map(str,set(map(type,rows))))
                    raise(Data.BadIndex(f"The provided row index, {index}, is not understood.\n It is typed {type_printout}."))

                # Handling the "columns"
                if (type(cols) == int):     cols = [self.col(cols)]
                elif (type(cols) == slice): cols = [self.col(i) for i in range(self.shape[1])[cols]]
                elif (type(cols) == str):
                    if self.view: names = list(self.names)
                    else:         names = self.names
                    if (cols not in names):
                        raise(Data.UnknownName(f"This data does not have a column named '{cols}'."))
                    cols = [names.index(cols)]
                elif is_iterable(cols):
                    given_cols = list(cols)
                    cols = []
                    for i,v in enumerate(given_cols):
                        if (type(v) not in {str,int,bool}):
                            raise(Data.BadIndex(f"The provided column index of type {type(v)}, {v}, is not understood."))
                        elif (type(v) == str):
                            if self.view: names = list(self.names)
                            else:         names = self.names
                            if (v not in names):
                                raise(Data.UnknownName(f"This data does not have a column named '{v}'."))
                            idx = names.index(v)
                        # If it is an int, use that as the index.
                        elif (type(v) == int): idx = self.col(v)
                        # If it is a boolean and it is False, skip this index.
                        elif (type(v) == bool) and (not v): continue
                        elif (type(v) == bool) and v: idx = self.col(i)
                        # Append the column index
                        cols.append( idx )
                else:
                    type_printout = sorted(map(str,set(map(type,cols))))
                    raise(Data.BadIndex(f"The provided column index, {index}, is not understood.\n It has types {type_printout}."))
        # Iterable index access.
        elif is_iterable(index):
            index = list(index)
            # Case for iterable access of rows.
            if all(type(i)==int for i in index):
                rows = [self.row(i) for i in index]
                cols = [self.col(i) for i in range(self.shape[1])]
            # Case for iterable access of columns.
            elif all(type(i)==str for i in index):
                rows = [self.row(i) for i in range(len(self))]
                cols = []
                for i in index:
                    if i not in self.names:
                        raise(Data.UnknownName(f"This data does not have a column named '{i}'."))
                    cols.append( self.names.index(i) )
            # Case for boolean-index access to rows.
            elif all(type(i)==bool for i in index):
                rows = [self.row(i) for (i,b) in enumerate(index) if b]
                cols = [self.col(i) for i in range(self.shape[1])]
            # Undefined behavior for this index.
            else:
                type_printout = sorted(map(str,set(map(type,index))))
                raise(Data.BadIndex(f"The provided index, {index}, is not understood.\n It is typed {type_printout}."))
        # Undefined behavior for this index.
        else:
            try:    type_printout = sorted(map(str,set(map(type,index))))
            except: type_printout = str(type(index))
            raise(Data.BadIndex(f"\n  The provided index, {index}, is not understood.\n  It is typed {type_printout}."))
        # Return the final list of integer-valued rows and columns.
        return rows, cols


    # =======================================
    #      Saving and Loading from Files     
    # =======================================


    # Convenience method for loading from file.
    def load(path, *args, **read_data_kwargs):
        from .read import read_data
        # Handle two different usages of the 'load' function.
        if (type(path) != str): 
            # If the user called "load" on an initialized data, ignore
            self = path
            if (len(args) == 1): path = args[0]
            else:                raise(Data.ImproperUsage("'load' method for Data must be given a path."))
        else:
            # Otherwise define 'self' as a new Data object.
            self = Data()
        # Check for extension, add default if none is provided.
        if "." not in path: path += ".pkl"
        # Check for compression
        compressed = path[-path[::-1].index("."):] in {"gz"}
        if compressed:
            import gzip, io
            file_opener = gzip.open
            base_path = path[:-path[::-1].index(".")-1]
            ext = base_path[-base_path[::-1].index("."):]
        else:
            file_opener = open
            ext = path[-path[::-1].index("."):]
        # Handle different base file extensions
        if (ext == "pkl"):
            import pickle
            with file_opener(path, "rb") as f:
                if self.empty: self =  pickle.load(f)
                else:          self += pickle.load(f)
        elif (ext == "dill"):
            try:    import dill
            except: raise(Data.MissingModule("Failed to import 'dill' module. Is it installed?"))
            with file_opener(path, "rb") as f:
                if self.empty: self =  dill.load(f)
                else:          self += dill.load(f)
        elif (ext in {"csv", "txt", "tsv"}):
            # Declare the read mode based on file type.
            mode = "r" + ("t" if compressed else "")
            # For big files, only read a random sample of the file by default.
            import os
            is_big = os.path.exists(path) and (os.path.getsize(path) > FILE_SAMPLE_SIZE)
            # If the "sample" is a float, assume its a ratio of lines.
            if ("sample" in read_data_kwargs) and (type(read_data_kwargs["sample"]) == float):
                read_data_kwargs["sample_ratio"] = read_data_kwargs.pop("sample")
            # By default, sample big data files.
            sample_unspecified = ("sample" not in read_data_kwargs)
            if (is_big and sample_unspecified):
                def read_chunks(f, num_bytes=2**20):
                    chunk = f.read(num_bytes)
                    while chunk:
                        yield chunk
                        chunk = f.read(num_bytes)
                # Open the file to read the number of lines.
                with file_opener(path, mode, errors='ignore') as f:
                    lines = sum(chunk.count("\n") for chunk in read_chunks(f))
                # Get the size of the (uncompressed) file.
                with file_opener(path, mode, errors='ignore') as f:
                    import io
                    file_size = f.seek(0, io.SEEK_END)
                # Guess at a sample number by assuming lines are equal
                # sizes, try and grab approximately "FILE_SAMPLE_SIZE".
                sample_ratio = read_data_kwargs.pop("sample_ratio",
                    FILE_SAMPLE_SIZE/file_size )
                sample_size = int(lines * sample_ratio)
                print("WARNING: Found",lines,"lines, randomly sampling",sample_size,"from this file.")
                print("         Use 'sample=None' if you want all lines from this file.")
                read_data_kwargs["sample"] = sample_size
            # Make the read operation verbose for large data files.
            if (is_big and ("verbose" not in read_data_kwargs)):
                read_data_kwargs["verbose"] = True
            # Set the separator based off the file extension if not provided.
            if ("sep" not in read_data_kwargs) and (ext in SEPARATORS):
                read_data_kwargs["sep"] = SEPARATORS[ext]
            # Open the file for an actual read.
            with file_opener(path, mode) as f:
                read_data_kwargs["opened_file"] = f
                if self.empty: self =  read_data(**read_data_kwargs)
                else:          self += read_data(**read_data_kwargs)
        else:
            raise(Data.Unsupported(f"Cannot load file with extension '{ext}'."))
        return self


    # Convenience method for saving to a file, offers automatic
    # creation of output folders (given they are not present).
    # WARNING: This method makes a temporary copy if `self.view`.
    def save(self, path, create=True):
        # Special case for being empty.
        if (self.empty): raise(Data.Empty("Cannot save empty data."))
        # Create the output folder if it does not exist.
        output_folder = os.path.split(os.path.abspath(path))[0]
        if create and not os.path.exists(output_folder): os.makedirs(output_folder)
        # Check for compression
        if "." not in path: path += ".pkl"
        compressed = path[-path[::-1].index("."):] == "gz"
        file_opener = open
        if compressed:
            import gzip
            file_opener = gzip.open
            base_path = path[:-path[::-1].index(".")-1]
            ext = base_path[-base_path[::-1].index("."):]
        else:
            ext = path[-path[::-1].index("."):]
        # Handle different base file extensions
        if (ext == "pkl"):
            import pickle
            with file_opener(path, "wb") as f:
                # Create a copy for the save operation if this is a view.
                if (self.view):
                    temp = self.copy()
                    pickle.dump(temp, f)
                    del temp
                else: pickle.dump(self, f)
        elif (ext == "dill"):
            try:    import dill
            except: raise(Data.MissingModule("Failed to import 'dill' module. Is it installed?"))
            with file_opener(path, "wb") as f:
                # Create a copy for the save operation if this is a view.
                if (self.view):
                    temp = self.copy()
                    dill.dump(temp, f)
                    del temp
                else: dill.dump(self, f)
        elif (ext in SEPARATORS):
            sep = SEPARATORS[ext]
            mode = "w" + ("t" if compressed else "")
            # Remove the seperator character from the 
            fix = lambda v: '"' + v.replace('"',"'") + '"' if sep in v else v
            with file_opener(path, mode) as f:
                print(sep.join(map(fix,self.names)), file=f)
                for row in self:
                    print_row = map(fix,(str(v) if (v is not None) else "" for v in row))
                    print(sep.join(print_row), file=f)
        else:
            raise(Data.Unsupported(f"Cannot save {'compressed ' if compressed else ''}file with base extension '{ext}'."))
        return self


    # ====================================
    #      Reorganizing Existing Data     
    # ====================================


    # Given a (sub)set of column names in this data, reorder the data
    # making those names first (with unprovided names remaining in
    # same order as before).
    def reorder(self, names):
        import time
        start = time.time()
        # Special case for being empty.
        if (self.empty):  raise(Data.Empty("Cannot reorder empty data."))
        elif (self.view): raise(Data.Unsupported("Cannot 'reorder' a data view. Copy this object to modify."))
        # Check for proper usage.
        my_names = set(self.names)
        for n in names:
            if (n not in my_names):
                raise(Data.BadSpecifiedName(f"No column named '{n}' exists in this data."))
        # Construct the full reordering of every row of data.
        order = [self.col(self.names.index(n)) for n in names]
        taken = set(order)
        order += [i for i in range(self.shape[1]) if i not in taken]
        # Re-order names, types, and all rows of data.
        self.names = [self.names[i] for i in order]
        self.types = [self.types[i] for i in order]
        for row in range(len(self)):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*row/len(self):.2f}% reorder", **self.print_kwargs)
                start = time.time()
            # Directly update the row (bypassing validation) because
            # we know all values were already validated.
            self[row].values = [self[row,i] for i in order]


    # Given a new list of types, re-cast all elements of columns with
    # changed types into the newly specified type.
    def retype(self, types, columns=None):
        import time
        start = time.time()
        # Special case for being empty and being a view.
        if (self.empty): raise(Data.Empty("Cannot retype empty data."))
        if (self.view): raise(Data.Unsupported("Cannot retype a data view, either retype original data or copy and then retype."))
        # Handle a single column or type being provided.
        if   (type(columns) == str): columns = [columns]
        elif (type(columns) == int): columns = [self.names[columns]]
        # Initialize "columns" if it was not provided to be the whole Data.
        if (columns is None):     columns = list(range(len(self.names)))
        if (type(types) == type): types = [types] * len(columns)
        elif (len(types) != len(columns)):
            raise(Data.BadSpecifiedType(f"{Data.retype} given {len(types)} types, epxected {len(columns)}."))
        # Verify that all columns are valid names, convert to integers.
        for i in range(len(columns)):
            if (type(columns[i]) == str):
                if (columns[i] not in self.names):
                    raise(Data.BadSpecifiedName(f"No column named '{col}' exists in this data."))
                columns[i] = self.names.index(columns[i])
            elif (type(columns[i]) == int):
                if not (-len(self.names) <= columns[i] < len(self.names)):
                    raise(Data.BadIndex(f"Provided column index {columns[i]} is out of range."))
            else: raise(Data.ImproperUsage(f"Unrecognized column index {columns[i]}."))
        for j,(new_t, c) in enumerate(zip(types, columns)):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*j/len(columns):.2f}% retype", **self.print_kwargs)
                start = time.time()
            old_t = self.types[c]
            if (new_t == old_t): continue
            # Update the stored type
            self.types[c] = new_t
            # Retype all non-missing elements in that column (in place)
            for i in range(len(self)):
                if (self[i,c] is None): continue
                try:
                    self[i,c] = new_t(self[i,c])
                except ValueError:
                    raise(Data.BadSpecifiedType(f"Type casting {new_t} for column {c} is not compatible with existing value '{self[i,c]}' on row {i}."))


    # Given a new column, add it to this Data.
    def add_column(self, column, name=None, index=None, new_type=type(None)):
        import time
        start = time.time()
        # Special case for being empty.
        if (self.empty):
            if (name is None): name = "0"
            self.names = []
            self.types = []
            index = 0
        # Set the default index to add a column to be the end.
        if (index == None): index = self.shape[1]
        # Verify the name.
        if (name is None):
            num = 0
            while (str(num) in self.names): num += 1
            name = str(num)
        elif (type(name) != str):
            raise(Data.BadSpecifiedName(f"Only string names are allowed. Received name '{name}' with type {type(name)}."))
        elif (name in self.names):
            raise(Data.BadSpecifiedName(f"Attempting to add duplicate column name '{name}' that already exists in this Data."))
        # Verify the column type dynamically. Add new values to all rows.
        i = -1
        for i,val in enumerate(column):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/max(1,len(self)):.2f}% add column", **self.print_kwargs)
                start = time.time()
            # Verify valid index first..
            if (self.shape[1] > 1) and (i >= len(self)):
                # Remove the added elements if the length was not right
                for j in range(len(self)): self[j].pop(index)
                # Raise error for too long of a column
                raise(Data.BadData(f"Provided column has at least {i+1} elements, more than the length of this data ({len(self)})."))
            # If this is the first column in the data.
            elif (self.shape[1] == 0):
                self.append([val])
            # Append the value to this row for normal operation.
            else: self[i].insert(index, val)
            # Only add to missing values entry if it's not already there
            if (val is None): pass
            # Capture the new type (type must be right)
            elif (new_type == type(None)): new_type = type(val)
            # Error out otherwise.
            elif (type(val) != new_type):
                # Remove the added elements because a problem was encountered.
                for j in range(i+1): self[j].pop(index)
                # This is a new type, problem!
                raise(Data.BadValue(f"Provided column has multiple types. Original type {new_type}, but '{val}' has type {type(val)}."))
        # Verify the column length
        if (i < len(self)-1):
            # Remove the added elements if the length was not right
            for j in range(i+1): self[j].pop(index)
            # Raise error for too short of a column
            raise(Data.BadData(f"Provided column has length {i+1}, less than the length of this data ({len(self)})."))
        # Set the name and type.
        self.names.insert(index, name)
        self.types.insert(index, new_type)


    # Given a column made of iterables, unpack all elements and make
    # as many columns are necessary to suit the largest length. Make
    # new columns with naming scheme, fill empty slots with None.
    def unpack(self, column):
        # Check for errors.
        if (self.view): raise(Data.Unsupported(f"This Data is a view and cannot be unpacked."))
        if (column not in self.names): raise(Data.BadIndex(f"No column named '{column}' exists in this Data."))
        # Check to see if the column contains things that are iterable.
        if (not any(is_iterable(v) for v in self[column])):
            raise(Data.ImproperUsage(f"No elements of '{column}' support iteration."))
        # Start tracking runtime.
        import time
        start = time.time()
        # Extract the old values from this Data one element at a time,
        # making a list of iterators for each element.
        values = []
        for v in self.pop(column):
            if (v is None):
                values.append(None)
            else:
                # Try iterating over the object, otherwise construct a
                # tuple containing just the first value (since it
                # can't provide iteration.
                try:    values.append( iter(v) )
                except: values.append( iter((v,)) )
        # One element at a time, add columns to this Data.
        idx = 1
        empty_elements = [0]
        while (empty_elements[0] < len(self)):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {idx} inflating..", **self.print_kwargs)
                start = time.time()
            # Look for the next element in each iterable.
            empty_elements[0] = 0
            # Construct a generator that pulls one element from each row.
            def column_generator():
                for i in range(len(values)):
                    if (values[i] is None):
                        empty_elements[0] += 1
                        yield None
                    else:
                        try:
                            yield next(values[i])
                        except StopIteration:
                            empty_elements[0] += 1
                            values[i] = None
                            yield None
            # Add the column using the generator object if it's not empty.
            column_name = column + f' {idx}'
            self.add_column(column_generator(), name=column_name)
            idx += 1
        # Pop out the last added column, because it only contains empty elements.
        self.pop(column_name)


    # Reverse the 'unpack' operation, using the same expected naming scheme.
    def pack(self, name):
        if (self.view): raise(Data.Unsupported(f"This Data is a view and cannot be packed."))
        if all((name != n[:len(name)]) for n in self.names):
            raise(Data.BadIndex(f"No flattened columns by name '{name}' exist in this Data."))
        # Start tracking runtime.
        import time
        start = [time.time()]
        # Identify those columns that need to be cobined into one column.
        to_collapse = [n for n in self.names if n[:len(name)] == name]
        # Sort the keys by their numerical index.
        to_collapse.sort(key=lambda n: int(n[len(name)+1:]))
        def row_generator():
            for i,row in enumerate(zip(*(self.pop(col) for col in to_collapse))):
                # Update user on progress if too much time has elapsed..
                if (time.time() - start[0]) > self.max_wait:
                    print(f" {i+1}:{len(self)} packing..", **self.print_kwargs)
                    start[0] = time.time()
                # Find the location of the last None value in this row.
                for last_none in range(1,len(row)+1):
                    if (row[-last_none] is not None): break
                # If there are *only* None values, yield None.
                else:
                    yield None
                    continue
                # If the entire row is made of None, then return None.
                if (last_none > 1): yield list(row[:1-last_none])
                else:               yield list(row)
        # Pop out all of the old columns and add one new column.
        self.add_column(row_generator(), name=name)


    # Given a list of column names, modify this Data so that all
    # specified column names are stacked in lists associated with
    # unique combinations of unspecified column names.
    def stack(self, columns):
        if (self.view): raise(Data.Unsupported("Cannot 'stack' a data view, either 'stack' original data or copy and then 'stack'."))
        from .utilities import hash
        import time
        start = time.time()
        # Adjust for usage (where user provides only one column).
        if (type(columns) != list): columns = [columns]
        # Verify all of the provided columns.
        for i in range(len(columns)):
            if   (type(columns[i]) == int): columns[i] = self.names[i]
            elif (type(columns[i]) == str):
                if (columns[i] not in self.names):
                    raise(Data.BadIndex(f"There is no column named '{columns[i]}' in this data."))
            else:
                raise(Data.BadIndex("The index '{columns[i]}' is not recognized. Only {int} and {str} are allowed."))
        # Create a view of the columns that will be kept for hashing.
        keep_columns = [i for (i,n) in enumerate(self.names) if n not in columns]
        keep_view = self[:,keep_columns]
        # Get all columns that will be stacked and rename them.
        stacked_columns = [n for n in self.names if n in columns]
        for i in map(self.names.index, stacked_columns):
            self.names[i] += " unstacked"
        old_col_names = [n + " unstacked" for n in stacked_columns]
        # Get the indices of the old and new columns.
        old_col_idxs = [self.names.index(n) for n in old_col_names]
        # Track the first occurrence of each 
        lookup = {}
        to_pop = []
        for i in range(len(self)):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(self):.2f}% stack", **self.print_kwargs)
                start = time.time()
            # Hash the non-stacked columns of this row.
            hashed = hash(list(keep_view[i]))
            # hashed = hash(list(self[i]))
            if hashed in lookup:
                row, stack = lookup[hashed]
                to_pop.append(i)
            else:
                row, stack = len(lookup), {c:list() for c in stacked_columns}
                lookup[hashed] = (row, stack)
            # Copy the values from the row into the stacks.
            for (new_c, old_c) in zip(stacked_columns, old_col_idxs):
                stack[new_c].append( self[i].values[old_c] )
        # Pop out all of the now-useless rows.
        for i in to_pop[::-1]:
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(self):.2f}% stack - pop rows", **self.print_kwargs)
                start = time.time()
            self.pop(i)
        # Pop out all of the old columns.
        for c in old_col_names: self.pop(c)
        # Insert all of the stacked values into new columns.
        new_stacks = sorted(lookup.values(), key=lambda i: i[0])
        for i,c in enumerate(stacked_columns):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(stacked_columns):.2f}% stack - add stacks", **self.print_kwargs)
                start = time.time()
            self[c] = (stack[c] for (_,stack) in new_stacks)


    # Undo the "stack" operation. Given columns that contain
    # iterables, unstack them by duplicating all unspecified columns
    # and creating new entries in the data.
    def unstack(self, columns):
        if (self.view): raise(Data.Unsupported("Cannot 'unstack' a data view, either 'unstack' original data or copy and then 'unstack'."))
        import time
        from .utilities import hash
        start = time.time()
        # Adjust for usage (where user provides only one column).
        if (type(columns) != list): columns = [columns]
        # Verify all of the provided columns.
        for i in range(len(columns)):
            if   (type(columns[i]) == int): columns[i] = self.names[i]
            elif (type(columns[i]) == str):
                if (columns[i] not in self.names):
                    raise(Data.BadIndex("There is no column named '{columns[i]}' in this data."))
            else:
                raise(Data.BadIndex("The index '{columns[i]}' is not recognized. Only {int} and {str} are allowed."))
        # Get all columns that will be stacked and rename them.
        unstacked_columns = [n for n in self.names if n in columns]
        for i in map(self.names.index, unstacked_columns):
            self.names[i] += " stacked"
        old_col_names = [n + " stacked" for n in unstacked_columns]
        # Initialize all stacked columns to hold lists of values.
        for n in unstacked_columns: self[n] = (None for i in range(len(self)))
        # Get the indices of the old and new columns.
        new_col_idxs = [self.names.index(n) for n in unstacked_columns]
        # Unstack all 
        i = 0
        while (i < len(self)):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(self):.2f}% unstack", **self.print_kwargs)
                start = time.time()
            not_none = lambda v: (v is not None)
            # Retreive the column values in a dictionary.
            try:    values = {c:list(self[i,c]) for c in old_col_names if not_none(self[i,c])}
            except: raise(Data.BadData("Could not convert stack into list. {self[i]}"))
            # Check to make sure the lengths are all the same.
            stack_size = set(map(len,values.values()))
            if (len(stack_size) > 1):
                raise(Data.BadData(f"Row contains stacks of different length. {self[i]}"))
            # If there are no stacked values, remove this row and continue.
            elif (len(stack_size) == 0):
                i += 1
                continue
            stack_size = stack_size.pop()
            # Loop over the stack size
            for step in range(stack_size-1):
                # Populate a new row.
                new_row = list(self[i])
                for (new_c, old_c) in zip(new_col_idxs, old_col_names):
                    new_row[new_c] = values[old_c].pop(-1)
                # Insert the new row.
                self.insert(i+1, new_row)
            # Increment if the stack size is zero.
            if (stack_size == 0): i += 1
            # Otherwise perform the last unstack operation.
            else:
                for (new_c, old_c) in zip(new_col_idxs, old_col_names):
                    self[i,new_c] = values[old_c].pop(-1)
                i += stack_size
        # Pop out all of the old columns.
        for c in old_col_names: self.pop(c)


    # Given a Data that has at least one column with the same names as
    # a column in self, collect all those values in non-mutual names
    # in lists of values where the shared columns had the same content.
    # 
    # Example:
    #   Combined with "unique", this function can be used to convert a
    #   Data object where some rows are repeats of each other having unique
    #   values in only one column to a Data object with only unique
    #   rows and one column contains lists of values associated with
    #   each unique row.
    # 
    #     data.unique([group-by columns]).copy().collect(data)
    # 
    #   The above is equivalent to the in-place version:
    # 
    #     data.stack([columns to stack])
    # 
    #   However, the collect method can be passed any data object,
    #   which may make it more flexible for dynamic data collection.
    def collect(self, data):
        import time
        from .utilities import hash
        # Get the names of shared columns and indices in each.
        match_names = set(n for n in self.names if n in data.names)
        self_columns = [i for (i,n) in enumerate(self.names) if n in match_names]
        # Sort the columns of self to be in the same order as that of other.
        self_columns = sorted(self_columns, key=lambda i: data.names.index(self.names[i]))
        other_columns = [i for (i,n) in enumerate(data.names) if n in match_names]
        # Check for duplicate column names.
        if (len(self_columns) != len(other_columns)):
            raise(Data.BadData("The provided Data to collect has duplicate column names."))
        # Collection columns
        names_to_collect = [n for n in data.names if n not in match_names]
        indices_to_collect  = [i for (i,n) in enumerate(data.names) if n not in match_names]
        collections = {n:[] for n in names_to_collect}
        # Add columns for holding the matches
        for n in names_to_collect:
            self.add_column(([] for i in range(len(self))), name=n)
        # Generate a set of lookups (hashed values even for mutable types)
        index_of_row = {}
        for i,row in enumerate(self):
            hash_value = hash([row[c] for c in self_columns])
            index_of_row[hash_value] = i
        # Paired list of source -> destination indices for copying.
        source_dest = list(zip(indices_to_collect, names_to_collect))
        start = time.time()
        # Perform the serach process
        for i,row in enumerate(data):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(data):.2f}% collect", **self.print_kwargs)
                start = time.time()
            # Convert row into its hash value, lookup storage location.
            hash_value = hash([row[c] for c in other_columns])
            if hash_value in index_of_row:
                row_idx = index_of_row[hash_value]
                for source, dest in source_dest:
                    self[row_idx, dest].append(row[source])
            else:
                print("Could not find:", [row[c] for c in other_columns])
        # Return the self (that has been modified)
        return self


    # =======================================
    #      Converting to pure matrix
    # =======================================


    # Generate a pair of functions. The first function will map rows
    # of Data to a real-valued list. The second function will map
    # real-valued lists back to rows in the original data object.
    def _generate_numeric_mapping(self, max_categories=float('inf')):
        if self.empty: raise(Data.ImproperUsage("Cannot map empty Data."))
        import time
        from .utilities import regular_simplex
        start = time.time()
        # Get those categoricals that need to be converted to reals
        num_cols = [i for (i,t) in enumerate(self.types)
                    if (t in {float, int})]
        num_values = {i:set() for i in num_cols}
        num_means = [0.0 for i in num_cols]
        num_counts = [0 for i in num_cols]
        cat_cols = [i for (i,t) in enumerate(self.types)
                    if (t not in {float, int})]
        cat_value_counts = {i:{} for i in cat_cols}
        # Iterate over self getting all unique hash values for non-numeric columns.
        for i,row in enumerate(self):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(self):.2f}% generating mapping", **self.print_kwargs)
                start = time.time()
            # Update the counts of the unique values.
            for i in cat_cols:
                # Store the counts.
                v = row[i]
                cat_value_counts[i][v] = cat_value_counts[i].get(v,0) + 1
            # Check for new numeric values in numeric columns.
            for j,i in enumerate(num_cols):
                if (i in num_values):
                    num_values[i].add(row[i])
                    if (len(num_values[i]) > 1):
                        num_values.pop(i)
                # If the value is not None, update running mean.
                if (row[i] is not None):
                    num_counts[j] += 1
                    num_means[j] += (row[i] - num_means[j]) / num_counts[j]
        # Find columns with only one unique value, they will be dropped.
        to_drop = {j for j in num_values} | {
            i for i in cat_value_counts if len(cat_value_counts[i]) <= 1}
        # Compute the real vector mappings for all column values.
        mappings = {}
        for i in cat_cols:
            encodings = regular_simplex(min(len(cat_value_counts[i]), max_categories))
            counts = sorted(cat_value_counts[i].items(), key=lambda v:-v[1])
            col_map = {}
            for c,(value,count) in enumerate(counts):
                if (c < max_categories-1):
                    col_map[value] = list(encodings[c])
                else:
                    col_map[EXTRA_CATEGORY_KEY] = list(encodings[-1])
            # Store the value map for this column.
            mappings[self.names[i]] = col_map
        # Store the mapped value for None types in numeric columns.
        for j,i in enumerate(num_cols):
            mappings[self.names[i]] = {None:[num_means[j]]}
        # Compute the names and output indices of numeric and categorical columns.
        groups = {}
        real_names = []
        num_inds = []
        cat_inds = []
        counter = 0
        for i,n in enumerate(self.names):
            groups[n] = []
            if (i in to_drop):
                continue
            elif (i in cat_cols):
                for i in range(len(mappings[n])-1):
                    real_names.append( f"{n}-{i+1}" )
                    cat_inds.append( counter )
                    groups[n].append( counter )
                    counter += 1
            elif (i in num_cols):
                real_names.append( n )
                # Record the indices of numerical values
                num_inds.append( counter )
                groups[n].append( counter )
                counter += 1
        # Return the two mapping functions and some info
        return mappings, groups, to_drop, real_names, num_inds, cat_inds


    # Convert this Data automatically to a real-valued array.
    # Return real-valued array, function for going to real from
    # elements of this Data, function for going back to elements of
    # this data from a real vector.
    def to_matrix(self, max_categories=float('inf'), print_column_width=70):
        import time
        start = time.time()
        # Get the information required to construct a numeric object.
        mappings, groups, dropped, real_names, num_inds, cat_inds = (
            self._generate_numeric_mapping(max_categories) )
        # Place all data into a specialized container and return.
        from .numeric import Numeric
        numeric = Numeric(
            (self.shape[0], len(real_names)),
            list(self.names),
            list(self.types),
            mappings,
            groups,
            dropped,
            real_names,
            num_inds,
            cat_inds,
            print_column_width,
        )
        # Convert all rows in this data matrix to numeric format.
        for i,row in enumerate(self):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(self):.2f}% to matrix", **self.print_kwargs)
                start = time.time()
            # How to handle rows with missing values? Add column for "is missing"?
            numeric[i,:] = numeric.to_real(row)
        # Update the shift and scale variables for the numeric object.
        numeric.compute_shift_and_scale()
        # Return the container object
        return numeric


    # ==============================
    #      Statistical utilities
    # ==============================


    # Return an iterator that provides "k" {(k-1)/k, 1/k} paired Data
    # from this Data. Randomly shuffle indices with seed "seed".
    def k_fold(self, k=10, seed=0, only_indices=False):
        # Generate random indices
        import random
        old_state = random.getstate()
        random.seed(seed)
        indices = list(range(len(self)))
        random.shuffle(indices)
        # Reset the random number generator as not to muck with
        # other processes that might be using it.
        random.setstate(old_state)
        # Store some variables for generating the folds
        total = len(self)
        if (type(self) != Data): only_indices = True
        if not only_indices:     source = self
        for batch in range(k):
            # Find the indices for training and testing
            first = int(.5 + batch * total / k)
            last = int(.5 + (batch + 1) * total / k)
            fold = indices[first : last]
            rest = indices[:first] + indices[last:]
            if only_indices: yield rest, fold
            else:            yield source[rest], source[fold]


    # Compute the pairwise effects between columns. Use the following:
    # 
    #    number vs. number     -- Correlation coefficient between the two sequences.
    #    category vs. number   -- "method" 1-norm difference between full distribution
    #                             and conditional distributions for categories.
    #    category vs. category -- "method" total difference between full distribution
    #                             of other sequence given value of one sequence.
    # 
    # Print out the sorted table of pairwise effects between columns,
    # showing the highest effects first, the smallest last.
    def effect(self, compare_with=None, method="mean", display=True, **kwargs):
        import time
        from itertools import combinations
        from .utilities import effect
        if (compare_with is None):         compare_with = set(self.names)
        elif (type(compare_with) == list): compare_with = set(compare_with)
        else:                              compare_with = {compare_with}
        # Record the starting time (for logging).
        start = time.time()
        # Make sure all names provided are valid.
        for n in compare_with:
            if n not in self.names:
                raise(Data.BadIndex(f"No column named '{n}' exists in this Data."))
        # Print out the effect in column form (not matrix form).
        effs = []
        # Return `True` if the column has more than one unique element.
        def has_more_than_one_unique_element(column):
            from .utilities import hash
            values = set()
            for el in column:
                values.add(hash(el))
                if (len(values) > 1):
                    return True
            return False
        # Get the names that have more than one unique element.
        names_with_variance = {n for n in self.names if has_more_than_one_unique_element(self[name])}
        # Cycle over all combinations of names and compute the effects.
        all_pairs = combinations(self.names, 2)
        for i, (col_1, col_2) in enumerate(all_pairs):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(all_pairs):.2f}% effect ({col_1} - {col_2})..  ", **self.print_kwargs)
                start = time.time()
            # If either column only has 1 unique element, then there is no effect.
            if ((col_1 not in names_with_variance) or
                (col_2 not in names_with_variance)):
                if (col_1 in compare_with):
                    effs.append( (col_1, col_2, 'N/A') )
                elif (col_2 in compare_with):
                    effs.append( (col_2, col_1, 'N/A') )
            # Otherwise, compute the effects.
            elif (col_1 in compare_with):
                eff = effect(list(self[col_1]), list(self[col_2]), method=method, **kwargs)
                effs.append( (col_1, col_2, eff) )
            elif (col_2 in compare_with):
                eff = effect(list(self[col_1]), list(self[col_2]), method=method, **kwargs)
                effs.append( (col_2, col_1, eff) )
        # Convert an effect to a sortable number (lowest most important).
        def to_num(eff): 
            if   type(eff) == str:  return 0.0
            elif type(eff) == dict: return -sum(eff.values()) / len(eff)
            else:                   return -abs(eff)
        # Convert an effect to a printable string.
        def to_str(eff):
            if type(eff) == dict:
                eff_str = []
                for key in sorted(eff, key=lambda k: -abs(eff[k])):
                    eff_str += [f"'{key}':{eff[key]:5.2f}"]
                return f"{-to_num(eff):5.2f}  {{"+", ".join(eff_str)+"}"
            else:
                return f"{eff:5.2f}"
        # Sort the data by magnitude of effect. (most effected -> first)
        effs = sorted(effs, key=lambda row: to_num(row[-1]))
        # Do a quick return without printing if display is turned off.
        if (not display): return effs
        # Print out the effects between each column nicely.
        max_len_1 = max( max(map(lambda i:len(i[0]), effs)), len("Column 1") )
        max_len_2 = max( max(map(lambda i:len(i[1]), effs)), len("Column 2") )
        header = f"{'Column 1':{max_len_1}s}  |  {'Column 2':{max_len_2}s}  |  Effect"
        rows = []
        for (col_1, col_2, eff) in effs:
            eff = to_str(eff)
            rows += [f"{col_1:{max_len_1}s}  |  {col_2:{max_len_2}s}  |  {eff}"]
        row_len = max(len(header), max(map(len, rows)))
        rows = ["", '-'*row_len, header, '-'*row_len] + rows + ['-'*row_len, ""]
        for row in rows: print(row)
        # Return the sorted effect triple list [(col 1, col 2, effect)].
        return effs


    # ============================
    #      Summarizing values
    # ============================


    # Generate a "view" on this data object that only has the first
    # occurrence of its unique rows and return it.
    def unique(self):
        import time
        from .utilities import hash
        # Cycle rows, finding unique ones.
        rows = []
        found = set()
        start = time.time()
        for i,row in enumerate(self):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(self):.2f}% unique", **self.print_kwargs)
                start = time.time()
            # Get the hashed value by consecutively hashing all of the
            # values within this row. Hashing the entire row sometimes
            # produces changing hashes (likely based on memory locations).
            hashed = hash(list(row))
            if hashed not in found:
                found.add(hashed)
                rows.append(i)
        return self[rows]


    # Collect the dictionaries of unique values (with counts) for each column.
    def counts(self, columns=None):
        import time
        from .utilities import hash
        start = time.time()
        if (columns is None): columns = self.names
        column_info = {n:{} for n in columns}
        for i,row in enumerate(self):
            # Update user on progress if too much time has elapsed..
            if (time.time() - start) > self.max_wait:
                print(f" {100.*i/len(self):.2f}% counts", **self.print_kwargs)
                start = time.time()
            for n,val in zip(columns, row):
                # If this column has been removed (because it is
                # unhashable), then skip it in the processing
                if n not in column_info: pass
                # Try to add the new value
                try:
                    column_info[n][val] = column_info[n].get(val,0) + 1
                except TypeError:
                    # Cannot hash an element of self[n], disable tracking
                    column_info.pop(n)
        return column_info


    # Give an overview (more detailed than just "str") of the contents
    # of this data. Useful for quickly understanding how data looks.
    def summarize(self, max_display=None, file=sys.stdout):
        # Define a new print function that redirects to the provided file.
        def print_to_file(*args, **kwargs):
            kwargs['flush'] = True
            kwargs['file'] = file
            return print(*args, **kwargs)
        # Special case for an empty data
        if len(self) == 0:
            print_to_file(self)
            return
        # Set the "max_display" to the default value for this class
        if (max_display is None): max_display = self.max_display
        num_rows, num_cols = self.shape
        print_to_file(f"SUMMARY:")
        print_to_file()
        print_to_file(f"  This data has {num_rows} row{'s' if num_rows != 1 else ''}, {num_cols} column{'s' if num_cols != 1 else ''}.")
        num_numeric = sum(1 for t in self.types if t in {float,int})
        print_to_file(f"    {num_numeric} column{'s are' if num_numeric != 1 else ' is'} recognized as numeric, {num_cols-num_numeric} {'are' if (num_cols-num_numeric) != 1 else 'is'} categorical.")
        # Get some statistics on the missing values.
        is_none = lambda v: (v is None)
        missing_rows = {r:sum(map(is_none,v)) for (r,v) in enumerate(self) if (None in v)}
        if len(missing_rows) > 0:
            missing_cols = {c:sum(map(is_none,self[n])) for (c,n) in enumerate(self.names) if (None in self[n])}
            row_indices = sorted(missing_rows)
            col_indices = [self.names[i] for i in sorted(missing_cols)]
            total_missing_values = sum(missing_rows.values())
            total_values = num_rows * num_cols
            print_to_file(f"    {len(missing_rows)} row{'s have' if len(missing_rows) != 1 else ' has'} missing values.")
            print_to_file(f"    {len(missing_cols)} column{'s have' if len(missing_cols) != 1 else ' has'} missing values.")
            print_to_file(f"    in total, {total_missing_values} of all {total_values} are missing.")
        print_to_file()
        print_to_file("COLUMNS:")
        print_to_file()
        name_len = max(map(len, self.names))
        type_len = max(map(lambda t: len(str(t)), self.types))
        count_string_len = len(str(len(self)))
        # Describe each column of the data
        for c,(n,t) in enumerate(zip(self.names, self.types)):
            # Count the number of elements for each value
            counts = {}
            to_string = False
            for val in self[n]:
                if to_string: val = str(val)
                try: counts[val] = counts.get(val,0) + 1
                except TypeError:
                    to_string = True
                    val = str(val)
                    counts[val] = counts.get(val,0) + 1
            print_to_file(f"  {c:{len(str(self.shape[1]))}d} -- \"{n}\"{'':{1+name_len-len(n)}s}{str(t):{type_len}s} ({len(counts)} unique value{'s' if (len(counts) != 1) else ''})")
            # Identify the length of the longest (string for a) value.
            val_len = max(map(lambda v: len(str(v)), counts))
            val_len = min(val_len, self.max_str_len)
            # Remove the "None" count from "counts" to prevent sorting problems
            none_count = counts.pop(None, 0)
            # For the special case of ordered values, reduce to ranges
            if (t in {int,float}) and (len(counts) > max_display):
                # Print out the count of None values.
                if (none_count > 0):
                    perc = 100. * (none_count / len(self))
                    print_to_file(f"    None                   {none_count:{count_string_len}d} ({perc:5.1f}%) {'#'*round(perc/2)}")
                # Order the values by intervals and print
                min_val = min(counts)
                max_val = max(counts)
                width = (max_val - min_val) / (max_display-1)
                for i in range(max_display-1):
                    lower = min_val + width*i
                    upper = min_val + width*(i+1)
                    if (i == (max_display - 2)):
                        num = sum(counts[v] for v in counts if lower <= v <= upper)
                        cap = "]"
                    else:
                        num = sum(counts[v] for v in counts if lower <= v < upper)
                        cap = ")"
                    perc = 100. * (num / len(self))
                    print_to_file(f"    [{lower:9.2e}, {upper:9.2e}{cap} {num:{count_string_len}d} ({perc:5.1f}%) {'#'*round(perc/2)}")
            else:
                if t in {int, float}:
                    # Order the values by their inate ordering
                    ordered_vals = sorted(counts)
                else:
                    # Order the values by their frequency and print
                    ordered_vals = sorted(counts, key=lambda v: -counts[v])
                if (t == str): val_len += 2
                if (none_count > 0):
                    perc = 100. * (none_count / len(self))
                    print_to_file(f"    {'None':{val_len}s}{'  ' if (t == str) else ''} {none_count:{count_string_len}d} ({perc:5.1f}%) {'#'*round(perc/2)}")
                for val in ordered_vals[:max_display]:
                    perc = 100. * (counts[val] / len(self))
                    if (t == str):
                        num_spaces = max(1 + val_len - len(str(val)), 0)
                        hash_bar = "#"*round(perc/2)
                        count = counts[val]
                        val = val + " "*num_spaces
                        if (len(val) > self.max_str_len): val = val[:self.max_str_len]+".."
                        print_to_file(f'    "{val}"{count:{count_string_len}d} ({perc:5.1f}%) {hash_bar}')
                    else:
                        hash_bar = "#"*round(perc/2)
                        count = counts[val]
                        val = self._val_to_str(val)
                        if (len(val) > self.max_str_len): val = val[:self.max_str_len]+".."
                        print_to_file(f"    {val:{val_len}s} {count:{count_string_len}d} ({perc:5.1f}%) {hash_bar}")
                if (len(ordered_vals) > max_display):
                    print_to_file("    ... (increase 'max_display' to see more summary statistics).")
            print_to_file()

