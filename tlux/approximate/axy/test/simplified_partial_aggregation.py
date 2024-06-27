# This file is for testing the logic of partial aggregation, how the evaluation
#  is done, and most importantly how the gradient is calculated.

# Define a function for mapping integers to letters.
# The sequence created should look like:
#   0 -> A
#   1 -> B
#   ...
#   26 -> AA
#   27 -> AB
#   ..
#   52 -> BA
#   53 -> BB
def to_letter(number):
    letters = ""
    while number >= 0:
        remainder = number % 26
        letters = chr(ord("A") + remainder) + letters
        number = number // 26 - 1
    return letters

# Define a simple class that we can use to track what unique values
#  exist in this computation graph, the parents, children, and
#  the gradients. Includes methods for operations such as:
# 
#   - addition with another Value object
#   - division by another Value object
#   - setting a Number as the current contents
#   - string formatting of the computation defined by Value
#   - initializing with a default and unique letter name,
#     the letter is used to reference each instance when
#     displaying analytically computed values
#   - gradient evaluation with respect to a subset of children
# 
class Value:
    _content = None
    stop = False
    instances = []
    def __init__(self, content=None, letter=None):
        self.parents = []
        self.children = []
        self.operation = None
        if (letter is None):
            letter = to_letter(len(Value.instances))
        self.letter = letter
        self.content = content
        # Add this to list of global instances.
        Value.instances.append(self)

    # Define a "content" property that provides 
    @property
    def content(self):
        return (self if (self._content is None) else self._content)
    @content.setter
    def content(self, content):
        if (content is None):
            pass
        elif (not isinstance(content, (int, float))):
            raise(ValueError(f"{type(self).__name__}.content can only be assigned as an 'int' or 'float', but got '{type(content).__name__}'."))
        self._content = content

    # Return either the letter for self (if undefined) or the literal value.
    def __repr__(self):
        if (self._content is None):
            return self.letter
        else:
            return repr(self._content)

    # Return a nicely formatted string representation of how this Value is computed.
    def __str__(self):
        # If this is a "stop" variable, do not traverse through parents.
        if (self.stop):
            return repr(self)
        # Otherwise, get the full definition of this variable.
        if (self.operation == "+"):
            left = self.parents[0]
            right = self.parents[1]
            if ((left.content != 0) and (right.content != 0)):
                left = str(left)
                right = str(right)
                if ((left[0] == "(") and (left[-1] == ")")):
                    left = left[1:-1]
                if ((right[0] == "(") and (right[-1] == ")")):
                    right = right[1:-1]
                result = f"{left} {self.operation} {right}"
                if (len(self.children) > 0):
                    result = "(" + result + ")"
                return result
            elif (left.content != 0):
                return str(left)
            elif (right.content != 0):
                return str(right)
            else:
                return ""
        elif (self.operation == "/"):
            left = self.parents[0]
            right = self.parents[1]
            if (left.content == 0):
                return "0"
            elif (right.content == 0):
                return "NaN"
            elif (right.content == 1):
                return str(left)
            else:
                result = f"{left} {self.operation} {right}"
                if (len(self.children) > 0):
                    result = "(" + result + ")"
                return result
        else:
            return repr(self)

    # Add another Value to self.
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        result = Value()
        result.parents = [self, other]
        result.operation = "+"
        other.children.append(result)
        self.children.append(result)
        return result
    def __radd__(self, other):
        return self.__add__(other)

    # Divide self by another Value.
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        result = Value()
        result.parents = [self, other]
        result.operation = "/"
        self.children.append(result)
        other.children.append(result)
        return result

    # Evaluate the experession defined by this Value.
    def __call__(self):
        if (self.operation == "+"):
            return self.parents[0].content + self.parents[1].content
        elif (self.operation == "/"):
            return self.parents[0].content / self.parents[1].content
        else:
            return self.content

    # Compute a new Value that is the gradient defined in terms of children.
    # Example of expected behavior is as follows:
    #   D = (A + B + C) / 3
    #   E = (A + B) / 2
    #   F = A
    # 
    #   C.gradient = D / 3
    #   B.gradient = D / 3 + E / 2
    #   A.gradeitn = D / 3 + E / 2 + F
    # 
    def gradient(self):
        # If there are no children, this is a leaf. Return a new Value with the
        #  same letter representation (returning 'self' could have side effects).
        if len(self.children) == 0:
            return Value(letter=self.letter)
        # Otherwise, this is an internal node that has children.
        else:
            # Initialize the gradient to 0.
            gradient = Value(0)
            for child in self.children:
                if child.operation == "+":
                    # Sums just add to the gradient for the parent.
                    gradient += child.gradient()
                elif child.operation == "/":
                    numerator, denominator = child.parents
                    if (numerator == self):
                        # The derivative of (a/b) with respect to (a) is (1/b).
                        gradient += child.gradient() / denominator
                    else:
                        # The derivative of (a/b) with respect to (b) is (-a/(b^2)).
                        gradient -= (child.gradient() * numerator) / (denominator * denominator)
                else:
                    raise ValueError(f"Unrecognized operation '{child.operation}', cannot compute gradient.")
            return gradient



# Declare the number of elements being aggregated.
n = 3

# Create a holder for the raw values.
inputs = [Value() for i in range(n)]

# Perform the partial aggregation calculation using
#   a range that iterates from [1, ..., n].
partials = [sum(inputs[-i:]) / i for i in range(1, n+1)]

# Print the computation graph for each of the partials.
print("Partials:")
for p in partials:
    value = str(p)
    p.stop = True
    print("", p, "=", value)

print()
print("Gradients:")
for i in inputs:
    print("", i.letter, "=", i.gradient())

