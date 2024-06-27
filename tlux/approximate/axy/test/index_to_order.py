# The index-to-pair pattern matches the following.
# 
#   2 elements -> (1,1) (1,2) (2,1) (2,2)
#                   1     1     1     2
#                  1     1     2     2
#                    1     2     1     2
#   3 elements -> (1,1) (1,2) (2,1) (1,3) (3,1) (2,2) (2,3) (3,2) (3,3)
#                   1     1     1     1     1     2     2     2     3
#                  1     1     2     1     3     2     2     3     3
#                    1     2     1     3     1     2     3     2     3
#   4 elements -> (1,1) (1,2) (2,1) (1,3) (3,1) (1,4) (4,1) (2,2) (2,3) (3,2) (2,4) (4,2) (3,3) (3,4) (4,3) (4,4)
#                   1     1     1     1     1     1     1     2     2     2     2     2     3     3     3     4
#                  1     1     2     1     3     1     4     2     2     3     2     4     3     3     4     4
#                    1     2     1     3     1     4     1     2     3     2     4     2     3     4     3     4
# 
# From the back, the number of elements such that the min is i goes:
#    1  3  5  7 ... (1 + 2 <num-others>)
# 
# so given there are n unique elements, the number of elements with min(i) forward is:
#   (1+2n)  (1+2(n-1))  ...  3  1
# 
#  n - (i-n) / 2 - 1
# 
# To consider:
#   Perhaps the pairing function should consider all pairs up to
#   the current set instead of focusing on achieving first -> last
#   in a way that prioritizes comparison with the elements further
#   up in the list. Like this:
# 
#     (1,1) ...
#     (1,1) (2,2) (1,2) (2,1) ...
#     (1,1) (2,2) (1,2) (2,1) (3,3) (1,3) (3,1) (2,3) (3,2) ...
#     (1,1) (2,2) (1,2) (2,1) (3,3) (1,3) (3,1) (2,3) (3,2) (4,4) (1,4) (4,1) (2,4) (4,2) (3,4) (4,3) ...
# 
#   Or maybe not! Maybe the former is a good mechanism where the early
#   elements in the list are prioritized for comparison with everything
#   else and later elements will only get compared with first elements.


def index_to_pair(index_of_pair, num_elements):
    index_from_back = num_elements**2 - index_of_pair
    group_from_back = int(index_from_back**(1/2))
    group = num_elements - group_from_back
    remaining_in_group = index_from_back - group_from_back**2
    group_size = 2*group_from_back + 1
    index_in_group = group_size - remaining_in_group
    other = group + index_in_group // 2
    if (index_in_group % 2 == 0):
        return (group, other)
    else:
        return (other, group)


def pair_to_index(pair, num_elements):
    group = min(pair)
    index_in_group = 2 * (max(pair) - group)
    if (pair[0] < pair[1]):
        index_in_group -= 1
    group_from_back = 1 + num_elements - group
    group_start_position = num_elements**2 - group_from_back**2 + 1
    index = group_start_position + index_in_group
    return index


# Abstracted version of the index to pair function.
def i2p(i, n):
    # b = n**2 - i
    # c = int(b**(1/2))
    # g = n - c
    # r = b - c**2
    # s = 2*c + 1
    # j = s - r
    #   = (2c + 1) - (b - c2)
    #   = 2c + 1 - b - c2
    #   = c2 + 2c + 1 - b
    #   = (c+1)2 - n2 + i
    # o = g + j // 2
    # 
    n2 = n**2
    c = int((n2 - i)**(1/2))
    g = n - c
    j = (c+1)**2 - n2 + i
    o = g + j // 2
    if (j % 2 == 0):
        return (g, o)
    else:
        return (o, g)

# Abstracted version of the pair to index function.
def p2i(p, n):
    p1, p2 = p
    # 
    # g = min(p1, p2)
    # o = max(p1, p2)
    # j = 2 * (o - g)
    # if (p[0] < p[1]):
    #     j -= 1
    # c = 1 + n - g
    # s = n**2 - c**2 + 1
    # i = s + j
    # 
    # if (p1 < p2):
    #     g = p1
    #     o = p2
    #     j = 2 * (o - g) - 1
    #     c = 1 + n - g
    #     s = n**2 - c**2 + 1
    #     i = s + j
    # else:
    #     g = p2
    #     o = p1
    #     j = 2 * (o - g)
    #     c = 1 + n - g
    #     s = n**2 - c**2 + 1
    #     i = s + j
    # 
    if (p1 < p2):
        i = 2*(n*(p1 - 1) + p2) - p1**2 - 1
    else:
        i = 2*(n*(p2 - 1) + p1) - p2**2
    return i


print()

# Generate example.
n = 4
print("Pretty")
for i in range(1, n**2+1):
    print(f'{i:3d} -> {index_to_pair(i, n)} -> {pair_to_index(index_to_pair(i,n),n)}')
print()
print("Abstracted")
for i in range(1, n**2+1):
    print(f'{i:3d} -> {index_to_pair(i, n)} -> {p2i(i2p(i,n),n)}')
print()

# Matrix operations module.
import fmodpy
random = fmodpy.fimport("../axy_random.f90", dependencies=["pcg32.f90"], name="test_axy_random").random
print("Fortran")
for i in range(1, n**2+1):
    print(f'{i:3d} -> {random.index_to_pair(n, i)} -> {random.pair_to_index(n, *random.index_to_pair(n,i))}')
