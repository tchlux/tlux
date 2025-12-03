# A worker whose job is to steadily consume files from a doc
#  directory, steadily write docs to appropriate cluster_XXXX/doc/
#  directories based on proximity, and accumulate summary staticis
#  across writes per cluster (writing metadata summary to that cluster
#  upon finishing).
# 
# Reserves files by moving them to our own working directory ./docs/worker_XXXX/
# and writes aggregate summary statistic information (categorical counts, numeric
# percentiles, and bloom filters) to ./docs/worker_XXXX/ including the files:
#   category_map.json  --  Category label counts.
#   numeric_map.json   --  Numeric percentiles [0, 100].
#   hashbitmask.bin    --  Bloom filter over n-grams of tokens in docs scanned.
# 

