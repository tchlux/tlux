from tlux.search.hkm.search.loader import DocumentStore
from tlux.search.hkm import FileSystem
store = DocumentStore(FileSystem(), "tmp/idx")      # path to your tmp idx
print({i: store.get_metadata(i)["num_token_count"] for i in (0, 1, 2)})
