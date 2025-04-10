import os
import hashlib
import shutil

# Recursively snapshot a directory.
# Returns a dict mapping relative paths to metadata:
#   - For files: {'type': 'file', 'content': bytes, 'hash': sha256 hash}
#   - For directories: {'type': 'dir'}
# 
def snapshot_directory(directory):
    snapshot = {}
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            relpath = os.path.relpath(os.path.join(root, d), directory)
            snapshot[relpath] = {'type': 'dir'}
        for f in files:
            relpath = os.path.relpath(os.path.join(root, f), directory)
            try:
                with open(os.path.join(root, f), 'rb') as fp:
                    content = fp.read()
            except Exception:
                content = None
            snapshot[relpath] = {
                'type': 'file',
                'content': content,
                'hash': hashlib.sha256(content).hexdigest() if content is not None else None
            }
    return snapshot


# Computes changes from old_snap to new_snap and returns a list of undo operations.
# 
# Undo diff entries are dicts with:
#   - action: 'create', 'delete', 'modify'
#   - path: relative file/directory path
#   - type: 'file' or 'dir'
#   - undo: dict specifying how to revert the change.
# 
# 'create' in new snapshot ➔ undo = delete the item.
# Deletion from old snapshot ➔ undo = restore file (with content) or create directory.
# For modified files ➔ undo = restore the old content.
# 
def diff_snapshots(old_snap, new_snap):
    diff = []
    # Process deletions (item existed before but now is missing)
    for path, meta in old_snap.items():
        if path not in new_snap:
            if meta['type'] == 'file':
                undo = {'action': 'restore_file', 'content': meta['content']}
            else:  # Directory deleted
                undo = {'action': 'create_dir'}
            diff.append({'action': 'delete', 'path': path, 'type': meta['type'], 'undo': undo})
    # Process additions and modifications
    for path, meta in new_snap.items():
        if path not in old_snap:
            # New file or directory: undo by deletion
            undo = {'action': 'delete_item'}
            diff.append({'action': 'create', 'path': path, 'type': meta['type'], 'undo': undo})
        else:
            if meta['type'] == 'file' and meta['hash'] != old_snap[path]['hash']:
                # File modified: undo by restoring original content
                undo = {'action': 'modify', 'content': old_snap[path]['content']}
                diff.append({'action': 'modify', 'path': path, 'type': 'file', 'undo': undo})
    return diff


# Apply the undo operations to revert directory changes.
def undo_diff(directory, diff):
    for change in diff:
        full_path = os.path.join(directory, change['path'])
        act = change['undo']['action']
        if act == 'restore_file':
            # Recreate a deleted file.
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'wb') as fp:
                fp.write(change['undo']['content'])
        elif act == 'create_dir':
            os.makedirs(full_path, exist_ok=True)
        elif act == 'delete_item':
            # Delete added file or directory.
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            elif os.path.exists(full_path):
                os.remove(full_path)
        elif act == 'modify':
            # Revert a modification.
            with open(full_path, 'wb') as fp:
                fp.write(change['undo']['content'])

# Example usage
if __name__ == "__main__":
    import tempfile, time

    # Create a temporary directory for testing.
    test_dir = tempfile.mkdtemp()
    try:
        # Setup: create a file and a subdirectory with a file.
        os.makedirs(os.path.join(test_dir, "subdir"))
        with open(os.path.join(test_dir, "file1.txt"), "wb") as f:
            f.write(b"Original content")
        with open(os.path.join(test_dir, "subdir", "file2.txt"), "wb") as f:
            f.write(b"File in subdir")

        snap1 = snapshot_directory(test_dir)
        time.sleep(1)

        # Make changes: modify file1, add file3, delete file2 and subdir.
        with open(os.path.join(test_dir, "file1.txt"), "wb") as f:
            f.write(b"Modified content")
        with open(os.path.join(test_dir, "file3.txt"), "wb") as f:
            f.write(b"New file")
        os.remove(os.path.join(test_dir, "subdir", "file2.txt"))
        os.rmdir(os.path.join(test_dir, "subdir"))

        snap2 = snapshot_directory(test_dir)
        diff = diff_snapshots(snap1, snap2)

        print()
        print("Diff operations:")
        for op in diff:
            print("", op)

        # To revert the changes, you can call:
        undo_diff(test_dir, diff)

        # Take a final snapshot.
        snap3 = snapshot_directory(test_dir)
        diff = diff_snapshots(snap2, snap3)

        print()
        print("Undo diff operations:")
        for op in diff:
            print("", op)

    finally:
        # shutil.rmtree(test_dir)
        pass
