import hashlib, json, time

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

prev_hash = "0"*64

def log_inspection(frame_bytes, preds, gps):
    # … paste scaffold code here …
    pass
