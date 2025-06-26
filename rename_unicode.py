import os
import re

def decode_unicode_escapes(s):
    return re.sub(r'#U([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

base_dir = './train'  # train 폴더 기준

for root, dirs, files in os.walk(base_dir):
    for name in dirs + files:
        if '#U' in name:
            new_name = decode_unicode_escapes(name)
            os.rename(os.path.join(root, name), os.path.join(root, new_name))

