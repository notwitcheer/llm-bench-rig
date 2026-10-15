"""dump tokenizer.chat_template of one GGUF to a file (CPU only). usage: python dump_gguf_template.py <gguf> <out.jinja>"""
import sys

from gguf import GGUFReader

r = GGUFReader(sys.argv[1])
f = r.fields["tokenizer.chat_template"]
open(sys.argv[2], "w").write(bytes(f.parts[f.data[0]]).decode("utf-8"))
