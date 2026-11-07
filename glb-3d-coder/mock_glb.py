#!/usr/bin/env python3
"""CPU stub of an OpenAI chat endpoint for the proofs. MOCK_MODE per model alias: good | mixed. usage: mock_glb.py PORT [alias=mode ...]
'mixed' cycles per prompt index: good, no-export-button, js-error, blank-canvas, no-html, invalid-glb(export of an empty group), good"""
import json, sys, hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
GOOD = """<!doctype html><html><head><meta charset=utf-8><style>body{margin:0;background:#202830}button{position:fixed;top:8px;left:8px}</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.170.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}</script></head><body>
<button id="exportBtn">Export GLB</button>
<script type="module">
import * as THREE from 'three'; import {GLTFExporter} from 'three/addons/exporters/GLTFExporter.js';
const r=new THREE.WebGLRenderer({antialias:true}); r.setSize(innerWidth,innerHeight); document.body.appendChild(r.domElement);
const s=new THREE.Scene(); s.background=new THREE.Color(0x203040); const c=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,100); c.position.set(0,2,6); c.lookAt(0,1,0);
s.add(new THREE.AmbientLight(0xffffff,1.2)); const l=new THREE.DirectionalLight(0xffffff,2); l.position.set(3,5,4); s.add(l);
const g=new THREE.Group(); const m1=new THREE.Mesh(new THREE.BoxGeometry(1.4,1.6,1),new THREE.MeshStandardMaterial({color:0xe0603a})); m1.position.y=0.8; g.add(m1);
const m2=new THREE.Mesh(new THREE.SphereGeometry(0.6,16,12),new THREE.MeshStandardMaterial({color:0xf0d060})); m2.position.y=2.2; g.add(m2);
const m3=new THREE.Mesh(new THREE.ConeGeometry(0.5,0.9,10),new THREE.MeshStandardMaterial({color:0x40a060})); m3.position.set(0,3.1,0); g.add(m3); %EXTRA%
s.add(g); r.setAnimationLoop(()=>{g.rotation.y+=0.01; r.render(s,c)});
document.getElementById('exportBtn').onclick=()=>{ new GLTFExporter().parse(%TARGET%,(res)=>{const b=new Blob([res],{type:'model/gltf-binary'}); const a=document.createElement('a'); a.href=URL.createObjectURL(b); a.download='asset.glb'; a.click();},(e)=>console.error(e),{binary:true}); };
</script></body></html>"""
def page(kind):
    if kind == "good": return GOOD.replace("%EXTRA%", "").replace("%TARGET%", "g")
    if kind == "noexport": return GOOD.replace("%EXTRA%", "").replace("%TARGET%", "g").replace('<button id="exportBtn">Export GLB</button>', "").replace("document.getElementById('exportBtn').onclick=", "window.nothing=")
    if kind == "jserror": return GOOD.replace("%EXTRA%", "undefinedFunction();").replace("%TARGET%", "g")
    if kind == "blank": return GOOD.replace("%EXTRA%", "g.visible=false; s.background=new THREE.Color(0x203040);").replace("%TARGET%", "g").replace("g.add(m1);", "").replace("g.add(m2);", "").replace("g.add(m3);", "")
    if kind == "badglb": return GOOD.replace("%EXTRA%", "").replace("%TARGET%", "new THREE.Group()")
    return None
CYCLE = ["good", "noexport", "jserror", "blank", "none", "badglb", "good"]
modes = {}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send(200, {"status": "ok"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n))
        alias = body.get("model"); mode = modes.get(alias, "good"); user = body["messages"][-1]["content"]
        texts = [p["text"] for p in json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "prompts.json")))["prompts"]]
        idx = (texts.index(user) if user in texts else 0) % len(CYCLE)
        kind = "good" if mode == "good" else CYCLE[idx]
        html = page(kind)
        txt = "Here is the page.\n```html\n" + html + "\n```\n" if html else "Sorry, I can only describe it in words."
        self._send(200, {"choices": [{"message": {"content": txt, "reasoning_content": "mock thinking " * 20}, "finish_reason": "stop"}],
                         "usage": {"completion_tokens": len(txt) // 4 + (3000 if alias.startswith("qwen") else 500), "prompt_tokens": 120}, "timings": {"predicted_per_second": 88.8}})
if __name__ == "__main__":
    for kv in sys.argv[2:]: k, v = kv.split("="); modes[k] = v
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
