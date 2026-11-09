"""Build isolated preview copy; never edits the existing dirty mod sources."""
from pathlib import Path
import hashlib
import json
import shutil
ROOT=Path(__file__).parent
REPO=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
target=ROOT/'installed-mod';target.mkdir(exist_ok=True)
for name in ['main.lua','config.lua','action_injector.lua','state_serializer.lua','game_control.lua','tcp_server.lua','json.lua','metadata.xml']:
    shutil.copyfile(REPO/'mod'/name,target/name)
shutil.copyfile(ROOT/'live_preview.lua',target/'live_preview.lua')
main=(target/'main.lua').read_text(encoding='utf8')
(target/'main.lua').write_text('package.loaded["live_preview"] = nil\ncollectgarbage("collect")\nif require("live_preview").start() then return end\n'+main,encoding='utf8')
with (target/'config.lua').open('r',encoding='utf8') as f:config=f.read()
assert config.count('return Config')==1
config=config.replace('return Config','Config.LIVE_PREVIEW = true\nreturn Config')
(target/'config.lua').write_text(config,encoding='utf8')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}
(ROOT/'build.json').write_text(json.dumps(dict(files=hashes,source=str(REPO/'mod'),profile='preview only, original main returns before legacy callbacks'),indent=2))
print('Built',target)
