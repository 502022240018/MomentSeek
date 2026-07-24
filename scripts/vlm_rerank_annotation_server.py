from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MomentSeek VLM 精排标注</title>
<style>
:root{color-scheme:dark;--bg:#101319;--panel:#181d26;--muted:#98a2b3;--line:#303746;--blue:#5b8cff;--green:#37b779;--red:#e35d6a;--amber:#e7a83e}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef2f7;font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}button,select,input,textarea{font:inherit}.top{position:sticky;top:0;z-index:4;display:flex;gap:12px;align-items:center;padding:10px 16px;background:#121722;border-bottom:1px solid var(--line)}.top strong{font-size:17px}.progress{flex:1;height:8px;background:#2a303c;border-radius:9px;overflow:hidden}.progress i{display:block;height:100%;background:var(--green)}.layout{display:grid;grid-template-columns:280px 1fr;min-height:calc(100vh - 52px)}aside{border-right:1px solid var(--line);padding:12px;overflow:auto;height:calc(100vh - 52px);position:sticky;top:52px}.filters{display:grid;gap:8px;margin-bottom:12px}.q{width:100%;text-align:left;padding:9px;border:1px solid transparent;border-radius:7px;background:transparent;color:#dce3ed;cursor:pointer}.q:hover,.q.active{background:#232a36;border-color:#3c4658}.q small{display:block;color:var(--muted)}main{padding:18px;max-width:1500px;width:100%;margin:auto}.query{font-size:23px;margin:0 0 8px}.chips,.sources{display:flex;gap:6px;flex-wrap:wrap}.chip{padding:3px 8px;border:1px solid var(--line);border-radius:999px;color:#c7d0dd}.toolbar{display:flex;justify-content:space-between;align-items:center;margin:14px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px}.meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);margin-bottom:12px}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.frames img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#080a0e;border-radius:6px}.evidence{margin:12px 0;padding:10px;background:#11151c;border-radius:7px;white-space:pre-wrap}.scores{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:14px 0}.score{padding:12px;border:2px solid var(--line);border-radius:8px;background:#202633;color:#fff;cursor:pointer}.score.selected{border-color:var(--blue);background:#26395f}.constraints{display:grid;gap:7px}.constraint{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;padding:7px 0;border-bottom:1px solid #292f3a}.tri button{border:1px solid var(--line);background:#202633;color:#bec7d5;padding:5px 9px;cursor:pointer}.tri button.on.true{background:#174d35;color:#7be0ac}.tri button.on.false{background:#5b252d;color:#ffadb5}.tri button.on.null{background:#4b4430;color:#f3cd77}textarea{width:100%;min-height:70px;margin-top:12px;background:#11151c;color:#fff;border:1px solid var(--line);border-radius:7px;padding:9px}.actions{display:flex;gap:8px;margin-top:12px}.actions button,.top button,.top select{padding:7px 11px;border:1px solid var(--line);border-radius:6px;background:#242b37;color:#fff;cursor:pointer}.primary{background:#285ac7!important}.status{color:var(--muted)}@media(max-width:900px){.layout{grid-template-columns:1fr}aside{display:none}.frames{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<div class="top"><strong>VLM 精排标注</strong><span id="count"></span><div class="progress"><i id="bar"></i></div><select id="mode"><option value="all">全部模式</option><option value="visual_only">纯视觉</option><option value="evidence_fusion">Evidence 融合</option></select><select id="state"><option value="all">全部状态</option><option value="pending">未标注</option><option value="done">已标注</option></select></div>
<div class="layout"><aside><div id="queries"></div></aside><main id="main"></main></div>
<script>
let data=null, qi=0, ci=0, saving=false; const $=s=>document.querySelector(s); const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load(){data=await fetch('/api/data').then(r=>r.json());renderNav();show(0,0)}
function ann(c){return data.annotations[c.candidate_id]}
function done(c){return ann(c)?.relevance!==null&&ann(c)?.relevance!==undefined}
function allCandidates(){return data.queries.flatMap((q,i)=>q.candidates.map((c,j)=>({q,i,c,j})))}
function renderProgress(){const a=allCandidates(), n=a.filter(x=>done(x.c)).length;$('#count').textContent=`${n}/${a.length}`;$('#bar').style.width=`${a.length?n/a.length*100:0}%`}
function visible(q){const mode=$('#mode').value,state=$('#state').value;if(mode!=='all'&&q.mode!==mode)return false;if(state==='all')return true;return q.candidates.some(c=>state==='done'?done(c):!done(c))}
function renderNav(){const box=$('#queries');box.innerHTML=data.queries.map((q,i)=>visible(q)?`<button class="q ${i===qi?'active':''}" onclick="show(${i},0)">${esc(q.query_id)} · ${esc(q.mode)}<small>${q.candidates.filter(done).length}/${q.candidates.length}　${esc(q.query)}</small></button>`:'').join('');renderProgress()}
function evidence(c){const rows=c.model_input?.evidence||[];return rows.map(x=>`[${x.modality}] ${x.text||''} ${x.best_time??''}`).join('\n')||'无文本 Evidence（纯视觉模式或该候选只有视觉证据）'}
function show(i,j){qi=i;ci=Math.max(0,Math.min(j,data.queries[i].candidates.length-1));const q=data.queries[qi],c=q.candidates[ci],a=ann(c);const matches=a.constraint_matches||Object.fromEntries(q.constraints.map(x=>[x,null]));$('#main').innerHTML=`<h1 class="query">${esc(q.query)}</h1><div class="chips">${q.constraints.map(x=>`<span class="chip">${esc(x)}</span>`).join('')}</div><div class="toolbar"><span>${esc(q.query_id)} · ${esc(q.mode)} · 候选 ${ci+1}/${q.candidates.length}</span><span class="status" id="saveStatus"></span></div><section class="card"><div class="meta"><b>${esc(c.video_name)}</b><span>${c.start_time.toFixed(3)}s – ${c.end_time.toFixed(3)}s</span><span>候选池 #${c.rank}</span><span>${c.retrieval_sources.map(s=>`${s.channel} #${s.rank}`).join(' · ')}</span></div><div class="frames">${c.frame_paths.map(p=>`<a href="/files/${encodeURIComponent(p)}" target="_blank"><img src="/files/${encodeURIComponent(p)}" loading="eager"></a>`).join('')}</div><div class="evidence">${esc(evidence(c))}</div><div class="scores">${[0,1,2,3].map(n=>`<button class="score ${a.relevance===n?'selected':''}" onclick="setScore(${n})"><b>${n}</b><br>${['不相关','部分线索/困难负例','基本符合但不完整','完整符合'][n]}</button>`).join('')}</div><h3>逐约束判断</h3><div class="constraints">${q.constraints.map((x,k)=>`<div class="constraint"><span>${esc(x)}</span><span class="tri"><button class="${matches[x]===true?'on true':''}" onclick="setConstraint(${k},true)">满足</button><button class="${matches[x]===false?'on false':''}" onclick="setConstraint(${k},false)">不满足</button><button class="${matches[x]==null?'on null':''}" onclick="setConstraint(${k},null)">不确定</button></span></div>`).join('')}</div><textarea id="reason" placeholder="判定理由、关键帧或异常说明">${esc(a.reason||'')}</textarea><input id="reviewer" placeholder="标注人（可选）" value="${esc(a.reviewer||'')}" style="margin-top:8px;padding:8px;background:#11151c;color:white;border:1px solid var(--line);border-radius:6px"><div class="actions"><button onclick="move(-1)">← 上一个</button><button class="primary" onclick="save(true)">保存并下一个</button><button onclick="nextPending()">下一个未标注</button></div></section>`;renderNav()}
function current(){const q=data.queries[qi],c=q.candidates[ci];return{q,c,a:ann(c)}}
function syncDraft(){const x=current().a,reason=$('#reason'),reviewer=$('#reviewer');if(reason)x.reason=reason.value;if(reviewer)x.reviewer=reviewer.value}
function setScore(n){syncDraft();current().a.relevance=n;show(qi,ci);save(false)}
function setConstraint(k,v){syncDraft();const {q,a}=current();a.constraint_matches=a.constraint_matches||Object.fromEntries(q.constraints.map(x=>[x,null]));a.constraint_matches[q.constraints[k]]=v;show(qi,ci);save(false)}
async function save(advance=false){if(saving)return;const {c,a}=current();a.reason=$('#reason').value;a.reviewer=$('#reviewer').value;saving=true;$('#saveStatus').textContent='保存中…';const r=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(a)});saving=false;if(!r.ok){alert('保存失败：'+await r.text());return}$('#saveStatus').textContent='已保存';renderProgress();renderNav();if(advance)move(1)}
function move(d){let i=qi,j=ci+d;if(j>=data.queries[i].candidates.length){i=(i+1)%data.queries.length;j=0}else if(j<0){i=(i-1+data.queries.length)%data.queries.length;j=data.queries[i].candidates.length-1}show(i,j)}
function nextPending(){const a=allCandidates(),pos=a.findIndex(x=>x.i===qi&&x.j===ci);for(let n=1;n<=a.length;n++){const x=a[(pos+n)%a.length];if(!done(x.c)){show(x.i,x.j);return}}alert('全部候选都已标注')}
$('#mode').onchange=()=>renderNav();$('#state').onchange=()=>renderNav();document.addEventListener('keydown',e=>{if(e.target.matches('textarea,input,select'))return;if('0123'.includes(e.key))setScore(+e.key);else if(e.key==='ArrowRight')move(1);else if(e.key==='ArrowLeft')move(-1);else if(e.key.toLowerCase()==='s')save(true)});load();
</script></body></html>"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.candidates_path = self.root / "candidates" / "candidate_sets.jsonl"
        self.annotations_path = self.root / "annotations.jsonl"
        self.queries = read_jsonl(self.candidates_path)
        self.annotations = read_jsonl(self.annotations_path)
        self.by_candidate = {row["candidate_id"]: row for row in self.annotations}
        self.lock = threading.Lock()

    def payload(self) -> dict[str, Any]:
        return {"queries": self.queries, "annotations": self.by_candidate}

    def update(self, incoming: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(incoming.get("candidate_id") or "")
        if candidate_id not in self.by_candidate:
            raise ValueError("unknown candidate_id")
        relevance = incoming.get("relevance")
        if relevance is not None and (not isinstance(relevance, int) or not 0 <= relevance <= 3):
            raise ValueError("relevance must be null or integer 0..3")
        matches = incoming.get("constraint_matches", {})
        if not isinstance(matches, dict) or any(value not in (None, True, False) for value in matches.values()):
            raise ValueError("constraint_matches values must be null/true/false")
        with self.lock:
            current = self.by_candidate[candidate_id]
            for key in ("relevance", "constraint_matches", "reason", "reviewer"):
                if key in incoming:
                    current[key] = incoming[key]
            write_jsonl_atomic(self.annotations_path, self.annotations)
        return current


def make_handler(store: DatasetStore):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                body = HTML.encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if parsed.path == "/api/data": self._json(store.payload()); return
            if parsed.path.startswith("/files/"):
                relative = urllib.parse.unquote(parsed.path[len("/files/"):])
                target = (store.root / relative).resolve()
                if target != store.root and store.root not in target.parents: self.send_error(403); return
                if not target.is_file(): self.send_error(404); return
                body = target.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            self.send_error(404)

        def do_POST(self) -> None:
            if urllib.parse.urlparse(self.path).path != "/api/annotation": self.send_error(404); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 1_000_000: raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict): raise ValueError("JSON body must be an object")
                self._json(store.update(payload))
            except (ValueError, json.JSONDecodeError) as exc: self._json({"error": str(exc)}, 400)

        def log_message(self, format: str, *args: Any) -> None:
            if args and str(args[1]) != "200": super().log_message(format, *args)
    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local VLM reranking annotation UI.")
    parser.add_argument("--dataset", type=Path, default=Path("runtime/eval/vlm_rerank_phase1_channel_union"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    store = DatasetStore(args.dataset)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}/"
    print(f"VLM rerank annotation UI: {url}")
    print(f"Annotations: {store.annotations_path}")
    if not args.no_open: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
