"""
ブラウザで使う画面（このパソコンの中だけで動くサイト）
起動: ./start.sh   → ブラウザで http://127.0.0.1:8765 が開きます
"""
import os
import queue
import shutil
import threading
import time
import uuid
import webbrowser

from flask import Flask, abort, jsonify, render_template_string, request, send_file

import masker

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "work")
os.makedirs(WORK, exist_ok=True)

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024  # 8GB

jobs: dict[str, dict] = {}
q: "queue.Queue[str]" = queue.Queue()


def worker():
    while True:
        jid = q.get()
        job = jobs[jid]
        job["status"] = "running"

        def prog(p, msg):
            job["progress"], job["message"] = p, msg

        try:
            fn = masker.process_video if job["kind"] == "video" else masker.process_image
            job["report"] = fn(job["src"], job["dst"], cats=job["cats"], words=job["words"],
                               style=job["style"], interval=job["interval"], progress=prog)
            job["status"], job["progress"], job["message"] = "done", 1.0, "完了"
        except Exception as e:  # noqa
            job["status"], job["message"] = "error", f"エラー: {e}"
        finally:
            try:
                os.remove(job["src"])  # 元ファイルはすぐ消す
            except OSError:
                pass


threading.Thread(target=worker, daemon=True).start()


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/upload")
def upload():
    files = request.files.getlist("files")
    cats = set(request.form.getlist("cats"))
    words = [w.strip() for w in request.form.get("words", "").splitlines() if w.strip()]
    style = request.form.get("style", "fill")
    interval = float(request.form.get("interval", "0.5"))
    ids = []
    for f in files:
        name = os.path.basename(f.filename or "file")
        ext = os.path.splitext(name)[1].lower()
        if ext not in VIDEO_EXT + IMAGE_EXT:
            continue
        jid = uuid.uuid4().hex[:12]
        d = os.path.join(WORK, jid)
        os.makedirs(d)
        src = os.path.join(d, "input" + ext)
        f.save(src)
        kind = "video" if ext in VIDEO_EXT else "image"
        out_name = os.path.splitext(name)[0] + "_masked" + (".mp4" if kind == "video" else ".png")
        jobs[jid] = dict(id=jid, name=name, out_name=out_name, kind=kind, src=src,
                         dst=os.path.join(d, "output" + (".mp4" if kind == "video" else ".png")),
                         cats=cats, words=words, style=style, interval=interval,
                         status="waiting", progress=0.0, message="順番待ち", report=[], created=time.time())
        q.put(jid)
        ids.append(jid)
    return jsonify(ids=ids)


@app.get("/jobs")
def list_jobs():
    keys = ("id", "name", "out_name", "kind", "status", "progress", "message", "report")
    return jsonify([{k: j[k] for k in keys} for j in sorted(jobs.values(), key=lambda j: j["created"])])


@app.get("/file/<jid>")
def file(jid):
    j = jobs.get(jid) or abort(404)
    if j["status"] != "done":
        abort(404)
    return send_file(j["dst"], as_attachment=request.args.get("dl") == "1", download_name=j["out_name"])


@app.post("/delete/<jid>")
def delete(jid):
    j = jobs.pop(jid, None)
    if j:
        shutil.rmtree(os.path.dirname(j["dst"]), ignore_errors=True)
    return jsonify(ok=True)


PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>個人情報マスキング</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#1d2433;--sub:#667085;--line:#e3e6ec;--accent:#2f5bd3;--ok:#1f8a4c;--err:#c0392b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Hiragino Sans","Yu Gothic UI",sans-serif;line-height:1.6}
.wrap{max-width:920px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:22px;margin:0 0 4px}.lead{color:var(--sub);margin:0 0 20px;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:16px}
.drop{border:2px dashed #b9c3d8;border-radius:12px;padding:36px 16px;text-align:center;cursor:pointer;transition:.15s}
.drop.over{background:#eef3ff;border-color:var(--accent)}.drop b{font-size:17px}.drop small{display:block;color:var(--sub);margin-top:6px}
h2{font-size:15px;margin:0 0 10px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px 14px}
label{font-size:14px;cursor:pointer}textarea{width:100%;min-height:80px;border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit;font-size:14px}
.row{display:flex;gap:24px;flex-wrap:wrap;margin-top:6px}.hint{color:var(--sub);font-size:12.5px;margin:4px 0 0}
details summary{cursor:pointer;font-weight:600;font-size:15px}
.job{border:1px solid var(--line);border-radius:10px;padding:14px;margin-top:12px}
.job .top{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}
.bar{height:8px;background:#e9edf3;border-radius:99px;overflow:hidden;margin:8px 0 4px}.bar i{display:block;height:100%;background:var(--accent);transition:width .4s}
.msg{font-size:13px;color:var(--sub)}.done .bar i{background:var(--ok)}.error .msg{color:var(--err)}
.btn{display:inline-block;background:var(--accent);color:#fff;border:0;border-radius:8px;padding:7px 14px;font-size:14px;text-decoration:none;cursor:pointer}
.btn.ghost{background:none;color:var(--sub);border:1px solid var(--line)}
video,img.prev{width:100%;border-radius:8px;margin-top:10px;background:#000}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}td,th{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line)}
.warn{background:#fff8e6;border:1px solid #f3d68b;border-radius:10px;padding:10px 14px;font-size:13px}
</style></head><body><div class="wrap">
<h1>個人情報マスキング</h1>
<p class="lead">操作録画や画面のスクショから、電話番号・メール・住所・名前などを自動で塗りつぶします。<b>処理はすべてこのパソコンの中</b>で行い、外部には送りません。</p>

<div class="card">
  <div class="drop" id="drop"><b>ここに動画・画像をドラッグ＆ドロップ</b><small>またはクリックして選択（複数OK／mp4・mov・png・jpg）</small></div>
  <input type="file" id="file" multiple accept="video/*,image/*" hidden>
</div>

<div class="card">
<details open><summary>隠すもの・設定</summary>
<div style="margin-top:12px">
<h2>自動で見つけて隠すもの</h2>
<div class="grid">
<label><input type="checkbox" name="cats" value="phone" checked> 電話番号</label>
<label><input type="checkbox" name="cats" value="email" checked> メールアドレス</label>
<label><input type="checkbox" name="cats" value="address" checked> 住所</label>
<label><input type="checkbox" name="cats" value="postal" checked> 郵便番号</label>
<label><input type="checkbox" name="cats" value="card" checked> カード・マイナンバー等の長い番号</label>
<label><input type="checkbox" name="cats" value="honorific" checked> 「〇〇様」「〇〇さん」の名前</label>
<label><input type="checkbox" name="cats" value="label" checked> 「氏名」「住所」などの欄の中身</label>
</div>
<h2 style="margin-top:16px">必ず隠したい言葉（1行に1つ）</h2>
<textarea id="words" placeholder="例）&#10;山田太郎&#10;株式会社〇〇&#10;顧客の会社名・担当者名・自分の名前など"></textarea>
<p class="hint">スペースの有無・全角半角の違いは無視して探します。この画面を閉じても次回まで覚えています。</p>
<div class="row">
<div><h2>隠し方</h2>
<label><input type="radio" name="style" value="fill" checked> 塗りつぶし（おすすめ）</label>&nbsp;
<label><input type="radio" name="style" value="mosaic"> モザイク</label></div>
<div><h2>文字を読む間隔（動画）</h2>
<select id="interval"><option value="0.25">0.25秒（丁寧・遅い）</option><option value="0.5" selected>0.5秒（標準）</option><option value="1">1秒（速い）</option></select></div>
</div>
</div></details>
</div>

<div class="warn">⚠ 自動判定は100%ではありません。AIや社外に渡す前に、仕上がりの動画をざっと見て確認してください。</div>

<div id="jobs"></div>
</div>
<script>
const $=s=>document.querySelector(s);
const words=$("#words");
try{words.value=localStorage.getItem("words")||""}catch(e){}
words.addEventListener("input",()=>{try{localStorage.setItem("words",words.value)}catch(e){}});
const drop=$("#drop"),file=$("#file");
drop.onclick=()=>file.click();
file.onchange=()=>send(file.files);
["dragenter","dragover"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add("over")}));
["dragleave","drop"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove("over")}));
drop.addEventListener("drop",ev=>send(ev.dataTransfer.files));
function send(files){
  if(!files.length)return;
  const fd=new FormData();
  for(const f of files)fd.append("files",f);
  document.querySelectorAll('input[name=cats]:checked').forEach(c=>fd.append("cats",c.value));
  fd.append("words",words.value);
  fd.append("style",document.querySelector('input[name=style]:checked').value);
  fd.append("interval",$("#interval").value);
  drop.querySelector("b").textContent="アップロード中…";
  fetch("/upload",{method:"POST",body:fd}).then(()=>{drop.querySelector("b").textContent="ここに動画・画像をドラッグ＆ドロップ";file.value="";poll()});
}
const shown={};
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function fmt(t){t=Math.floor(t);return Math.floor(t/60)+":"+String(t%60).padStart(2,"0")}
function render(js){
  const box=$("#jobs");
  for(const j of js){
    let el=document.getElementById("j"+j.id);
    if(!el){el=document.createElement("div");el.id="j"+j.id;el.className="job";box.prepend(el);
      el.innerHTML=`<div class="top"><b>${esc(j.name)}</b><span class="act"></span></div><div class="bar"><i></i></div><div class="msg"></div><div class="res"></div>`;}
    el.className="job "+j.status;
    el.querySelector(".bar i").style.width=(j.progress*100).toFixed(0)+"%";
    el.querySelector(".msg").textContent=j.message;
    if(j.status==="done"&&!shown[j.id]){
      shown[j.id]=1;
      el.querySelector(".act").innerHTML=`<a class="btn" href="/file/${j.id}?dl=1">ダウンロード</a> <button class="btn ghost" onclick="del('${j.id}')">削除</button>`;
      const media=j.kind==="video"?`<video src="/file/${j.id}" controls></video>`:`<img class="prev" src="/file/${j.id}">`;
      const rows=j.report.map(r=>`<tr><td>${j.kind==="video"?fmt(r.time):""}</td><td>${esc(r.category)}</td><td>${esc(r.text)}</td></tr>`).join("");
      el.querySelector(".res").innerHTML=media+`<details style="margin-top:8px"><summary>隠した内容 ${j.report.length}件（中身は伏せて表示）</summary><table><tr><th>時刻</th><th>種類</th><th>内容</th></tr>${rows}</table></details>`;
    }
  }
}
function del(id){fetch("/delete/"+id,{method:"POST"}).then(()=>{document.getElementById("j"+id).remove()})}
let timer=null;
function poll(){clearTimeout(timer);fetch("/jobs").then(r=>r.json()).then(js=>{render(js);if(js.some(j=>j.status==="waiting"||j.status==="running"))timer=setTimeout(poll,1000)})}
poll();
</script></body></html>"""

if __name__ == "__main__":
    port = 8765
    print(f"\n  ブラウザで開いてください →  http://127.0.0.1:{port}\n  （終わるときはこの画面で Ctrl+C）\n")
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    masker._get_engine()  # 最初の読み込みを先に済ませる
    app.run(host="127.0.0.1", port=port, threaded=True)
