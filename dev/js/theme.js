/* Advect docs shell: display modes, tree pane, keyboard nav, search. */
"use strict";

/* ---------------- display modes ---------------- */
const root = document.documentElement;
const mq = window.matchMedia("(prefers-color-scheme: dark)");
const MODE_LABEL = { light: "lp0", dark: "dark", crt: "crt" };
const MODE_NEXT = { light: "dark", dark: "crt", crt: "light" };
const curTheme = () => root.dataset.theme || (mq.matches ? "dark" : "light");
const themeBtn = document.getElementById("themeToggle");
const syncMode = () => { themeBtn.textContent = `[ mode: ${MODE_LABEL[curTheme()] || "lp0"} ]`; };
const flip = () => {
  root.dataset.theme = MODE_NEXT[curTheme()] || "light";
  try { localStorage.setItem("advect-mode", root.dataset.theme); } catch { /* private mode */ }
  syncMode();
};
themeBtn.addEventListener("click", flip);
mq.addEventListener("change", syncMode);
syncMode();

/* ---------------- tree pane ---------------- */
const treetile = document.getElementById("treetile");
/* the open state lives on <html> so the pre-paint script in main.html can
   apply it before first paint — a body class only exists after parse and
   made the pane animate open on every navigation */
function setTree(open) {
  document.documentElement.classList.toggle("tree-open", open);
  if (treetile) treetile.setAttribute("aria-hidden", String(!open));
  try { localStorage.setItem("advect-tree", open ? "1" : "0"); } catch { /* private mode */ }
}
function toggleTree() { setTree(!document.documentElement.classList.contains("tree-open")); }
const treeButton = document.getElementById("treebtn");
if (treetile) {
  treetile.setAttribute(
    "aria-hidden",
    String(!document.documentElement.classList.contains("tree-open")),
  );
  treetile.querySelector("a.cur")?.scrollIntoView({ block: "center" });
  treeButton.addEventListener("click", toggleTree);
} else {
  treeButton.hidden = true;
}

/* ---------------- copy buttons on code blocks ---------------- */
for (const pre of document.querySelectorAll(".md pre")) {
  const code = pre.querySelector("code");
  if (!code) continue;
  const btn = document.createElement("button");
  btn.className = "copybtn";
  btn.textContent = "[copy]";
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      btn.textContent = "[copied]";
      setTimeout(() => { btn.textContent = "[copy]"; }, 1200);
    } catch { btn.textContent = "[denied]"; }
  });
  pre.appendChild(btn);
}

/* ---------------- search ---------------- */
const searchBox = document.getElementById("search");
const searchQ = document.getElementById("searchq");
const searchHits = document.getElementById("searchhits");
const base = document.body.dataset.base ? `${document.body.dataset.base}/` : "";
let index = null;
let hits = [];
let selHit = 0;

async function loadIndex() {
  if (index) return index;
  const res = await fetch(`${base}search/search_index.json`);
  index = (await res.json()).docs.map((d) => ({
    ...d,
    ltitle: d.title ? d.title.toLowerCase() : "",
    ltext: d.text ? d.text.toLowerCase() : "",
  }));
  return index;
}
const escHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
function snippet(text, term) {
  const at = text.toLowerCase().indexOf(term);
  if (at < 0) return escHtml(text.slice(0, 90));
  const s = Math.max(0, at - 36);
  const raw = text.slice(s, at) + "\u0001" + text.slice(at, at + term.length) + "\u0002" +
    text.slice(at + term.length, at + term.length + 54);
  return (s > 0 ? "…" : "") + escHtml(raw).replace("\u0001", "<mark>").replace("\u0002", "</mark>") + "…";
}
function runSearch() {
  const q = searchQ.value.trim().toLowerCase();
  selHit = 0;
  if (!q || !index) { searchHits.innerHTML = ""; hits = []; return; }
  const terms = q.split(/\s+/);
  hits = index
    .map((d) => {
      let score = 0;
      for (const t of terms) {
        if (d.ltitle.includes(t)) score += 4;
        if (d.ltext.includes(t)) score += 1;
      }
      return { d, score };
    })
    .filter((h) => h.score >= terms.length)
    .sort((a, b) => b.score - a.score || a.d.location.length - b.d.location.length)
    .slice(0, 12);
  if (!hits.length) {
    searchHits.innerHTML = `<div class="search-none">No manual entry for "${escHtml(q)}"</div>`;
    return;
  }
  searchHits.innerHTML = hits
    .map((h, i) => `<a class="search-hit${i === selHit ? " selq" : ""}" href="${base}${h.d.location}">` +
      `<span class="h-title">${escHtml(h.d.title || h.d.location)}</span> ` +
      `<span class="h-loc">${escHtml(h.d.location.split("#")[0] || "index")}</span><br>` +
      `${snippet(h.d.text || "", terms[0])}</a>`)
    .join("");
}
function moveSel(dir) {
  if (!hits.length) return;
  selHit = (selHit + dir + hits.length) % hits.length;
  const els = searchHits.querySelectorAll(".search-hit");
  els.forEach((el, i) => el.classList.toggle("selq", i === selHit));
  els[selHit].scrollIntoView({ block: "nearest" });
}
function openSearch() {
  searchBox.classList.add("on");
  searchQ.select();
  loadIndex().then(runSearch).catch(() => {
    searchHits.innerHTML = '<div class="search-none">search index unavailable</div>';
  });
}
const closeSearch = () => searchBox.classList.remove("on");
searchQ.addEventListener("input", runSearch);
searchQ.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeSearch(); return; }
  if (e.key === "ArrowDown") { moveSel(1); e.preventDefault(); return; }
  if (e.key === "ArrowUp") { moveSel(-1); e.preventDefault(); return; }
  if (e.key === "Enter" && hits.length) window.location.href = `${base}${hits[selHit].d.location}`;
  e.stopPropagation();
});
document.getElementById("searchbtn").addEventListener("click", openSearch);

/* ---------------- keyboard ---------------- */
const help = document.getElementById("help");
document.getElementById("helpbtn").addEventListener("click", () => help.classList.toggle("on"));
const scroller = () => document.getElementById("manscroll") || document.scrollingElement;
const go = (sel) => { const a = document.querySelector(sel); if (a) window.location.href = a.href; };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "Escape") { help.classList.remove("on"); closeSearch(); return; }
  const man = scroller();
  switch (e.key) {
    case "t": flip(); break;
    case "e": toggleTree(); break;
    case "/": openSearch(); break;
    case "?": help.classList.toggle("on"); break;
    case "n": go(".seealso .nn a"); break;
    case "p": go(".seealso a"); break;
    case "2": go("#wsplay"); break;
    case "1": if (document.getElementById("wsplay").getAttribute("aria-current") === "true") window.location.href = base || "./"; break;
    case "j": man.scrollBy(0, 64); break;
    case "k": man.scrollBy(0, -64); break;
    case "d": man.scrollBy(0, man.clientHeight / 2); break;
    case "u": man.scrollBy(0, -man.clientHeight / 2); break;
    case "g": man.scrollTo(0, 0); break;
    case "G": man.scrollTo(0, man.scrollHeight); break;
    default: return;
  }
  e.preventDefault();
});
