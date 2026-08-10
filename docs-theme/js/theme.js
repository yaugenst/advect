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
const searchQ = document.getElementById("mkdocs-search-query");
const searchButton = document.getElementById("searchbtn");
const base = document.body.dataset.base ? `${document.body.dataset.base}/` : "";
function openSearch() {
  searchBox.classList.add("on");
  searchBox.setAttribute("aria-hidden", "false");
  searchButton.setAttribute("aria-expanded", "true");
  searchQ.select();
}
function closeSearch() {
  searchBox.classList.remove("on");
  searchBox.setAttribute("aria-hidden", "true");
  searchButton.setAttribute("aria-expanded", "false");
}
searchQ.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeSearch();
    searchButton.focus();
    e.preventDefault();
  }
  e.stopPropagation();
});
searchButton.addEventListener("click", openSearch);

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
