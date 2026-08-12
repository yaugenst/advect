/* Advect docs shell: display modes, tree pane, keyboard nav, search. */
"use strict";

/* ---------------- display modes ---------------- */
const root = document.documentElement;
const MODE_NEXT = { system: "light", light: "dark", dark: "system" };
const curTheme = () => root.dataset.theme || "system";
const themeBtn = document.getElementById("themeToggle");
const syncMode = () => { themeBtn.textContent = `[ mode: ${curTheme()} ]`; };
const flip = () => {
  const next = MODE_NEXT[curTheme()] || "system";
  if (next === "system") delete root.dataset.theme;
  else root.dataset.theme = next;
  try {
    if (next === "system") localStorage.removeItem("advect-mode");
    else localStorage.setItem("advect-mode", next);
  } catch { /* private mode */ }
  syncMode();
};
themeBtn.addEventListener("click", flip);
syncMode();

/* ---------------- API overloads ---------------- */
for (const overloads of document.querySelectorAll(".doc-overloads")) {
  const count = overloads.childElementCount;
  const primary = overloads.nextElementSibling;
  const details = document.createElement("details");
  details.className = "doc-alternative-signatures";
  const summary = document.createElement("summary");
  summary.textContent = `${count} alternative signature${count === 1 ? "" : "s"}`;
  details.append(summary, ...overloads.children);
  overloads.replaceWith(details);
  if (primary?.classList.contains("doc-signature")) primary.after(details);
}

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

/* ---------------- page outline ---------------- */
const outlineTile = document.getElementById("outlinetile");
const outlineLinks = document.getElementById("outlinelinks");
const outlineButton = document.getElementById("outlinebtn");
const links = outlineLinks ? [...outlineLinks.querySelectorAll("a")] : [];
const outlineHeadings = links.map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1))));

function setOutline(open) {
  if (!outlineTile) return;
  root.classList.toggle("outline-open", open);
  outlineTile.setAttribute("aria-hidden", String(!open));
  outlineButton.setAttribute("aria-expanded", String(open));
  try { localStorage.setItem("advect-outline", open ? "1" : "0"); } catch { /* private mode */ }
}
function toggleOutline() {
  if (links.length) setOutline(!root.classList.contains("outline-open"));
}

if (links.length) {
  outlineLinks.addEventListener("click", () => {
    if (window.matchMedia("(max-width: 900px), (max-height: 560px)").matches) setOutline(false);
  });
  outlineTile.setAttribute("aria-hidden", String(!root.classList.contains("outline-open")));
  outlineButton.setAttribute("aria-expanded", String(root.classList.contains("outline-open")));
  outlineButton.addEventListener("click", toggleOutline);

  const manScroll = document.getElementById("manscroll");
  let active = -1;
  function syncOutline() {
    const stacked = getComputedStyle(manScroll).overflowY === "visible";
    const threshold = stacked ? 120 : manScroll.getBoundingClientRect().top + 120;
    const atEnd = stacked
      ? window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2
      : manScroll.scrollTop + manScroll.clientHeight >= manScroll.scrollHeight - 2;
    let next = atEnd ? outlineHeadings.length - 1 : 0;
    if (!atEnd) {
      for (let i = 0; i < outlineHeadings.length; i += 1) {
        if (outlineHeadings[i].getBoundingClientRect().top > threshold) break;
        next = i;
      }
    }
    if (next === active) return;
    if (active >= 0) links[active].removeAttribute("aria-current");
    active = next;
    links[active].setAttribute("aria-current", "location");
    links[active].scrollIntoView({ block: "nearest" });
  }
  manScroll.addEventListener("scroll", syncOutline, { passive: true });
  window.addEventListener("scroll", syncOutline, { passive: true });
  syncOutline();
} else {
  root.classList.remove("outline-open");
  if (outlineTile) outlineTile.hidden = true;
  outlineButton.hidden = true;
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

/* ---------------- internal-link previews ---------------- */
const linkPreview = document.getElementById("linkpreview");
const previewPath = document.getElementById("previewpath");
const previewTitle = document.getElementById("previewtitle");
const previewSignature = document.getElementById("previewsignature");
const previewDescription = document.getElementById("previewdescription");
const documentCache = new Map();
let previewTimer = 0;
let activePreviewLink = null;

function clipText(text, limit) {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > limit ? `${clean.slice(0, limit - 1).trimEnd()}…` : clean;
}

function documentFor(url) {
  const source = new URL(url);
  source.hash = "";
  const here = new URL(window.location.href);
  here.hash = "";
  if (source.href === here.href) return Promise.resolve(document);
  if (!documentCache.has(source.href)) {
    documentCache.set(source.href, fetch(source).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.text();
    }).then((html) => new DOMParser().parseFromString(html, "text/html")));
  }
  return documentCache.get(source.href);
}

function paragraphAfter(heading) {
  for (let node = heading?.nextElementSibling; node; node = node.nextElementSibling) {
    if (/^H[1-4]$/.test(node.tagName)) break;
    if (node.tagName === "P") return node;
    const paragraph = node.querySelector?.("p");
    if (paragraph) return paragraph;
  }
  return null;
}

async function previewData(url) {
  const source = await documentFor(url);
  const id = url.hash ? decodeURIComponent(url.hash.slice(1)) : "";
  const target = id ? source.getElementById(id) : null;
  const object = target?.closest(".doc-object");
  const markdown = source.querySelector(".md");
  const heading = target?.matches("h1, h2, h3, h4")
    ? target
    : object?.querySelector(".doc-heading") || markdown?.querySelector("h1");
  const title = heading?.querySelector(".doc-object-name")?.textContent.trim()
    || heading?.childNodes[0]?.textContent.trim()
    || heading?.textContent.replace("¶", "").trim();
  if (!title) return null;

  const signature = object?.querySelector(":scope > .doc-signature code")?.textContent.trim() || "";
  const paragraph = object?.querySelector(":scope > .doc-contents > p")
    || paragraphAfter(heading)
    || markdown?.querySelector(":scope > p");
  return {
    title,
    signature,
    description: paragraph ? clipText(paragraph.textContent, 280) : "",
    path: `${decodeURIComponent(url.pathname)}${url.hash}`,
  };
}

function positionPreview(link) {
  const anchor = link.getBoundingClientRect();
  const card = linkPreview.getBoundingClientRect();
  let left = anchor.right + 12;
  let top = anchor.top;
  if (left + card.width > window.innerWidth - 8) left = anchor.left - card.width - 12;
  if (left < 8) {
    left = Math.max(8, Math.min(anchor.left, window.innerWidth - card.width - 8));
    top = anchor.bottom + 10;
  }
  top = Math.max(8, Math.min(top, window.innerHeight - card.height - 36));
  linkPreview.style.left = `${left}px`;
  linkPreview.style.top = `${top}px`;
  linkPreview.style.visibility = "visible";
}

function hidePreview(link = null) {
  if (link && link !== activePreviewLink) return;
  clearTimeout(previewTimer);
  activePreviewLink?.removeAttribute("aria-describedby");
  activePreviewLink = null;
  linkPreview.setAttribute("aria-hidden", "true");
}

function queuePreview(link, delay) {
  clearTimeout(previewTimer);
  activePreviewLink = link;
  previewTimer = setTimeout(async () => {
    try {
      const data = await previewData(new URL(link.href));
      if (!data || activePreviewLink !== link) return;
      previewPath.textContent = data.path;
      previewTitle.textContent = data.title;
      previewSignature.hidden = !data.signature;
      previewSignature.querySelector("code").textContent = data.signature;
      previewDescription.hidden = !data.description;
      previewDescription.textContent = data.description;
      linkPreview.style.visibility = "hidden";
      linkPreview.setAttribute("aria-hidden", "false");
      link.setAttribute("aria-describedby", "linkpreview");
      positionPreview(link);
    } catch { /* a preview must never interfere with navigation */ }
  }, delay);
}

const canHover = window.matchMedia("(hover: hover)").matches;
for (const link of document.querySelectorAll(".md a[href]")) {
  const url = new URL(link.href);
  if (url.origin !== window.location.origin
      || link.matches(".headerlink")
      || link.matches(".footnote-ref, .footnote-backref")
      || link.closest("pre, .doc-signature")
      || /\.[a-z0-9]+$/i.test(url.pathname)) continue;
  if (canHover) {
    link.addEventListener("pointerenter", () => queuePreview(link, 220));
    link.addEventListener("pointerleave", () => hidePreview(link));
  }
  link.addEventListener("focus", () => queuePreview(link, 0));
  link.addEventListener("blur", () => hidePreview(link));
}
window.addEventListener("resize", () => hidePreview());
window.addEventListener("scroll", () => hidePreview(), { passive: true });
document.getElementById("manscroll")?.addEventListener("scroll", () => hidePreview(), { passive: true });

/* ---------------- search ---------------- */
const searchBox = document.getElementById("search");
const searchQ = document.getElementById("mkdocs-search-query");
const searchButton = document.getElementById("searchbtn");
const base = document.body.dataset.base ? `${document.body.dataset.base}/` : "";
function openSearch() {
  hidePreview();
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
searchButton.addEventListener("click", () => {
  if (searchBox.classList.contains("on")) closeSearch();
  else openSearch();
});
document.addEventListener("pointerdown", (e) => {
  if (searchBox.classList.contains("on")
      && !searchBox.contains(e.target) && !searchButton.contains(e.target)) closeSearch();
});

/* ---------------- keyboard ---------------- */
const help = document.getElementById("help");
document.getElementById("helpbtn").addEventListener("click", () => help.classList.toggle("on"));
const scroller = () => document.getElementById("manscroll") || document.scrollingElement;
const go = (sel) => { const a = document.querySelector(sel); if (a) window.location.href = a.href; };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "Escape") {
    help.classList.remove("on");
    closeSearch();
    hidePreview();
    if (window.matchMedia("(max-width: 900px), (max-height: 560px)").matches) setOutline(false);
    return;
  }
  const man = scroller();
  switch (e.key) {
    case "t": flip(); break;
    case "e": toggleTree(); break;
    case "o": toggleOutline(); break;
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
