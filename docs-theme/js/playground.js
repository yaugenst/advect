"use strict";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/";
const WHEEL_MANIFEST = "advect-browser-wheel.json";
const root = document.body.dataset.base || ".";
const assetUrl = (name) => new URL(`${root}/assets/${name}`, window.location.href).href;

let traceInPython = null;
let evaluateInPython = null;
let artifactInPython = null;
let loadInPython = null;

/* ---------------- session narration ---------------- */
const transcript = document.getElementById("transcript");
const esc = (value) => String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;");
function tline(html, cls) {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.innerHTML = `${html}\n`;
  transcript.appendChild(span);
  while (transcript.childNodes.length > 300) transcript.removeChild(transcript.firstChild);
  transcript.scrollTop = transcript.scrollHeight;
}
const narrate = (command) => tline(`<span class="ps">&gt;&gt;&gt;</span> ${esc(command)}`);
const continueLine = (command) => tline(`<span class="ps">...</span> ${esc(command)}`);
const pyFloat = (value) => (Number.isInteger(value) ? value.toFixed(1) : Number(value).toPrecision(8));

/* ---------------- braille plot ---------------- */
const COLS = 56;
const ROWS = 12;
const DX = COLS * 2;
const DY = ROWS * 4;
const XMIN = -4;
const XMAX = 4;
const state = {
  source: null,
  mode: "expr",
  graphMode: "expr",
  direction: "jvp",
  tapeView: "live",
  graph: null,
  report: null,
  loaded: false,
  expandedRuns: new Set(),
  x0: 0.55,
  ymin: -1,
  ymax: 1,
  ys: [],
  pinned: null,
  lastValue: NaN,
  lastDerivative: NaN,
  lastSecond: NaN,
  lastNarrated: NaN,
};
const grid = () => Array.from({ length: ROWS }, () => new Array(COLS).fill(0));
const bit = (dx, dy) => (dx === 0 ? (dy < 3 ? 1 << dy : 1 << 6) : (dy < 3 ? 1 << (3 + dy) : 1 << 7));
function setDot(target, x, y) {
  if (x < 0 || x >= DX || y < 0 || y >= DY || !Number.isFinite(y)) return;
  target[(y / 4) | 0][(x / 2) | 0] |= bit(x & 1, y & 3);
}
function renderGrid(target, overlay) {
  let output = "";
  for (let row = 0; row < ROWS; row += 1) {
    for (let column = 0; column < COLS; column += 1) {
      const mark = overlay && overlay[row] && overlay[row][column];
      output += mark || (target[row][column] ? String.fromCharCode(0x2800 + target[row][column]) : " ");
    }
    output += "\n";
  }
  return output;
}
const ydot = (value) => Math.round((state.ymax - value) / (state.ymax - state.ymin) * (DY - 1));
const xval = (index) => XMIN + (index + 0.5) / DX * (XMAX - XMIN);
const xdot = (value) => Math.round((value - XMIN) / (XMAX - XMIN) * (DX - 1));
const fmt = (value) => `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;

function drawSeries(values, target) {
  let previous = null;
  for (let index = 0; index < DX; index += 1) {
    if (values[index] === null) {
      previous = null;
      continue;
    }
    const dot = ydot(values[index]);
    if (previous !== null) {
      for (let row = Math.min(previous, dot); row <= Math.max(previous, dot); row += 1) {
        setDot(target, index, row);
      }
    } else {
      setDot(target, index, dot);
    }
    previous = dot;
  }
}

function drawAxes() {
  const axes = grid();
  const zeroRow = ydot(0);
  for (let column = 0; column < DX; column += 4) setDot(axes, column, zeroRow);
  for (let value = XMIN + 1; value < XMAX; value += 1) {
    const column = Math.round((value - XMIN) / (XMAX - XMIN) * (DX - 1));
    for (let row = 0; row < DY; row += 6) setDot(axes, column, row);
  }
  document.getElementById("lgrid").textContent = renderGrid(axes);

  const labels = new Array(COLS).fill(" ");
  for (const value of [-4, -2, 0, 2, 4]) {
    const position = Math.round((value - XMIN) / (XMAX - XMIN) * (COLS - 1));
    const label = `${value > 0 ? "+" : ""}${value}`;
    for (let index = 0; index < label.length; index += 1) {
      const target = Math.max(0, position - 1) + index;
      if (target < COLS) labels[target] = label[index];
    }
  }
  document.getElementById("xlabels").textContent = labels.join("");
}

function redraw() {
  const finite = state.ys.filter((value) => value !== null);
  if (finite.length < 8) {
    showError("ValueError: f is not finite on enough of [-4, 4] to plot");
    return;
  }
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  const padding = (high - low) * 0.15 + 1e-9;
  state.ymin = low - padding;
  state.ymax = high + padding;

  drawAxes();
  const curve = grid();
  drawSeries(state.ys, curve);
  document.getElementById("lcurve").textContent = renderGrid(curve);
  updatePosition();
}

function updatePosition() {
  if (!evaluateInPython) return;
  try {
    const [value, derivative, second] = JSON.parse(evaluateInPython(state.x0));
    state.lastValue = value;
    state.lastDerivative = derivative;
    state.lastSecond = second;

    const radius = 1.35;
    const tangent = grid();
    if (value !== null && derivative !== null) {
      for (let index = 0; index < DX; index += 1) {
        const x = xval(index);
        if (x >= state.x0 - radius && x <= state.x0 + radius) {
          setDot(tangent, index, ydot(value + derivative * (x - state.x0)));
        }
      }
    }
    const parabola = grid();
    if (value !== null && derivative !== null && second !== null) {
      for (let index = 0; index < DX; index += 1) {
        const x = xval(index);
        if (x >= state.x0 - radius && x <= state.x0 + radius) {
          const dx = x - state.x0;
          setDot(parabola, index, ydot(value + derivative * dx + 0.5 * second * dx * dx));
        }
      }
    }
    document.getElementById("lpara").textContent = renderGrid(parabola);

    const column = (xdot(state.x0) / 2) | 0;
    const row = value === null ? -1 : (ydot(value) / 4) | 0;
    const overlay = {};
    if (value !== null) {
      overlay[row] = {};
      overlay[row][column] = "●";
    }
    document.getElementById("ltan").textContent = renderGrid(tangent, overlay);
    document.getElementById("rx").textContent = fmt(state.x0);
    document.getElementById("rf").textContent = value === null ? "nan" : value.toFixed(4);
    document.getElementById("rg").textContent = derivative === null ? "nan" : derivative.toFixed(4);
    document.getElementById("rh").textContent = second === null ? "—" : second.toFixed(4);
    document.getElementById("statpos").textContent = `x₀ ${fmt(state.x0)}`;
  } catch (error) {
    showError(errorText(error));
  }
}

/* ---------------- errors ---------------- */
const errorElement = document.getElementById("ferr");
function showError(message) {
  errorElement.textContent = message;
  errorElement.style.display = "block";
}
function clearError() {
  errorElement.style.display = "none";
}
function errorText(error) {
  return String(error && error.message ? error.message : error).trim();
}

/* ---------------- graph + program panes ---------------- */
const tapeElement = document.getElementById("tape");
const treeElement = document.getElementById("ctree");
const mirrorElement = document.getElementById("finmirror");
const defMirrorElement = document.getElementById("fdefmirror");
const functionInput = document.getElementById("fin");
const functionDef = document.getElementById("fdef");
const defWrap = document.getElementById("fdefwrap");

/* paint the source spans linked to a node under the active editor; span
   offsets are only valid for the source the current graph was traced from */
function renderMirror(ranges) {
  const isDef = state.mode === "def";
  const target = isDef ? defMirrorElement : mirrorElement;
  (isDef ? mirrorElement : defMirrorElement).innerHTML = "";
  if (!ranges.length || state.graphMode !== state.mode) {
    target.innerHTML = "";
    return;
  }
  const source = isDef ? functionDef.value : functionInput.value;
  const sorted = ranges.slice().sort((a, b) => a[0] - b[0]);
  let html = "";
  let position = 0;
  for (const [start, end] of sorted) {
    if (start < position) continue;
    html += `${esc(source.slice(position, start))}<mark>${esc(source.slice(start, end))}</mark>`;
    position = end;
  }
  html += esc(source.slice(position));
  /* the trailing newline keeps the mirror's scrollable height equal to the
     textarea's, so scrollTop sync stays aligned on the last line */
  target.innerHTML = isDef ? `${html}\n` : html;
  /* the mirror was empty until now, so it must adopt the editor's current
     scroll offset — the scroll listener alone only covers later scrolling */
  if (isDef) target.scrollTop = functionDef.scrollTop;
  else target.scrollLeft = functionInput.scrollLeft;
}

function foldBadge(node) {
  const count = state.graph && state.graph.folds ? state.graph.folds[String(node)] : undefined;
  return count ? ` <span class="fold">×${count}</span>` : "";
}

function lineClasses(line) {
  const classes = ["irline", `role-${line.role || "none"}`];
  if (line.shared) classes.push("shr");
  if (state.pinned === line.node) classes.push("sel");
  return classes.join(" ");
}

function renderLive() {
  return state.graph.program.map((line) => {
    const body = esc(line.text).replace(/^(%\d+|return)/, '<span class="b">$1</span>');
    return line.node === undefined
      ? `<span class="irplain">${body}</span>`
      : `<span class="${lineClasses(line)}" data-n="${line.node}">${body}${foldBadge(line.node)}</span>`;
  }).join("");
}

function renderTraced() {
  const trace = state.graph.trace;
  if (!trace) return '<span class="irplain dim"># no trace: loaded artifacts carry only the optimized graph</span>';
  const rows = trace.rows;
  const pieces = [];
  let index = 0;
  while (index < rows.length) {
    const row = rows[index];
    if (row.status === "dead") {
      let runEnd = index;
      while (runEnd < rows.length && rows[runEnd].status === "dead") runEnd += 1;
      const length = runEnd - index;
      if (length > 3 && !state.expandedRuns.has(index)) {
        pieces.push(`<span class="irline irrun" data-run="${index}">   ⋯ ${length} traced nodes removed (dce) — click to expand ⋯</span>`);
        index = runEnd;
        continue;
      }
      for (let at = index; at < runEnd; at += 1) {
        const dead = rows[at];
        const body = esc(dead.text).replace(/^(%\d+)/, '<span class="b">$1</span>');
        pieces.push(`<span class="irline dead">${body} <span class="irnote"># removed</span></span>`);
      }
      index = runEnd;
      continue;
    }
    const body = esc(row.text).replace(/^(%\d+)/, '<span class="b">$1</span>');
    if (row.status === "merged") {
      pieces.push(`<span class="irline merged" data-n="${row.new}">${body} <span class="irnote">→ %${row.into} (cse)</span></span>`);
    } else {
      pieces.push(`<span class="irline role-${row.role || "none"}" data-n="${row.new}">${body}</span>`);
    }
    index += 1;
  }
  return pieces.join("");
}

function renderProgram() {
  tapeElement.innerHTML = state.tapeView === "traced" ? renderTraced() : renderLive();
}

function renderTree() {
  const headers = state.graph.headers || [];
  let html = "";
  state.graph.tree.forEach((group, index) => {
    if (index > 0) html += '<span class="irplain"> </span>';
    html += `<span class="irplain dim">${esc(headers[index] || `# output[${index}]`)}</span>`;
    for (const line of group) {
      const reference = line.shared ? "↖ " : "";
      html += `<span class="${lineClasses(line)}" data-n="${line.node}">` +
        `${esc(line.pre)}${reference}<span class="b">${esc(line.label)}</span> ` +
        `<span class="tid">%${line.node}</span></span>`;
    }
  });
  treeElement.innerHTML = html;
}

function renderReport() {
  const report = state.report;
  const passes = report.passes
    .map((item) => {
      const dropped = item.nodes_before - item.nodes_after;
      return dropped
        ? `${item.name} <span class="b">−${dropped}</span>`
        : `<span class="dim">${item.name} −0</span>`;
    })
    .join(" · ");
  const ms = (seconds) => (seconds * 1000).toFixed(1);
  const timing = report.eval_seconds === undefined ? "" :
    `<span class="dim"> · staged in ${ms(report.compile_seconds)} ms · ${report.samples} evals in ${ms(report.eval_seconds)} ms</span>`;
  const label = state.direction === "vjp" ? "vjp_program" : "stage(jvp)";
  document.getElementById("report").innerHTML =
    `${label}: ${report.nodes_before} traced → ${passes} → <span class="b">${report.nodes_after} live</span>${timing}`;
}

function renderLegend() {
  const fold = ' · <span class="warn">×n</span> <span class="dim">= n source sites, one node (cse)</span>';
  document.getElementById("legend").innerHTML = state.direction === "vjp"
    ? 'chains: x · <span class="acc">adjoint</span> · <span class="warn">reused primal work</span>' + fold
    : 'chains: f(x) · <span class="acc">df/dx</span> · <span class="warn">shared</span>' + fold;
}

function renderPanes() {
  renderProgram();
  renderTree();
  renderReport();
  renderLegend();
  document.getElementById("tapeToggle").textContent = `[ tape: ${state.tapeView} ]`;
  document.getElementById("dirToggle").textContent = `[ graph: ${state.direction} ]`;
}

function clearLinks() {
  document.querySelectorAll("#tape .hl, #tape .peer, #ctree .hl, #ctree .peer")
    .forEach((element) => element.classList.remove("hl", "peer"));
  renderMirror([]);
}

function linkNode(node, origin) {
  clearLinks();
  if (node === null) return;
  const spans = state.graph && state.graph.spans ? state.graph.spans[String(node)] : undefined;
  if (spans) renderMirror(spans);
  const originInTree = treeElement.contains(origin);
  document.querySelectorAll(`#tape [data-n="${node}"], #ctree [data-n="${node}"]`)
    .forEach((element) => {
      if (element === origin || (originInTree && tapeElement.contains(element))) {
        element.classList.add("hl");
      } else {
        element.classList.add("peer");
      }
    });
}

function wireLinks(container) {
  container.addEventListener("mouseover", (event) => {
    const element = event.target.closest("[data-n]");
    linkNode(element ? Number(element.dataset.n) : null, element);
  });
  container.addEventListener("mouseleave", clearLinks);
  container.addEventListener("click", (event) => {
    const run = event.target.closest(".irrun");
    if (run) {
      state.expandedRuns.add(Number(run.dataset.run));
      renderProgram();
      return;
    }
    const element = event.target.closest("[data-n]");
    if (!element) return;
    const node = Number(element.dataset.n);
    state.pinned = state.pinned === node ? null : node;
    renderPanes();
  });
}
wireLinks(tapeElement);
wireLinks(treeElement);

/* ---------------- tracing ---------------- */
const currentSource = () => (state.mode === "def" ? functionDef.value : functionInput.value.trim());

function narrateSource(source) {
  if (state.mode === "def") {
    const lines = source.split("\n");
    narrate(lines[0]);
    for (const line of lines.slice(1)) {
      if (line.trim()) continueLine(line);
    }
  } else {
    narrate(`f = lambda x: ${source}`);
  }
}

let traceGeneration = 0;
async function trace(source, narrated = true) {
  if (!traceInPython) return false;
  const generation = ++traceGeneration;
  statboot.textContent = "tracing…";
  await new Promise((resolve) => requestAnimationFrame(resolve));
  try {
    const result = JSON.parse(traceInPython(source, state.mode, state.direction));
    if (generation !== traceGeneration) return false;
    Object.assign(state, {
      source,
      graphMode: result.mode,
      graph: result.graph,
      report: result.report,
      ys: result.series.values,
      pinned: null,
      loaded: false,
      expandedRuns: new Set(),
    });
    clearError();
    if (narrated) {
      narrateSource(source);
      narrate("df = ad.jvp(f)");
      narrate("program = ad.stage(lambda x: df(x, tangents=np.asarray(1.0)), np.asarray(0.0))");
      if (state.direction === "vjp") {
        narrate("pullback = ad.vjp_program(ad.stage(f, np.asarray(0.0)))");
      }
    }
    renderPanes();
    redraw();
    narrateGradient(true);
    document.getElementById("bootmeta").innerHTML =
      `<b>advect ${esc(result.runtime.advect)}</b><br>Pyodide · NumPy ${esc(result.runtime.numpy)}<br>real Python API`;
    statboot.textContent = "Python · live";
    return true;
  } catch (error) {
    showError(errorText(error));
    statboot.textContent = "trace error";
    return false;
  }
}

function narrateGradient(force = false) {
  updatePosition();
  if (!Number.isFinite(state.lastDerivative)) return;
  if (!force && Math.abs(state.x0 - state.lastNarrated) < 0.01) return;
  state.lastNarrated = state.x0;
  narrate(`program(np.asarray(${state.x0.toFixed(2)}))`);
  tline(`(${pyFloat(state.lastValue)}, ${pyFloat(state.lastDerivative)})`, "out acc");
}

/* ---------------- autocomplete ---------------- */
/* candidates come from the adapter's real namespaces, fetched after boot */
let completionNames = null;
const completeBox = document.createElement("div");
completeBox.className = "complete";
completeBox.hidden = true;
document.body.appendChild(completeBox);
const completion = { items: [], index: 0, word: "", editor: null };

function closeComplete() {
  completeBox.hidden = true;
  completion.items = [];
}

function renderComplete() {
  completeBox.innerHTML = completion.items
    .map((name, index) =>
      `<span class="ci${index === completion.index ? " on" : ""}" data-i="${index}">` +
      `<b>${esc(name.slice(0, completion.word.length))}</b>${esc(name.slice(completion.word.length))}</span>`)
    .join("");
}

/* both editors are monospace, so the caret's pixel position is arithmetic */
const measureCanvas = document.createElement("canvas");
function caretPoint(editor) {
  const style = getComputedStyle(editor);
  const context = measureCanvas.getContext("2d");
  context.font = style.font;
  const charWidth = context.measureText("0").width;
  const before = editor.value.slice(0, editor.selectionStart);
  const lines = before.split("\n");
  const column = lines[lines.length - 1].length;
  const rectangle = editor.getBoundingClientRect();
  const x = rectangle.left + parseFloat(style.borderLeftWidth) + parseFloat(style.paddingLeft)
    + column * charWidth - editor.scrollLeft;
  if (editor.tagName === "INPUT") return { x, y: rectangle.bottom + 3 };
  const lineHeight = parseFloat(style.lineHeight);
  const y = rectangle.top + parseFloat(style.borderTopWidth) + parseFloat(style.paddingTop)
    + lines.length * lineHeight - editor.scrollTop + 2;
  return { x, y };
}

function updateComplete(editor) {
  if (!completionNames || editor.selectionStart !== editor.selectionEnd) return closeComplete();
  const before = editor.value.slice(0, editor.selectionStart);
  const match = /(?:\b(np)\.)?([A-Za-z_][A-Za-z0-9_]*)?$/.exec(before);
  const qualified = Boolean(match[1]);
  const word = match[2] || "";
  if (!word && !qualified) return closeComplete();
  let pool;
  if (state.mode === "expr") {
    if (qualified) return closeComplete();
    pool = completionNames.expr;
  } else {
    pool = qualified ? completionNames.def.np : completionNames.def.plain;
  }
  const items = pool.filter((name) => name.startsWith(word) && name !== word).slice(0, 8);
  if (!items.length) return closeComplete();
  Object.assign(completion, { items, index: 0, word, editor });
  const point = caretPoint(editor);
  completeBox.style.left = `${Math.round(point.x)}px`;
  completeBox.style.top = `${Math.round(point.y)}px`;
  completeBox.hidden = false;
  renderComplete();
  const overflow = completeBox.getBoundingClientRect().right - (window.innerWidth - 8);
  if (overflow > 0) completeBox.style.left = `${Math.round(point.x - overflow)}px`;
}

function acceptComplete() {
  const editor = completion.editor;
  const name = completion.items[completion.index];
  const caret = editor.selectionStart;
  editor.setRangeText(name.slice(completion.word.length), caret, caret, "end");
  closeComplete();
  editor.dispatchEvent(new Event("input"));
}

completeBox.addEventListener("mousedown", (event) => {
  const item = event.target.closest(".ci");
  if (!item) return;
  event.preventDefault();
  completion.index = Number(item.dataset.i);
  acceptComplete();
});

/* registered before the editors' own handlers, so an open menu owns the keys */
for (const editor of [functionInput, functionDef]) {
  editor.addEventListener("input", () => updateComplete(editor));
  editor.addEventListener("click", closeComplete);
  editor.addEventListener("blur", () => setTimeout(closeComplete, 120));
  editor.addEventListener("keydown", (event) => {
    if (completeBox.hidden) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const step = event.key === "ArrowDown" ? 1 : completion.items.length - 1;
      completion.index = (completion.index + step) % completion.items.length;
      renderComplete();
    } else if (event.key === "Enter" || event.key === "Tab") {
      acceptComplete();
    } else if (event.key === "Escape") {
      closeComplete();
    } else {
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
  });
}

/* ---------------- input modes ---------------- */
const modeToggle = document.getElementById("pymode");
const DEF_DEFAULT = `def bump(x):
    return np.exp(-x**2 / 4)

def f(x):
    return np.sin(3 * x) * bump(x)
`;
function setMode(mode) {
  state.mode = mode;
  const isDef = mode === "def";
  defWrap.hidden = !isDef;
  functionInput.closest(".fxrow").style.display = isDef ? "none" : "";
  modeToggle.textContent = isDef ? "[ expression ]" : "[ python ]";
  mirrorElement.innerHTML = "";
  defMirrorElement.innerHTML = "";
  closeComplete();
}
setMode("expr");
modeToggle.addEventListener("click", () => {
  if (!traceInPython) return;
  if (state.mode === "expr") {
    if (!functionDef.value.trim()) functionDef.value = DEF_DEFAULT;
    setMode("def");
    functionDef.focus();
    trace(functionDef.value);
  } else {
    setMode("expr");
    trace(functionInput.value.trim());
  }
});

let debounceTimer = null;
functionInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    const source = functionInput.value.trim();
    if (source !== state.source) trace(source);
  }, 500);
});
functionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    clearTimeout(debounceTimer);
    trace(functionInput.value.trim());
    functionInput.blur();
  }
  event.stopPropagation();
});
functionDef.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    if (functionDef.value !== state.source) trace(functionDef.value);
  }, 700);
});
functionDef.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    clearTimeout(debounceTimer);
    trace(functionDef.value);
  } else if (event.key === "Enter") {
    // auto-indent: keep the current line's indentation, deepen after ":"
    event.preventDefault();
    const { selectionStart, selectionEnd, value } = functionDef;
    const lineStart = value.lastIndexOf("\n", selectionStart - 1) + 1;
    const line = value.slice(lineStart, selectionStart);
    const indent = line.match(/^[ \t]*/)[0] + (line.trimEnd().endsWith(":") ? "    " : "");
    functionDef.setRangeText(`\n${indent}`, selectionStart, selectionEnd, "end");
    functionDef.dispatchEvent(new Event("input"));
  } else if (event.key === "Tab") {
    event.preventDefault();
    const { selectionStart, selectionEnd } = functionDef;
    functionDef.setRangeText("    ", selectionStart, selectionEnd, "end");
    functionDef.dispatchEvent(new Event("input"));
  }
  event.stopPropagation();
});
functionDef.addEventListener("scroll", () => {
  defMirrorElement.scrollTop = functionDef.scrollTop;
  closeComplete();
});

/* ---------------- pane [c] controls ---------------- */
document.getElementById("tapeToggle").addEventListener("click", () => {
  state.tapeView = state.tapeView === "live" ? "traced" : "live";
  if (state.tapeView === "traced") narrate("program.trace   # the tape before dce/simplify/cse");
  renderPanes();
});
document.getElementById("dirToggle").addEventListener("click", () => {
  if (!traceInPython || state.loaded) return;
  state.direction = state.direction === "jvp" ? "vjp" : "jvp";
  narrate(state.direction === "vjp"
    ? "pullback = ad.vjp_program(ad.stage(f, np.asarray(0.0)))"
    : "program = ad.stage(lambda x: df(x, tangents=np.asarray(1.0)), np.asarray(0.0))");
  trace(currentSource(), false);
});
document.getElementById("save").addEventListener("click", () => {
  if (!artifactInPython) return;
  const text = artifactInPython();
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "f.advect.json";
  anchor.click();
  URL.revokeObjectURL(url);
  narrate('json.dump(program.to_dict(), open("f.advect.json", "w"))');
});
document.getElementById("loadfile").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file || !loadInPython) return;
  narrate(`program = ad.StagedProgram.from_dict(json.load(open("${file.name}")))`);
  try {
    const result = JSON.parse(loadInPython(await file.text()));
    Object.assign(state, {
      graph: result.graph,
      report: result.report,
      direction: result.direction,
      pinned: null,
      loaded: true,
      expandedRuns: new Set(),
    });
    renderPanes();
    tline(esc(`# ${result.inputs} input(s), ${result.outputs} output(s) — loaded for inspection; the plot still shows the traced f`), "cmt");
  } catch (error) {
    tline(`ArtifactError: ${esc(errorText(error))}`, "err");
  }
  event.target.value = "";
});

/* ---------------- plot pointer + keys ---------------- */
const screen = document.getElementById("screen");
let dragging = false;
let pendingFrame = false;
const toX = (event) => {
  const rectangle = screen.getBoundingClientRect();
  const fraction = (event.clientX - rectangle.left - 10) / (rectangle.width - 20);
  return Math.max(XMIN, Math.min(XMAX, XMIN + fraction * (XMAX - XMIN)));
};
function movePoint(event) {
  state.x0 = toX(event);
  if (pendingFrame) return;
  pendingFrame = true;
  requestAnimationFrame(() => {
    pendingFrame = false;
    updatePosition();
  });
}
screen.addEventListener("pointerdown", (event) => {
  if (!evaluateInPython) return;
  dragging = true;
  screen.setPointerCapture(event.pointerId);
  movePoint(event);
});
screen.addEventListener("pointermove", (event) => {
  if (dragging) movePoint(event);
});
screen.addEventListener("pointerup", () => {
  dragging = false;
  narrateGradient();
});

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA" || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === "ArrowLeft") state.x0 = Math.max(XMIN, state.x0 - 0.05);
  else if (event.key === "ArrowRight") state.x0 = Math.min(XMAX, state.x0 + 0.05);
  else if (event.key === "o") { document.getElementById("tapeToggle").click(); return; }
  else if (event.key === "v") { document.getElementById("dirToggle").click(); return; }
  else return;
  updatePosition();
  event.preventDefault();
});

/* ---------------- dot-matrix logo ---------------- */
(function drawLogo() {
  const columns = 20;
  const rows = 6;
  const target = () => Array.from({ length: rows }, () => new Array(columns).fill(0));
  const put = (buffer, x, y) => {
    if (x >= 0 && x < columns * 2 && y >= 0 && y < rows * 4) {
      buffer[(y / 4) | 0][(x / 2) | 0] |= bit(x & 1, y & 3);
    }
  };
  const draw = (buffer) => buffer
    .map((row) => row.map((value) => (value ? String.fromCharCode(0x2800 + value) : " ")).join(""))
    .join("\n");
  const ink = target();
  const accent = target();
  for (let angle = 0; angle < Math.PI * 2; angle += 0.02) {
    put(ink, Math.round(11 + 9 * Math.cos(angle)), Math.round(12 + 8.6 * Math.sin(angle)));
  }
  for (let y = 3; y <= 21; y += 1) {
    put(ink, 24, y);
    put(ink, 25, y);
  }
  for (let y = -3; y <= 3; y += 1) {
    for (let x = -3; x <= 3; x += 1) {
      if (x * x + y * y <= 6.5) put(accent, 21 + x, 12 + y);
    }
  }
  document.getElementById("logoInk").textContent = draw(ink);
  document.getElementById("logoAcc").textContent = draw(accent);
}());

/* ---------------- boot ---------------- */
const bar = document.querySelector(".bar");
const statboot = document.createElement("span");
statboot.id = "statboot";
statboot.className = "hidesm";
statboot.textContent = "loading Python…";
bar.insertBefore(statboot, bar.querySelector(".grow"));
const statpos = document.createElement("span");
statpos.id = "statpos";
statpos.hidden = true;
bar.insertBefore(statpos, document.getElementById("searchbtn"));

drawAxes();

(async () => {
  try {
    tline('<span class="cmt">$ advect playground</span>');
    const [{ loadPyodide }, runtimeResponse, wheelResponse] = await Promise.all([
      import(`${PYODIDE_URL}pyodide.mjs`),
      fetch(assetUrl("playground_runtime.py")),
      fetch(assetUrl(WHEEL_MANIFEST)),
    ]);
    if (!runtimeResponse.ok) throw new Error("playground Python adapter is missing");
    if (!wheelResponse.ok) throw new Error("Advect browser wheel manifest is missing");
    const runtimeSource = await runtimeResponse.text();
    const { filename: wheel } = await wheelResponse.json();
    const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
    statboot.textContent = "loading NumPy…";
    await pyodide.loadPackage(["micropip", "numpy"]);
    statboot.textContent = "loading Advect…";
    const wheelUrl = assetUrl(wheel);
    await pyodide.runPythonAsync(`import micropip\nawait micropip.install(${JSON.stringify(wheelUrl)})`);
    pyodide.runPython(runtimeSource, { filename: "playground_runtime.py" });
    traceInPython = pyodide.globals.get("playground_trace_json");
    evaluateInPython = pyodide.globals.get("playground_evaluate_json");
    artifactInPython = pyodide.globals.get("playground_artifact_json");
    loadInPython = pyodide.globals.get("playground_load_json");
    completionNames = JSON.parse(pyodide.globals.get("playground_names_json")());
    narrate("import advect as ad");
    narrate("import numpy as np");
    tline("# values, derivatives, graphs, and traces below come from Advect in Pyodide", "cmt");
    await trace(functionInput.value.trim());
    functionInput.disabled = false;
    statpos.hidden = false;
  } catch (error) {
    showError(`BootError: ${errorText(error)}`);
    statboot.textContent = "Python failed to load";
    tline(esc(errorText(error)), "err");
  }
})();

window.__playground = { state, trace };
