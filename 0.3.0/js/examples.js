"use strict";

/* Runnable snippets: any fenced block marked ```{.python .run} gets a
   [ run ] chip. The page is one Python session, like typing the tutorial
   into a REPL: running a block first executes any not-yet-run marked blocks
   above it (once each, in order), then shows this block's output. Re-running
   an earlier block re-executes just that block in the same session. Pyodide
   boots lazily on the first click. */
(function () {
  const blocks = [...document.querySelectorAll(".highlight.run")];
  if (!blocks.length) return;

  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/";
  const WHEEL_MANIFEST = "advect-browser-wheel.json";
  const root = document.body.dataset.base || ".";
  const assetUrl = (name) => new URL(`${root}/assets/${name}`, window.location.href).href;

  let bootPromise = null;
  function boot(status) {
    if (!bootPromise) {
      bootPromise = (async () => {
        status("loading Python — downloads once, ~15 s…");
        const [{ loadPyodide }, wheelResponse] = await Promise.all([
          import(`${PYODIDE_URL}pyodide.mjs`),
          fetch(assetUrl(WHEEL_MANIFEST)),
        ]);
        if (!wheelResponse.ok) throw new Error("Advect browser wheel manifest is missing");
        const { filename: wheel } = await wheelResponse.json();
        const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
        status("loading NumPy…");
        await pyodide.loadPackage(["micropip", "numpy"]);
        status("loading Advect…");
        await pyodide.runPythonAsync(
          `import micropip\nawait micropip.install(${JSON.stringify(assetUrl(wheel))})`,
        );
        return pyodide;
      })();
      bootPromise.catch(() => { bootPromise = null; });
    }
    return bootPromise;
  }

  /* read the code element, not the pre: theme.js appends a [copy] button
     to the pre, which must not leak into the executed source */
  const sourceOf = (block) => block.querySelector("pre > code").textContent;

  const session = { namespace: null, executed: 0 };

  async function run(index, status) {
    const pyodide = await boot(status);
    if (!session.namespace) session.namespace = pyodide.globals.get("dict")();
    const pending = index - session.executed;
    const queue = [];
    for (let at = Math.min(index, session.executed); at < index; at += 1) queue.push(at);
    queue.push(index);
    status(pending > 0
      ? `running — ${pending} earlier snippet${pending > 1 ? "s" : ""} first…`
      : "running…");
    const lines = [];
    const capture = { batched: (line) => lines.push(line) };
    const discard = { batched: () => {} };
    for (const at of queue) {
      const last = at === index;
      pyodide.setStdout(last ? capture : discard);
      pyodide.setStderr(last ? capture : discard);
      try {
        await pyodide.runPythonAsync(sourceOf(blocks[at]), { globals: session.namespace });
      } catch (error) {
        const message = String(error && error.message ? error.message : error).trim();
        lines.push(message);
        status(last ? "error" : `error in the earlier snippet ${at + 1}`);
        return lines;
      }
      if (at >= session.executed) session.executed = at + 1;
    }
    status("done — ran in your browser");
    if (!lines.length) lines.push("# no output — ran to completion");
    return lines;
  }

  blocks.forEach((block, index) => {
    /* chips straddle the block's own bottom border (.highlight is the
       positioning context); the output panel follows the block */
    const button = document.createElement("button");
    button.className = "tbtn exgo";
    button.textContent = "[ run ]";
    const statusEl = document.createElement("span");
    statusEl.className = "exstat dim";
    const output = document.createElement("pre");
    output.className = "exout";
    output.hidden = true;
    block.append(statusEl, button);
    block.after(output);
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const lines = await run(index, (text) => { statusEl.textContent = text; });
        output.textContent = `${lines.join("\n")}\n`;
        output.hidden = false;
      } catch (error) {
        statusEl.textContent = `BootError: ${String(error).trim()}`;
      } finally {
        button.disabled = false;
      }
    });
  });
}());
