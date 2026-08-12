"use strict";

const assert = require("node:assert/strict");
const { execFileSync, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const root = path.resolve(__dirname, "..");
const playground = path.join(root, "site", "playground", "index.html");
const checkPage = path.join(root, "site", "playground", `.shell-check-${process.pid}.html`);
const navigationPage = path.join(root, "site", "playground", `.navigation-check-${process.pid}.html`);
const chrome = [process.env.CHROME_BIN, "google-chrome", "chromium", "chromium-browser"]
  .filter(Boolean)
  .find((candidate) => !spawnSync(candidate, ["--version"]).error);

assert.ok(chrome, "Chrome or Chromium is required for the documentation shell check");
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "advect-shell-"));
const dump = (page) => execFileSync(chrome, [
  "--headless=new",
  "--no-sandbox",
  "--disable-gpu",
  "--allow-file-access-from-files",
  `--user-data-dir=${profile}`,
  "--dump-dom",
  pathToFileURL(page).href,
], { encoding: "utf8" });

const check = `<script>
(() => {
  const failures = [];
  const check = (condition, label) => { if (!condition) failures.push(label); };
  const outline = document.getElementById("outlinebtn");
  const help = document.getElementById("help");
  const helpButton = document.getElementById("helpbtn");
  const search = document.getElementById("search");
  const searchButton = document.getElementById("searchbtn");
  const docsLink = document.querySelector('[data-short="[1]"]');

  check(outline.hidden, "outline hidden");
  helpButton.click();
  check(help.classList.contains("on"), "help click");
  helpButton.click();
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "?" }));
  check(help.classList.contains("on"), "help key");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
  searchButton.click();
  check(search.classList.contains("on"), "search click");
  searchButton.click();
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "/" }));
  check(search.classList.contains("on"), "search key");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
  check(docsLink.href === new URL("../", location.href).href, "docs target");
  check(help.textContent.includes("return to the docs"), "playground help");
  check(!help.textContent.includes("page outline"), "docs-only help hidden");
  document.documentElement.dataset.advectShellCheck = failures.length ? failures.join("|") : "passed";
})();
</script>`;

let html = fs.readFileSync(playground, "utf8")
  .replace('<script src="../js/examples.js"></script>', "")
  .replace('<script src="../search/main.js"></script>', "")
  .replace('<script type="module" src="../js/playground.js"></script>', "")
  .replace("</body>", `${check}\n</body>`);
const navigationHtml = html
  .replace(check, '<script>document.dispatchEvent(new KeyboardEvent("keydown", { key: "1" }));</script>');

try {
  fs.writeFileSync(checkPage, html);
  fs.writeFileSync(navigationPage, navigationHtml);
  html = dump(checkPage);
  assert.match(html, /data-advect-shell-check="passed"/);
  html = dump(navigationPage);
  assert.match(html, /<title id="title">Index of .*\/site\/<\/title>/);
  console.log("documentation shell checks: playground controls and keyboard navigation");
} finally {
  fs.rmSync(checkPage, { force: true });
  fs.rmSync(navigationPage, { force: true });
  fs.rmSync(profile, { force: true, recursive: true });
}
