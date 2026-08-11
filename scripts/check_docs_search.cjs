"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const searchDir = path.join(root, "site", "search");
const lunr = require(path.join(searchDir, "lunr.js"));
const worker = require(path.join(root, "docs-theme", "search", "worker.js"));
const data = JSON.parse(fs.readFileSync(path.join(searchDir, "search_index.json"), "utf8"));

worker.buildIndex(data, lunr);
const docs = worker.searchableDocuments(data.docs);
const normalize = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
const compact = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, "");
const pageTitles = new Map(
  data.docs.filter((doc) => !doc.location.includes("#")).map((doc) => [doc.location, doc.title]),
);

function assertFirst(query, location) {
  assert.equal(worker.search(query)[0]?.location, location, query);
}

const representative = [
  ["numpy.fft.fft", "compatibility/numpy/#functions"],
  ["value_and_grad", "api/transforms/#advect.value_and_grad"],
  ["serializ", "tutorials/staging/#staging-and-serialization"],
  ["does advect support fft", "compatibility/numpy/#functions"],
];
for (const [query, location] of representative) assertFirst(query, location);

const titleGroups = new Map();
for (const doc of docs) {
  const key = normalize(doc.title);
  if (!titleGroups.has(key)) titleGroups.set(key, []);
  titleGroups.get(key).push(doc);
}
let titleChecks = 0;
for (const group of titleGroups.values()) {
  if (group.length !== 1 || !normalize(group[0].title)) continue;
  assertFirst(group[0].title, group[0].location);
  titleChecks += 1;
}
assert.ok(titleChecks > 0);

let qualifiedChecks = 0;
for (const doc of docs) {
  const anchor = doc.location.includes("#")
    ? decodeURIComponent(doc.location.split("#")[1])
    : "";
  if (!anchor.startsWith("advect.")) continue;
  assertFirst(anchor, doc.location);
  qualifiedChecks += 1;
}
assert.ok(qualifiedChecks > 0);

const contextGroups = new Map();
for (const doc of docs) {
  if (!doc.location.includes("#")) continue;
  const pageTitle = pageTitles.get(doc.location.split("#")[0]);
  if (!pageTitle) continue;
  const query = `${pageTitle} ${doc.title}`;
  const key = normalize(query);
  if (!contextGroups.has(key)) contextGroups.set(key, []);
  contextGroups.get(key).push({ query, location: doc.location });
}
let contextChecks = 0;
for (const group of contextGroups.values()) {
  if (group.length !== 1) continue;
  const locations = worker.search(group[0].query).slice(0, 3).map((result) => result.location);
  assert.ok(locations.includes(group[0].location), group[0].query);
  contextChecks += 1;
}
assert.ok(contextChecks > 0);

let prefixChecks = 0;
for (const doc of docs) {
  const words = doc.title.match(/[A-Za-z0-9]+/g) || [];
  const last = words.at(-1) || "";
  if (last.length < 6) continue;
  const query = [...words.slice(0, -1), last.slice(0, -2)].join(" ");
  const key = compact(query);
  const candidates = docs.filter((candidate) => compact(candidate.title).startsWith(key));
  if (candidates.length !== 1 || candidates[0].location !== doc.location) continue;
  assertFirst(query, doc.location);
  prefixChecks += 1;
}
assert.ok(prefixChecks > 0);

assert.match(worker.search("numpy.fft.fft")[0].summary, /numpy\.fft\.fft/i);
assert.deepEqual(worker.search("completely unrelated"), []);

const gradientEntrypoints = new Set([
  "#your-first-gradient",
  "tutorials/gradients/#gradients-and-pytrees",
]);
assert.ok(worker.search("show me gradient examples").slice(0, 3)
  .some((result) => gradientEntrypoints.has(result.location)));

const treeMapSummary = worker.search("how to use tree_map")[0].summary;
const treeMapVariadic = "tree_map ( f : Any , tree : Any , / , * rest : Any ) -> Any";
assert.equal(treeMapSummary.split(treeMapVariadic).length - 1, 1, "tree_map duplicate signature");

assert.ok([
  "compatibility/numpy/#functions",
  "compatibility/array-api/#array-api-202212-baseline",
].includes(worker.search("all")[0]?.location), "all");

const broadResults = worker.search("array api");
assert.equal(broadResults.length, 12);
assert.equal(new Set(broadResults.map((result) => result.location)).size, broadResults.length);
assert.doesNotThrow(() => worker.search("value:grad"));

console.log(
  `search checks: ${representative.length} representative, ${titleChecks} titles, `
  + `${qualifiedChecks} qualified anchors, ${contextChecks} contexts, `
  + `${prefixChecks} prefixes`,
);
