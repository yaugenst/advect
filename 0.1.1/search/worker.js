/* Advect's tuned MkDocs/Lunr search worker. */
"use strict";

var index;
var searchEngine;
var documents = Object.create(null);
var documentSymbols = Object.create(null);
var knownSymbolLeaves = new Set();
var pageTitles = Object.create(null);

var MAX_RESULTS = 12;
var QUERY_QUALIFIER_STEMS = new Set(["doc", "document", "guid", "refer", "instruct", "manual"]);

function locationParts(location) {
  var hash = location.indexOf("#");
  return {
    page: hash < 0 ? location : location.slice(0, hash),
    anchor: hash < 0 ? "" : decodeURIComponent(location.slice(hash + 1)),
  };
}

function identifierAliases(value) {
  var words = value
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2");
  return words === value ? value : value + " " + words;
}

function inventorySymbols(text) {
  var symbols = text.match(/\b(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\b/g) || [];
  var spaced = text.match(/\b[A-Za-z_]\w*(?:\s+\.\s+[a-z_]\w*)+\b/g) || [];
  var tableRows = [...text.matchAll(/\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*) (?:yes|no) (?:yes|no|n\/a)\b/g)]
    .map(function (match) { return match[1]; });
  return symbols.concat(tableRows, spaced).map(function (symbol) {
    return symbol.replace(/\s*\.\s*/g, ".").toLowerCase();
  });
}

function originalAndStem(token) {
  var original = token.clone();
  var stemmed = searchEngine.stemmer(token.clone());
  return original.toString() === stemmed.toString() ? original : [original, stemmed];
}

function codeAwareStopWordFilter(token) {
  var fields = token.metadata.fields || [];
  return fields.includes("symbol") || fields.includes("title") || fields.includes("page")
      || fields.includes("inventory")
    ? token
    : searchEngine.stopWordFilter(token);
}

function searchableDocuments(entries) {
  var sectionTitles = Object.create(null);
  for (var i = 0; i < entries.length; i += 1) {
    var hash = entries[i].location.indexOf("#");
    if (hash < 0) continue;
    var page = entries[i].location.slice(0, hash);
    if (!sectionTitles[page]) sectionTitles[page] = new Set();
    sectionTitles[page].add(compact(entries[i].title));
  }
  return entries.filter(function (doc) {
    return doc.location.includes("#")
      || !sectionTitles[doc.location]
      || !sectionTitles[doc.location].has(compact(doc.title));
  });
}

function buildIndex(searchData, lunrEngine) {
  searchEngine = lunrEngine;
  searchEngine.tokenizer.separator = new RegExp(searchData.config.separator);
  documents = Object.create(null);
  documentSymbols = Object.create(null);
  knownSymbolLeaves = new Set();
  pageTitles = Object.create(null);
  for (var i = 0; i < searchData.docs.length; i += 1) {
    var source = searchData.docs[i];
    if (!source.location.includes("#")) pageTitles[source.location] = source.title;
  }

  var entries = searchableDocuments(searchData.docs);
  var sectionPages = new Set(searchData.docs.filter(function (doc) {
    return doc.location.includes("#");
  }).map(function (doc) {
    return doc.location.split("#")[0];
  }));
  searchEngine.Pipeline.registerFunction(originalAndStem, "advectOriginalAndStem");
  searchEngine.Pipeline.registerFunction(codeAwareStopWordFilter, "advectCodeAwareStopWordFilter");
  index = searchEngine(function () {
    this.ref("location");
    this.field("symbol", { boost: 30 });
    this.field("title", { boost: 15 });
    this.field("page", { boost: 5 });
    this.field("inventory", { boost: 3 });
    this.field("text");
    this.metadataWhitelist = ["position"];
    this.pipeline.remove(searchEngine.stopWordFilter);
    this.pipeline.remove(searchEngine.stemmer);
    this.pipeline.add(codeAwareStopWordFilter);
    this.pipeline.add(originalAndStem);

    for (var j = 0; j < entries.length; j += 1) {
      var doc = entries[j];
      var parts = locationParts(doc.location);
      var indexed = {
        location: doc.location,
        symbol: identifierAliases(parts.anchor),
        title: identifierAliases(doc.title),
        page: identifierAliases((pageTitles[parts.page] || "") + " " + parts.page),
        inventory: inventorySymbols(doc.text).join(" "),
        text: !parts.anchor && sectionPages.has(doc.location) ? "" : doc.text,
      };
      this.add(indexed);
      documents[doc.location] = doc;
      documentSymbols[doc.location] = new Set(inventorySymbols(doc.text));
      for (var symbol of documentSymbols[doc.location]) {
        knownSymbolLeaves.add(symbol.split(".").pop());
      }
    }
  });
}

function queryIdentifiers(rawQuery) {
  return rawQuery.match(/[A-Za-z_][A-Za-z0-9]*(?:[._][A-Za-z0-9_]+)+/g) || [];
}

function isQueryQualifier(term) {
  return QUERY_QUALIFIER_STEMS.has(
    searchEngine.stemmer(new searchEngine.Token(term)).toString(),
  );
}

function queryTerms(rawQuery, keepAll) {
  var seen = Object.create(null);
  var terms = [];
  var tokens = searchEngine.tokenizer(rawQuery);
  var identifierTerms = new Set(
    searchEngine.tokenizer(queryIdentifiers(rawQuery).join(" ")).map(function (token) {
      return token.toString();
    }),
  );
  for (var i = 0; i < tokens.length; i += 1) {
    var term = tokens[i].toString();
    if (!keepAll && tokens.length > 1 && !identifierTerms.has(term) && isQueryQualifier(term)) {
      continue;
    }
    var token = keepAll || tokens.length === 1 || identifierTerms.has(term)
      ? tokens[i]
      : searchEngine.stopWordFilter(tokens[i]);
    if (token === undefined) continue;
    if (!seen[term]) {
      seen[term] = true;
      terms.push(term);
    }
  }
  return terms;
}

function runQuery(terms, prefix, required) {
  var options = {};
  if (required) options.presence = searchEngine.Query.presence.REQUIRED;
  if (prefix) {
    options.wildcard = searchEngine.Query.wildcard.TRAILING;
    options.usePipeline = false;
  }
  return index.query(function (query) {
    query.term(terms, options);
  });
}

function compact(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function resultTitle(doc) {
  var parts = locationParts(doc.location);
  var anchorParts = parts.anchor.split(".");
  var localTitle = anchorParts.length >= 3
      && anchorParts[anchorParts.length - 1] === doc.title
      && /^[A-Z]/.test(anchorParts[anchorParts.length - 2])
    ? anchorParts.slice(-2).join(".") : doc.title;
  var pageTitle = pageTitles[parts.page];
  return pageTitle && pageTitle !== localTitle ? pageTitle + " › " + localTitle : localTitle;
}

function contentWords(value) {
  var tokens = searchEngine.tokenizer(identifierAliases(value));
  return tokens.filter(function (token) {
    return searchEngine.stopWordFilter(token.clone()) !== undefined
      && (tokens.length === 1 || !isQueryQualifier(token.toString()));
  }).map(function (token) { return token.toString(); });
}

function titleWordPrefix(value, rawQuery) {
  var titleWords = contentWords(value);
  var queryWords = contentWords(rawQuery);
  if (queryWords.length === 0 || queryWords.length > titleWords.length) return false;
  for (var start = 0; start + queryWords.length <= titleWords.length; start += 1) {
    if (queryWords.every(function (word, index) {
      var titleWord = titleWords[start + index];
      return titleWord.startsWith(word)
        || searchEngine.stemmer(new searchEngine.Token(titleWord)).toString()
          === searchEngine.stemmer(new searchEngine.Token(word)).toString();
    })) return true;
  }
  return false;
}

function labelPriority(doc, rawQuery) {
  var needle = compact(rawQuery);
  var rawAnchor = locationParts(doc.location).anchor;
  var anchor = compact(rawAnchor);
  var anchorLeaf = compact(rawAnchor.split(".").pop() || "");
  var title = compact(doc.title);
  var displayed = compact(resultTitle(doc));
  var parts = locationParts(doc.location);
  var pageTitle = pageTitles[parts.page] || "";
  if (rawAnchor === rawQuery || rawAnchor.endsWith("." + rawQuery)) return 60;
  if (anchor === needle || anchorLeaf === needle
      || (/[._\s]/.test(rawQuery) && anchor.endsWith(needle))) return 50;
  if (title === needle) return 40;
  if (compact(pageTitle + " " + doc.title) === needle) return 37;
  if (displayed === needle) return 35;
  if (compact(parts.page.replace(/\/$/, "")) === needle) return 33;
  if ((titleWordPrefix(doc.title, rawQuery) || titleWordPrefix(resultTitle(doc), rawQuery))
      && (!/[._]/.test(doc.title) || title.startsWith(needle))) return 20;
  if (title.startsWith(needle)) return 20;
  if (displayed.startsWith(needle)) return 15;
  if (title.includes(needle) || displayed.includes(needle)) return 10;
  return 0;
}

function bodySymbolPriority(doc, rawQuery, allowLoose) {
  var needle = rawQuery.trim();
  var symbols = documentSymbols[doc.location];
  if (!needle || !symbols) return 0;
  var needles = [needle];
  if (/\s/.test(needle)) needles.push(needle.replace(/\s+/g, "."));
  for (var candidate of needles) {
    if (symbols.has(candidate.toLowerCase())) return 55;
  }
  if (needles.length > 1) {
    if (!allowLoose) return 0;
    var words = needle.split(/\s+/);
    var leaf = words[words.length - 1].toLowerCase();
    var context = compact((pageTitles[locationParts(doc.location).page] || "")
      + " " + locationParts(doc.location).page);
    if (words.length > 1 && words.slice(0, -1).every(function (word) {
      return context.includes(compact(word));
    })) {
      for (var symbol of symbols) {
        if (symbol.split(".").pop() === leaf) return 45;
      }
    }
    var queryTokens = searchEngine.tokenizer(needle);
    var hasFiller = queryTokens.some(function (token) {
      return searchEngine.stopWordFilter(token.clone()) === undefined
        && !knownSymbolLeaves.has(token.toString());
    });
    if (hasFiller) return 0;
    var queryParts = queryTokens.map(function (token) {
      return token.toString();
    });
    for (var symbol of symbols) {
      if (queryParts.includes(symbol.split(".").pop().toLowerCase())) return 18;
    }
    return 0;
  }
  if (needle.includes(".")) return 0;
  for (var symbol of symbols) {
    if (symbol.split(".").pop() === needle.toLowerCase()) return 18;
  }
  return 0;
}

function titleMatchLength(doc, rawQuery) {
  var needle = compact(rawQuery);
  var title = compact(doc.title);
  var displayed = compact(resultTitle(doc));
  if (title.includes(needle)) return title.length;
  if (displayed.includes(needle)) return displayed.length;
  return Infinity;
}

function locationDepth(doc) {
  return locationParts(doc.location).page.split("/").filter(Boolean).length;
}

function queryPhrases(rawQuery) {
  var words = searchEngine.tokenizer(rawQuery).map(function (token) { return token.toString(); });
  var phrases = [{ query: rawQuery, words: words.length }];
  var seen = new Set([compact(rawQuery)]);
  for (var identifier of queryIdentifiers(rawQuery)) {
    if (seen.has(compact(identifier))) continue;
    seen.add(compact(identifier));
    phrases.push({
      query: identifier,
      words: searchEngine.tokenizer(identifier).length,
      exactIdentifier: true,
    });
  }
  for (var length = Math.min(3, words.length); length > 0; length -= 1) {
    for (var start = 0; start + length <= words.length; start += 1) {
      var query = words.slice(start, start + length).join(" ");
      if (length === 1 && words.length > 1 && isQueryQualifier(query)) continue;
      if (length === 1 && searchEngine.stopWordFilter(new searchEngine.Token(query)) === undefined
          && !(words.length === 1 && knownSymbolLeaves.has(query))) continue;
      if (seen.has(compact(query))) continue;
      seen.add(compact(query));
      phrases.push({ query: query, words: length });
    }
  }
  return phrases;
}

function bestLabelMatch(doc, phrases) {
  var best = {
    priority: 0,
    query: phrases[0].query,
    words: 0,
    exact: false,
  };
  var queryWords = phrases[0].words;
  for (var phrase of phrases) {
    var priority = labelPriority(doc, phrase.query);
    var exact = compact(doc.title) === compact(phrase.query)
      || compact(resultTitle(doc)) === compact(phrase.query);
    if (phrase.words < queryWords && !phrase.exactIdentifier) {
      priority = Math.min(priority, 15 + 5 * phrase.words);
    }
    if (priority > best.priority
        || (priority === best.priority && exact && !best.exact)
        || (priority === best.priority && exact === best.exact && phrase.words > best.words)) {
      best = { priority: priority, query: phrase.query, words: phrase.words, exact: exact };
    }
  }
  return best;
}

function bestBodyPriority(doc, phrases, allowLoose) {
  var best = 0;
  var queryWords = phrases[0].words;
  for (var phrase of phrases) {
    var priority = bodySymbolPriority(doc, phrase.query, allowLoose);
    if (phrase.words < queryWords && !phrase.exactIdentifier) {
      var codeLike = phrase.words === 1 && knownSymbolLeaves.has(phrase.query)
        && phrase.query.length <= 4;
      priority = codeLike && priority ? 30 : Math.min(priority, 15 + 5 * phrase.words);
    }
    best = Math.max(best, priority);
  }
  return best;
}

function pathContext(doc) {
  var parts = locationParts(doc.location).page.split("/").filter(Boolean);
  if (parts.length > 1 && compact(parts[parts.length - 1]) === compact(doc.title)) parts.pop();
  return parts.map(function (part) {
    if (part === "api") return "API";
    return part.charAt(0).toUpperCase() + part.slice(1).replace(/-/g, " ");
  }).join(" / ") || "Home";
}

function resultCoverage(result, terms) {
  var matches = Object.keys(result.matchData.metadata);
  return terms.filter(function (term) {
    var stem = searchEngine.stemmer(new searchEngine.Token(term)).toString();
    return matches.some(function (match) {
      return match === term || match === stem || match.startsWith(term);
    });
  }).length;
}

function rankedResults(searches, phrases, allowLoose, coverageTerms) {
  var matches = Object.create(null);
  var queryWords = phrases[0].words;
  var queryContentWords = contentWords(phrases[0].query).length;
  for (var i = 0; i < searches.length; i += 1) {
    for (var j = 0; j < searches[i].results.length; j += 1) {
      var result = searches[i].results[j];
      if (!matches[result.ref]) {
        matches[result.ref] = { result: result, tier: searches[i].tier };
      }
    }
  }
  return Object.keys(matches).map(function (ref) {
    var match = matches[ref];
    var doc = documents[match.result.ref];
    var label = bestLabelMatch(doc, phrases);
    var bodyPriority = bestBodyPriority(doc, phrases, allowLoose);
    var priority = Math.max(label.priority, bodyPriority);
    return {
      result: match.result,
      tier: match.tier,
      doc: doc,
      label: label,
      bodyPriority: bodyPriority,
      priority: priority,
      strongPriority: label.priority >= 20
          && (label.words === queryWords
            || contentWords(label.query).length === queryContentWords)
        ? Math.max(priority, 40) : (priority >= 30 ? priority : 0),
      coverage: resultCoverage(match.result, coverageTerms),
    };
  }).filter(function (match) {
    return match.tier > 1 || coverageTerms.length <= 1
      || match.coverage >= Math.min(2, coverageTerms.length) || match.priority >= 20;
  }).sort(function (left, right) {
    return right.strongPriority - left.strongPriority
      || Number(right.tier === 4) - Number(left.tier === 4)
      || Number(right.label.exact) - Number(left.label.exact)
      || (left.bodyPriority >= 30 && left.bodyPriority === right.bodyPriority
          && left.label.priority === right.label.priority
        ? documentSymbols[right.doc.location].size - documentSymbols[left.doc.location].size : 0)
      || right.label.priority - left.label.priority
      || right.priority - left.priority
      || right.label.words - left.label.words
      || (compact(left.doc.title) === compact(right.doc.title)
        ? locationDepth(left.doc) - locationDepth(right.doc) : 0)
      || (compact(left.doc.title) === compact(left.label.query)
        && compact(right.doc.title) === compact(right.label.query)
        ? locationDepth(left.doc) - locationDepth(right.doc) : 0)
      || (left.label.priority === 20 && right.label.priority === 20
        ? titleMatchLength(left.doc, left.label.query) - titleMatchLength(right.doc, right.label.query) : 0)
      || right.tier - left.tier
      || (left.bodyPriority && left.bodyPriority === right.bodyPriority
        ? documentSymbols[right.doc.location].size - documentSymbols[left.doc.location].size : 0)
      || right.result.score - left.result.score;
  });
}

function usefulTextPosition(result) {
  var best;
  for (var match of Object.values(result.matchData.metadata)) {
    var positions = match.text && match.text.position;
    if (!positions) continue;
    if (!best || positions.length < best.length) best = positions;
  }
  return best && best[0][0];
}

function literalTextPosition(text, rawQuery) {
  var haystack = text.toLowerCase();
  var needles = [rawQuery.trim()].concat(queryIdentifiers(rawQuery));
  var tokens = searchEngine.tokenizer(rawQuery);
  for (var i = tokens.length - 1; i >= 0; i -= 1) {
    var term = tokens[i].toString();
    if (knownSymbolLeaves.has(term)) needles.push(term);
  }
  for (var needle of needles) {
    needle = needle.toLowerCase();
    var position = haystack.indexOf(needle);
    while (needle && position >= 0) {
      var before = position > 0 ? haystack.charAt(position - 1) : "";
      var after = haystack.charAt(position + needle.length);
      if (!/[a-z0-9_]/.test(before) && !/[a-z0-9_]/.test(after)) return position;
      position = haystack.indexOf(needle, position + 1);
    }
  }
  return undefined;
}

function excerpt(result, doc, rawQuery) {
  var text = doc.text;
  var position = labelPriority(doc, rawQuery)
    ? 0
    : literalTextPosition(text, rawQuery) ?? usefulTextPosition(result);
  if (position === undefined) return text.slice(0, 220);

  var start = Math.max(0, position - 70);
  var end = Math.min(text.length, position + 170);
  if (start > 0) {
    var previousSpace = text.lastIndexOf(" ", start);
    if (previousSpace >= 0) start = previousSpace + 1;
  }
  if (end < text.length) {
    var nextSpace = text.indexOf(" ", end);
    if (nextSpace >= 0) end = nextSpace;
  }
  var summary = (start > 0 ? "… " : "") + text.slice(start, end).trim()
    + (end < text.length ? " …" : "");
  return summary.replace(/(.{20,240}?) \1(?= |$)/g, "$1");
}

function search(rawQuery) {
  var rawTokens = searchEngine.tokenizer(rawQuery);
  var terms = queryTerms(rawQuery, true);
  if (terms.length === 0) return [];
  var filteredTerms = queryTerms(rawQuery, false);
  var filtered = filteredTerms.length !== terms.length;
  var searches = [
    { results: runQuery(terms, false, true), tier: 4 },
    { results: runQuery(terms, true, true), tier: 3 },
  ];
  if (filteredTerms.length && filtered) {
    searches.push(
      { results: runQuery(filteredTerms, false, true), tier: 3 },
      { results: runQuery(filteredTerms, true, true), tier: 2 },
    );
  }
  if (filteredTerms.length) {
    searches.push(
      { results: runQuery(filteredTerms, false, false), tier: 1 },
      { results: runQuery(filteredTerms, true, false), tier: 0 },
    );
  }
  var rankingQuery = filtered
    ? filteredTerms.join(" ")
    : (rawTokens.length === terms.length || /[._]/.test(rawQuery) ? rawQuery : terms.join(" "));
  var identifiers = queryIdentifiers(rawQuery);
  var phrases = identifiers.length === 1 && rawQuery.trim() === identifiers[0]
    ? [{ query: rankingQuery, words: terms.length }]
    : queryPhrases(rawQuery);
  var results = rankedResults(searches, phrases, !filtered, filteredTerms);
  var selected = results.slice(0, MAX_RESULTS);
  var titleCounts = Object.create(null);
  for (var i = 0; i < selected.length; i += 1) {
    var baseTitle = resultTitle(documents[selected[i].result.ref]);
    titleCounts[baseTitle] = (titleCounts[baseTitle] || 0) + 1;
  }

  var resultDocuments = [];
  for (var j = 0; j < selected.length; j += 1) {
    var result = selected[j].result;
    var doc = documents[result.ref];
    var title = resultTitle(doc);
    resultDocuments.push({
      location: doc.location,
      title: titleCounts[title] > 1 ? pathContext(doc) + " › " + title : title,
      summary: excerpt(result, doc, rawQuery),
    });
  }
  return resultDocuments;
}

async function init() {
  var searchData = await (await fetch("search_index.json")).json();
  buildIndex(searchData, lunr);
  postMessage({ config: searchData.config });
  postMessage({ allowSearch: true });
}

if (typeof importScripts === "function") {
  importScripts("lunr.js");
  onmessage = function (event) {
    if (event.data.init) init();
    else if (event.data.query !== undefined) {
      postMessage({ query: event.data.query, results: search(event.data.query) });
    }
    else console.error("Worker received an unrecognized message");
  };
}

if (typeof module === "object" && module.exports) {
  module.exports = {
    buildIndex: buildIndex,
    search: search,
    searchableDocuments: searchableDocuments,
  };
}
