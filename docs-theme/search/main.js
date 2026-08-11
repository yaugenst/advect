function joinUrl (base, path) {
  return path.startsWith("/") ? path : base.replace(/\/?$/, "/") + path;
}

function escapeHtml (value) {
  return value.replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function formatResult (location, title, summary) {
  return '<article><h3><a href="' + joinUrl(base_url, location) + '">'+ escapeHtml(title) + '</a></h3><p>' + escapeHtml(summary) +'</p></article>';
}

function displayResults (results) {
  var search_results = document.getElementById("mkdocs-search-results");
  search_results.replaceChildren();
  if (results.length > 0){
    for (var i=0; i < results.length; i++){
      var result = results[i];
      var html = formatResult(result.location, result.title, result.summary);
      search_results.insertAdjacentHTML('beforeend', html);
    }
  } else {
    var noResultsText = search_results.getAttribute('data-no-results-text') || "No results found";
    search_results.insertAdjacentHTML('beforeend', '<p>' + noResultsText + '</p>');
  }
}

function doSearch () {
  var query = document.getElementById('mkdocs-search-query').value;
  if (query.length > min_search_length) {
    searchWorker.postMessage({query: query});
  } else {
    // Clear results for short queries
    displayResults([]);
  }
}

function initSearch () {
  var search_input = document.getElementById('mkdocs-search-query');
  search_input.addEventListener("input", doSearch);
  var term = new URLSearchParams(window.location.search).get('q');
  if (term) {
    search_input.value = term;
  }
  if (search_input.value) {
    doSearch();
  }
}

function onWorkerMessage (e) {
  if (e.data.allowSearch) {
    initSearch();
  } else if (e.data.results
      && e.data.query === document.getElementById('mkdocs-search-query').value) {
    displayResults(e.data.results);
  } else if (e.data.config) {
    min_search_length = e.data.config.min_search_length-1;
  }
}

var searchWorker = new Worker(joinUrl(base_url, "search/worker.js"));
searchWorker.postMessage({init: true});
searchWorker.onmessage = onWorkerMessage;
