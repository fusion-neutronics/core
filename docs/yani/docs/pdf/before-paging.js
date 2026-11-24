// Runs in the PDF render only. mkdocs-exporter calls `MkDocsExporter.render`
// before Paged.js splits each page into A4 sheets.
window.MkDocsExporter = {
  render: async () => {
    // Paged.js treats whitespace-only text nodes as ignorable when it carries
    // a block over to the next sheet, so in a code block split across a page
    // break the newlines and spaces between the highlighted tokens vanish and
    // the continuation prints as one run-on line. In a span they are content.
    for (const pre of document.querySelectorAll("pre")) {
      const walker = document.createTreeWalker(pre, NodeFilter.SHOW_TEXT);
      const blanks = [];
      while (walker.nextNode()) {
        if (!/[^\t\n\r ]/.test(walker.currentNode.textContent)) {
          blanks.push(walker.currentNode);
        }
      }
      for (const node of blanks) {
        const span = document.createElement("span");
        node.replaceWith(span);
        span.appendChild(node);
      }
    }

    // Hold back until MathJax has typeset, so the breaks are placed using the
    // typeset equations rather than the raw TeX.
    while (!(window.MathJax && MathJax.startup && MathJax.startup.promise)) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    await MathJax.startup.promise;
    await MathJax.typesetPromise();
  },
};
