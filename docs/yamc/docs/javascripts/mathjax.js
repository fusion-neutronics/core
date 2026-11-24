// MathJax configuration for mkdocs-material's arithmatex "generic" mode.
// Arithmatex emits \(...\) and \[...\] wrappers; this tells MathJax to pick
// them up, and to re-typeset after the theme's instant-navigation swaps the
// page body (without this, maths renders on a hard load and not on a nav click).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
