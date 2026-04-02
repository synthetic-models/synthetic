document.addEventListener("DOMContentLoaded", function () {
  function getTheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate"
      ? "dark"
      : "default";
  }

  mermaid.initialize({
    startOnLoad: true,
    theme: getTheme(),
    securityLevel: "loose",
  });

  // Re-render diagrams when theme toggles
  var observer = new MutationObserver(function () {
    var theme = getTheme();
    mermaid.initialize({ startOnLoad: false, theme: theme, securityLevel: "loose" });
    mermaid.run();
  });
  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
});
