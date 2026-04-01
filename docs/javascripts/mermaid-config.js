document.addEventListener("DOMContentLoaded", function () {
  var config = {
    startOnLoad: true,
    theme:
      document.body.getAttribute("data-md-color-scheme") === "slate"
        ? "dark"
        : "default",
    securityLevel: "loose",
  };
  mermaid.initialize(config);
});
