// Shared by every page: DOM helpers, localStorage that tolerates being blocked, and the
// Catppuccin flavor button (#theme), which cycles the flavors and remembers the choice.
const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = Object.assign(document.createElement(tag), attrs);
  node.append(...children);
  return node;
}

// Shown in place of a page's content when its command has not been run yet.
function notRun(what, command) {
  return el("div", { className: "empty" },
    `No ${what} yet. Run `, el("code", {}, `uv run ${command}`), " and reload.");
}

function options(target, values, make) {
  target.append(...[...new Set(values)].sort().map(make));
}

function stored(key) {
  try { return localStorage.getItem(key) ?? ""; } catch { return ""; }
}

function store(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}

const THEMES = { latte: "Latte", frappe: "Frappé", macchiato: "Macchiato", mocha: "Mocha" };

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("theme").textContent = THEMES[theme];
  store("theme", theme);
}

{
  const names = Object.keys(THEMES);
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(names.includes(stored("theme")) ? stored("theme") : dark ? "mocha" : "latte");
  $("theme").addEventListener("click", () => {
    const now = names.indexOf(document.documentElement.dataset.theme);
    setTheme(names[(now + 1) % names.length]);
  });
}
