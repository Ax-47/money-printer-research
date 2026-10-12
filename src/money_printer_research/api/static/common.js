// Shared by every page: DOM helpers, localStorage that tolerates being blocked, the
// Catppuccin flavor button (#theme), which cycles the flavors and remembers the choice,
// and the language switch (#lang), one button per language, also remembered.
const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = Object.assign(document.createElement(tag), attrs);
  node.append(...children);
  return node;
}

// Shown in place of a page's content when its command has not been run yet.
function notRun(what, command) {
  const [before, after] = t("not_run", { what, command: "{command}" }).split("{command}");
  return el("div", { className: "empty" }, before, el("code", {}, `uv run ${command}`), after ?? "");
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

// --- Language ------------------------------------------------------------------------
// Each page adds its own strings with addStrings({ en: {...}, th: {...} }). Elements with
// data-i18n="key" get t(key) as their text (data-i18n-placeholder / -title for those
// attributes); code that builds the page calls t() itself and re-renders on onLang().

const STRINGS = {
  en: { nav_upcoming: "Upcoming", nav_backtest: "Backtest", nav_analysis: "Analysis",
    brand_sub: "Football market research", lang_label: "Language",
    theme_title: "Switch Catppuccin flavor", not_run: "No {what} yet. Run {command} and reload." },
  th: { nav_upcoming: "นัดถัดไป", nav_backtest: "ทดสอบย้อนหลัง", nav_analysis: "บทวิเคราะห์",
    brand_sub: "งานวิจัยตลาดพนันฟุตบอล", lang_label: "ภาษา",
    theme_title: "เปลี่ยนธีมสี", not_run: "ยังไม่มีข้อมูล ({what}) รัน {command} แล้วโหลดหน้าใหม่" },
};
const LANGS = Object.keys(STRINGS);
const LANG_NAMES = { en: "EN", th: "ไทย" };
let lang = LANGS.includes(stored("lang")) ? stored("lang")
  : (navigator.language || "").startsWith("th") ? "th" : "en";
const langListeners = [];

function addStrings(more) {
  for (const l of LANGS) Object.assign(STRINGS[l], more[l]);
  applyLang();
}

// t("key", { n: 3 }) fills {n} in the string; a missing key falls back to English.
function t(key, vars = {}) {
  const text = STRINGS[lang][key] ?? STRINGS.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? "");
}

const locale = () => (lang === "th" ? "th-TH" : "en-GB");

function onLang(listener) { langListeners.push(listener); }

function applyLang() {
  document.documentElement.lang = lang;
  for (const node of document.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of document.querySelectorAll("[data-i18n-placeholder]")) node.placeholder = t(node.dataset.i18nPlaceholder);
  for (const node of document.querySelectorAll("[data-i18n-title]")) node.title = t(node.dataset.i18nTitle);
  const group = $("lang");
  if (group) {
    group.setAttribute("aria-label", t("lang_label"));
    for (const button of group.children) button.setAttribute("aria-pressed", button.value === lang);
  }
}

function setLang(next) {
  lang = next;
  store("lang", next);
  applyLang();
  for (const listener of langListeners) listener();
}

{
  const group = $("lang");
  if (group) {
    group.replaceChildren(...LANGS.map((l) => el("button", { type: "button", value: l, lang: l }, LANG_NAMES[l])));
    group.addEventListener("click", (e) => { if (e.target.value && e.target.value !== lang) setLang(e.target.value); });
  }
  applyLang();
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
