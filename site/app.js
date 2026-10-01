(function () {
  "use strict";

  // ---------- Navigation ----------
  const PAGES = ["welcome", "getting-started", "citation", "faq"];

  function show() {
    const id = location.hash.replace("#", "") || "welcome";
    const page = PAGES.includes(id) ? id : "welcome";
    PAGES.forEach((p) => {
      document.getElementById("page-" + p).hidden = p !== page;
    });
    document.querySelectorAll(".nav a").forEach((a) => {
      a.classList.toggle("active", a.dataset.page === page);
      if (a.dataset.page === page) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", show);

  // ---------- Copy buttons ----------
  function copyText(text, button) {
    const done = () => {
      button.textContent = "Copied";
      button.classList.add("ok");
      setTimeout(() => {
        button.textContent = "Copy";
        button.classList.remove("ok");
      }, 1500);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, () => fallback(text, done));
    } else {
      fallback(text, done);
    }
  }
  function fallback(text, done) {
    const t = document.createElement("textarea");
    t.value = text;
    t.style.position = "fixed";
    t.style.opacity = "0";
    document.body.appendChild(t);
    t.select();
    try { document.execCommand("copy"); done(); } catch (e) { /* ignore */ }
    t.remove();
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest("button.copy");
    if (!b) return;
    copyText(b.parentElement.querySelector("code").textContent, b);
  });

  // ---------- Citation ----------
  function buildCitation(c) {
    const published = c.status === "published";
    const dotted = (ini) => ini.split("").join(". ") + ".";

    const textAuthors = c.authors.map(([last, ini]) => last + " " + ini).join(", ");
    let where;
    if (published) {
      where = c.venue + ". " + c.year;
      if (c.volume) where += ";" + c.volume + (c.issue ? "(" + c.issue + ")" : "");
      if (c.pages) where += ":" + c.pages;
      where += ".";
    } else {
      where = c.venue + " " + c.preprintId + ".";
    }
    const text = textAuthors + ". " + c.title + ". " + where + " doi:" + c.doi;

    const key = c.authors[0][0].toLowerCase().replace(/[^a-z]/g, "") + c.year + "abel";
    const bib = [
      "@article{" + key + ",",
      "  title   = {" + c.title.replace(/^ABEL/, "{ABEL}") + "},",
      "  author  = {" + c.authors.map(([l, i]) => l + ", " + dotted(i)).join(" and ") + "},",
      "  journal = {" + c.venue + "},",
      "  year    = {" + c.year + "},",
    ];
    if (published && c.volume) bib.push("  volume  = {" + c.volume + "},");
    if (published && c.issue) bib.push("  number  = {" + c.issue + "},");
    if (published && c.pages) bib.push("  pages   = {" + c.pages.replace("-", "--") + "},");
    bib.push("  doi     = {" + c.doi + "},");
    bib.push("  url     = {" + c.url + "}");
    bib.push("}");

    const ris = ["TY  - JOUR"];
    c.authors.forEach(([l, i]) => ris.push("AU  - " + l + ", " + dotted(i)));
    ris.push("TI  - " + c.title);
    ris.push("T2  - " + c.venue);
    ris.push("PY  - " + c.year);
    if (published && c.volume) ris.push("VL  - " + c.volume);
    if (published && c.issue) ris.push("IS  - " + c.issue);
    if (published && c.pages) {
      const [sp, ep] = c.pages.split("-");
      ris.push("SP  - " + sp);
      if (ep) ris.push("EP  - " + ep);
    }
    ris.push("DO  - " + c.doi);
    ris.push("UR  - " + c.url);
    ris.push("ER  - ");

    document.getElementById("cite-badge").textContent = published ? "Published" : "Preprint";
    document.getElementById("cite-title").textContent = c.title;
    document.getElementById("cite-authors").textContent = textAuthors;
    document.getElementById("cite-venue").textContent = where.replace(/\.$/, "");
    const link = document.getElementById("cite-link");
    link.href = c.url;
    link.textContent = published ? "Read the paper" : "Read the preprint";
    document.getElementById("cite-text").textContent = text;
    document.getElementById("cite-bibtex").textContent = bib.join("\n");
    document.getElementById("cite-ris").textContent = ris.join("\n");
  }

  // ---------- FAQ ----------
  const list = document.getElementById("faq-list");
  const input = document.getElementById("faq-search");
  const count = document.getElementById("faq-count");
  const empty = document.getElementById("faq-empty");
  const items = [];

  function escapeRe(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function buildFaq() {
    FAQ.forEach((entry) => {
      const d = document.createElement("details");
      d.className = "faq";
      const s = document.createElement("summary");
      s.textContent = entry.q;
      const body = document.createElement("div");
      body.className = "answer";
      body.innerHTML = entry.a;
      d.append(s, body);
      list.appendChild(d);
      items.push({ el: d, summary: s, q: entry.q, hay: (entry.q + " " + body.textContent).toLowerCase() });
    });
  }

  function filterFaq() {
    const raw = input.value.trim();
    const terms = raw.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    items.forEach((it) => {
      const hit = terms.every((t) => it.hay.includes(t));
      it.el.hidden = !hit;
      if (hit) shown++;
      it.summary.replaceChildren();
      if (terms.length) {
        const re = new RegExp("(" + terms.map(escapeRe).join("|") + ")", "i");
        it.q.split(new RegExp("(" + terms.map(escapeRe).join("|") + ")", "gi")).forEach((part) => {
          if (!part) return;
          if (re.test(part)) {
            const m = document.createElement("mark");
            m.textContent = part;
            it.summary.appendChild(m);
          } else {
            it.summary.appendChild(document.createTextNode(part));
          }
        });
      } else {
        it.summary.textContent = it.q;
      }
      it.el.open = terms.length > 0 && hit;
    });
    empty.hidden = shown > 0;
    count.textContent = terms.length
      ? shown + (shown === 1 ? " result" : " results")
      : items.length + " questions";
  }

  input.addEventListener("input", filterFaq);
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== input && !document.getElementById("page-faq").hidden) {
      e.preventDefault();
      input.focus();
    }
    if (e.key === "Escape" && document.activeElement === input) {
      input.value = "";
      filterFaq();
    }
  });

  buildCitation(CITATION);
  buildFaq();
  filterFaq();
  show();
})();
