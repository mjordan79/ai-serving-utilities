// dead-end background: procedurally generated neural-net graph.
// Builds an SVG feedforward net (input → dense·1 → dense·2 → output)
// with classic circle nodes on straight edges, then lets CSS do the
// effects on a shared 4s cycle (see .net rules in style.css):
// - a forward-pass wave lights each column left→right (each node's
//   --d delay = its layer index, plus small jitter)
// - ~45% of edges carry a travelling dash: pathLength=1 normalizes
//   every edge to unit length, so one short stroke-dasharray dash
//   slides start→end as dashoffset animates 0 → -1; per-edge delays
//   are staggered inside the source column's slot so pulses follow
//   the wave
// prefers-reduced-motion: CSS strips the animations and the graph
// renders static. The panel is aria-hidden (texture, not content).

(function () {
  "use strict";

  const svg = document.getElementById("net-svg");
  if (!svg) return;

  const NS = "http://www.w3.org/2000/svg";
  const W = 440;
  const H = 252;
  const LAYERS = [6, 8, 8, 4];
  const XS = [38, 164, 290, 416];
  const LABELS = ["input", "dense·1", "dense·2", "output"];
  const CYCLE = 4; // seconds — must match the CSS animation duration

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    (parent || svg).appendChild(n);
    return n;
  }

  // Node centers per layer, evenly spaced top→bottom.
  const cols = LAYERS.map((count, l) => {
    const top = 30;
    const bottom = H - 34;
    const step = count > 1 ? (bottom - top) / (count - 1) : 0;
    return Array.from({ length: count }, (_, i) => ({ x: XS[l], y: top + i * step }));
  });

  // Edges (full connectivity between adjacent layers). ~45% of them
  // carry a travelling dash; the rest stay as the static faint mesh.
  const edges = el("g", { class: "edges" });
  for (let l = 0; l < LAYERS.length - 1; l++) {
    for (const a of cols[l]) {
      for (const b of cols[l + 1]) {
        const attrs = {
          d: `M ${a.x} ${a.y} L ${b.x} ${b.y}`,
          pathLength: 1,
          class: "edge",
        };
        if (Math.random() < 0.45) {
          attrs.class += " pulse";
          // Fire inside the source column's slot of the 4s cycle, so
          // the pulses trail the wave as it crosses the layer.
          const t = l + 0.15 + Math.random() * 0.7;
          attrs.style = `--d: ${(-(CYCLE - t)).toFixed(2)}s`;
        }
        el("path", attrs, edges);
      }
    }
  }

  // Nodes: a soft halo behind a dark core with a glowing stroke.
  const nodes = el("g", { class: "nodes" });
  cols.forEach((col, l) => {
    col.forEach((p, i) => {
      const t = l + (i % 3) * 0.05; // small jitter within the layer slot
      const d = `--d: ${(-(CYCLE - t)).toFixed(2)}s`;
      el("circle", { class: "halo", cx: p.x, cy: p.y, r: 12, style: d }, nodes);
      el("circle", { class: "core", cx: p.x, cy: p.y, r: 6.5, style: d }, nodes);
    });
  });

  // Column labels.
  cols.forEach((_, l) => {
    const t = el("text", { class: "net-label", x: XS[l], y: H - 8 });
    t.textContent = LABELS[l];
  });
})();
