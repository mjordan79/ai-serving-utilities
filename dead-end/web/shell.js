// dead-end shell: simulated interactive kubectl + helm sessions.
// One generic engine (createShell) drives every .shell panel from a
// script of {cmd, out} steps. Each command is typed char-by-char at
// the prompt with human-ish jitter, output arrives line by line, and
// the oldest lines are dropped so the fixed-height box stays filled.
// The script loops forever with `clear` between passes.
// prefers-reduced-motion: render the final state once, no typing.
// The panels are aria-hidden (texture, not content) — a11y is fine.

(function () {
  "use strict";

  const PS1 = "root@digitalnucleus:~$ ";

  const KUBECTL_SCRIPT = [
    {
      cmd: "kubectl get deploy -n telco",
      out: [
        ["s-out dim", "NAME           READY   UP-TO-DATE   AVAILABLE   AGE"],
        ["s-out", 'line-gateway   <span class="s-err">0/2</span>     <span class="s-err">0</span>            <span class="s-err">0</span>           11d'],
        ["s-out dim", "billing-api    3/3     3            3           11d"],
        ["s-out dim", "sms-relay      2/2     2            2           11d"],
      ],
    },
    {
      cmd: "kubectl get pods -n telco -l app=line-gateway",
      out: [
        ["s-out dim", "NAME                           READY   STATUS             RESTARTS   AGE"],
        ["s-out", 'line-gateway-7f9c6d4b8-x2kqp  <span class="s-err">0/1</span>     CrashLoopBackOff   <span class="s-err">47</span>         96m'],
        ["s-out", 'line-gateway-7f9c6d4b8-zq8rn  <span class="s-err">0/1</span>     CrashLoopBackOff   <span class="s-err">46</span>         96m'],
      ],
    },
    {
      cmd: "kubectl logs line-gateway-7f9c6d4b8-x2kqp -n telco --tail=3",
      out: [
        ["s-out dim", "05:12:41.882  WARN  c.t.g.Bootstrap - dial peer unreachable (E103)"],
        ["s-out dim", "05:12:41.883  WARN  c.t.g.Bootstrap - dial peer unreachable (E103)"],
        ["s-err", "05:12:42.101 ERROR c.t.g.Bootstrap - NO CARRIER: aborting startup"],
      ],
    },
    {
      cmd: "kubectl rollout restart deploy/line-gateway -n telco",
      out: [["s-warn", "deployment.apps/line-gateway restarted"]],
    },
    {
      cmd: "kubectl rollout status deploy/line-gateway -n telco --timeout=30s",
      out: [
        ["s-out", 'Waiting for deployment "line-gateway" rollout to finish: 0 of 2'],
        ["s-err", "error: timed out waiting for the condition"],
      ],
    },
    {
      cmd: "kubectl delete pod line-gateway-7f9c6d4b8-zq8rn -n telco",
      out: [["s-out", 'pod "line-gateway-7f9c6d4b8-zq8rn" deleted']],
    },
    {
      cmd: "kubectl get pods -n telco -l app=line-gateway",
      out: [
        ["s-out dim", "NAME                           READY   STATUS    RESTARTS   AGE"],
        ["s-out", 'line-gateway-7f9c6d4b8-x2kqp  <span class="s-err">0/1</span>     Running   48         97m'],
        ["s-out", 'line-gateway-59b7d9f4c-m4x2w  <span class="s-err">0/1</span>     <span class="s-warn">Pending</span>   0          12s'],
      ],
    },
  ];

  const HELM_SCRIPT = [
    {
      cmd: "helm repo update",
      out: [
        ["s-out dim", "Hang tight while we grab the latest from your chart repositories..."],
        ["s-out dim", '"telco-gw" has been added to your repositories'],
        ["s-out", "Update Complete."],
      ],
    },
    {
      cmd: "helm template line-gateway ./line-gateway-chart -n telco",
      out: [
        ["s-out dim", "---"],
        ["s-out dim", "# Source: line-gateway/templates/deployment.yaml"],
        ["s-out", "apiVersion: apps/v1"],
        ["s-out", "kind: Deployment"],
        ["s-out dim", "... (51 more lines)"],
      ],
    },
    {
      cmd: "helm upgrade -i line-gateway ./line-gateway-chart -n telco --version 1.4.2",
      out: [
        ["s-out", 'Release "line-gateway" has been upgraded. Happy Helming!'],
        ["s-out dim", "NAME: line-gateway"],
        ["s-out dim", "LAST DEPLOYED: Wed Aug 13 04:02:11 2026"],
        ["s-out", "STATUS: deployed"],
        ["s-out dim", "REVISION: 17"],
        ["s-out dim", "TEST SUITE: None"],
      ],
    },
    {
      cmd: "helm get values line-gateway -n telco -o yaml",
      out: [
        ["s-out dim", "replicaCount: 2"],
        ["s-out dim", "env:"],
        ["s-out dim", "  peer:"],
        ["s-out dim", "    addr: 10.4.9.4:9200"],
      ],
    },
  ];

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const jitter = (base, spread) => base + Math.random() * spread;

  // MAX_LINES must match the panel's .shell-body height (see style.css):
  // 11px font × 1.7 line-height = 18.7px/line.
  function createShell(ttyId, MAX_LINES, SCRIPT) {
    const tty = document.getElementById(ttyId);
    if (!tty) return;

    function addLine(node) {
      tty.appendChild(node);
      while (tty.children.length > MAX_LINES) tty.firstChild.remove();
    }

    function outLine(cls, html) {
      const d = document.createElement("div");
      if (cls) d.className = cls;
      d.innerHTML = html; // static, trusted content
      return d;
    }

    function promptLine() {
      const d = document.createElement("div");
      d.className = "s-line";
      const ps = document.createElement("span");
      ps.className = "s-ps1";
      ps.textContent = PS1;
      const arg = document.createElement("span");
      const cur = document.createElement("span");
      cur.className = "cursor";
      cur.textContent = "\u258C";
      d.append(ps, arg, cur);
      addLine(d);
      return { d, arg };
    }

    async function typeText(arg, text) {
      for (const ch of text) {
        arg.textContent += ch;
        await sleep(jitter(24, 50));
      }
    }

    async function runCommand(item) {
      const { arg } = promptLine();
      await sleep(jitter(500, 1100)); // "thinking"
      await typeText(arg, item.cmd);
      await sleep(jitter(200, 500)); // read back, hit enter
      for (const [cls, html] of item.out) {
        addLine(outLine(cls, html));
        await sleep(jitter(30, 90));
      }
    }

    async function loop() {
      for (;;) {
        for (const item of SCRIPT) await runCommand(item);
        const { arg } = promptLine();
        await sleep(jitter(800, 900));
        await typeText(arg, "clear");
        await sleep(400);
        while (tty.firstChild) tty.firstChild.remove();
        await sleep(jitter(1500, 1500));
      }
    }

    // Reduced motion: render the final state of the session once, static.
    function renderStatic() {
      for (const item of SCRIPT) {
        const { arg } = promptLine();
        arg.textContent = item.cmd;
        for (const [cls, html] of item.out) addLine(outLine(cls, html));
      }
      promptLine(); // trailing empty prompt with blinking cursor
    }

    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      renderStatic();
    } else {
      loop();
    }
  }

  // kubectl panel (left): 14 lines ≈ .shell-body min(17.5rem, 46vh).
  // helm panel (top center): 7 lines ≈ .helm-shell .shell-body 10rem.
  createShell("shell-tty", 14, KUBECTL_SCRIPT);
  createShell("helm-tty", 7, HELM_SCRIPT);
})();
