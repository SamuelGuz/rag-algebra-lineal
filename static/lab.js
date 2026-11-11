// =====================================================================
//  Laboratorio vectorial: visualiza el álgebra lineal de cada búsqueda
// =====================================================================
const Lab = (() => {
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (x, n = 4) => Number(x).toLocaleString("es-CO", { minimumFractionDigits: n, maximumFractionDigits: n });
  const int = (x) => Number(x).toLocaleString("es-CO");
  const money = (n) => "$" + int(n);
  const deg = (rad) => (rad * 180) / Math.PI;

  const CAT = { "sopas": "#ff9f43", "platos fuertes": "#ff6b6b", "entradas": "#feca57", "bebidas": "#48dbfb", "postres": "#ff9ff3" };
  const catColor = (c) => CAT[c] || "#aaa";

  function svg(tag, attrs = {}, parent) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (parent) parent.appendChild(el);
    return el;
  }

  // Color divergente: azul (negativo) → fondo → naranja (positivo)
  function divColor(v, m) {
    const t = Math.max(-1, Math.min(1, v / m));
    const base = [20, 26, 46], pos = [255, 159, 67], neg = [77, 141, 255];
    const tgt = t >= 0 ? pos : neg, a = Math.pow(Math.abs(t), 0.75);
    const c = base.map((b, i) => Math.round(b + (tgt[i] - b) * a));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  // ---------------- motor de animación (cancelable) ----------------
  let speed = 1, token = 0;
  class Cancel extends Error {}
  const newRun = () => ++token;
  const sleep = (ms, t) => new Promise((res, rej) => setTimeout(() => (t === token ? res() : rej(new Cancel())), ms / speed));
  const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
  function tween(ms, fn, t) {
    return new Promise((res, rej) => {
      const start = performance.now(), d = Math.max(1, ms / speed);
      function frame(now) {
        if (t !== token) return rej(new Cancel());
        const p = Math.min(1, (now - start) / d);
        fn(ease(p));
        p < 1 ? requestAnimationFrame(frame) : res();
      }
      requestAnimationFrame(frame);
    });
  }
  const swallow = (e) => { if (!(e instanceof Cancel)) console.error(e); };

  // ---------------- estado ----------------
  const state = { traces: [], idx: 0, tab: "vector", pick: 0, matrix: null, rowIndex: {}, pca: null, graph: null };
  const cur = () => state.traces[state.idx];

  const TABS = [
    { id: "vector", label: "La frase en números" },
    { id: "dot", label: "Producto punto" },
    { id: "matrix", label: "Matriz X·q" },
    { id: "hnsw", label: "HNSW" },
    { id: "pca", label: "Mapa PCA" },
    { id: "complex", label: "Complejos" },
  ];
  const PIPE = [
    ["Texto", "vector"], ["f(texto) ∈ ℝ⁷⁶⁸", "vector"], ["q / ‖q‖", "vector"],
    ["s = X·q", "matrix"], ["HNSW", "hnsw"], ["top-k", "matrix"], ["Gemini", null],
  ];

  async function loadData() {
    const [pca, matrix, graph] = await Promise.all(
      ["/api/menu/pca", "/api/menu/matrix", "/api/hnsw/graph"].map((u) => fetch(u).then((r) => r.json())));
    state.pca = pca; state.graph = graph; state.matrix = matrix;
    matrix.filas.forEach((f, i) => (state.rowIndex[f.id] = i));
  }
  const vecOf = (id) => state.matrix.X[state.rowIndex[id]];
  const rowOf = (id) => state.matrix.filas[state.rowIndex[id]];
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

  // ---------------- interfaz general ----------------
  function renderPipeline() {
    const el = $("pipeline");
    el.innerHTML = PIPE.map(([label], i) =>
      (i ? '<span class="pipe-arrow">➜</span>' : "") + `<span class="pipe" data-i="${i}">${label}</span>`).join("");
    el.querySelectorAll(".pipe").forEach((p) => p.addEventListener("click", () => {
      const tab = PIPE[+p.dataset.i][1];
      if (tab && cur()) selectTab(tab);
    }));
  }
  async function lightPipeline() {
    const pipes = [...document.querySelectorAll(".pipe")], arrows = [...document.querySelectorAll(".pipe-arrow")];
    pipes.forEach((p) => p.classList.remove("on")); arrows.forEach((a) => a.classList.remove("on"));
    for (let i = 0; i < pipes.length; i++) {
      await new Promise((r) => setTimeout(r, 160));
      if (i) arrows[i - 1].classList.add("on");
      pipes[i].classList.add("on");
    }
  }

  function renderTabs() {
    $("tabs").innerHTML = TABS.map((t, i) =>
      `<button class="tab ${t.id === state.tab ? "active" : ""}" data-tab="${t.id}"><i>${i + 1}</i>${t.label}</button>`).join("");
    $("tabs").querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  }
  function renderTraceTabs() {
    const el = $("trace-tabs");
    el.innerHTML = state.traces.length > 1 ? state.traces.map((t, i) =>
      `<button class="${i === state.idx ? "active" : ""}" data-i="${i}">🔎 ${esc(t.consulta)}${t.directa ? " · directa" : ""}</button>`).join("") : "";
    el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.idx = +b.dataset.i; state.pick = 0; renderTraceTabs(); play(); }));
  }
  function selectTab(id) { state.tab = id; renderTabs(); play(); }

  function play() {
    const t = cur();
    const stage = $("stage");
    stage.style.animation = "none"; void stage.offsetWidth; stage.style.animation = "";
    const run = newRun();
    if (!t) return renderEmpty(run);
    SECTIONS[state.tab](t, run).catch(swallow);
  }

  function head(title, sub) {
    return `<div class="sec-head"><div><h2>${title}</h2><div class="sub">${sub}</div></div>
      <button class="btn ghost" data-replay>↻ Repetir animación</button></div>`;
  }
  function bindReplay() { $("stage").querySelector("[data-replay]")?.addEventListener("click", play); }
  const theory = (html) => `<details class="theory"><summary>📐 La matemática</summary>${html}</details>`;

  function renderEmpty(run) {
    $("stage").innerHTML = `<div class="hero-map">
      <div class="hero-text">
        <h2>Así ve la IA el menú: cada plato es un punto</h2>
        <p>Los platos que significan cosas parecidas quedan cerca. Arrastra para girar, usa la rueda para acercar
           y toca un plato para ver cuáles se le parecen. Escribe en el chat o prueba una frase:</p>
        <div class="chips">${["algo calientito con pollo", "un postre con arequipe", "algo del Caribe con pescado", "sushi"]
          .map((q) => `<button class="chip">${q}</button>`).join("")}</div>
      </div>
      <div class="map-wrap">${mapCanvasCard("")}${mapSideCard()}</div>
    </div>`;
    $("stage").querySelectorAll(".hero-text .chip").forEach((c) => c.addEventListener("click", () => explore(c.textContent)));
    startMap(null, run);
  }

  // pick: plato elegido para comparar (se comparte entre producto punto y complejos)
  function pickChips(t) {
    return `<div class="chips" style="margin-bottom:12px"><span class="muted" style="align-self:center">Comparar q con:</span>
      ${t.resultados.map((r, i) => `<button class="chip ${i === state.pick ? "active" : ""}" data-pick="${i}">${i + 1}. ${esc(r.nombre)}</button>`).join("")}</div>`;
  }
  function bindPick() {
    $("stage").querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => { state.pick = +b.dataset.pick; play(); }));
  }

  // =================================================================
  //  1. LA FRASE EN NÚMEROS: la frase entra al espacio 3D y se busca
  // =================================================================
  async function secVector(t, run) {
    const q = t.vector, n = q.length;
    $("stage").innerHTML = head("1 · La frase se convierte en un vector dentro de la base de datos",
      `Una red neuronal (E5) convierte la frase en un vector q ∈ ℝ<sup>${n}</sup>: 768 números que ubican su <b>significado</b>, como el GPS ubica un lugar con 2.
       Mira cómo el vector entra al espacio de los platos y se compara con cada uno.`) + `
      <div class="flow">
        <div class="text-chip">“${esc(t.consulta)}”</div>
        <div class="fx"><b>f( · )</b>red neuronal</div>
        <div class="vec-chip">q = (q₁, q₂, …, q₇₆₈) ∈ ℝ<sup>${n}</sup></div>
      </div>
      <div class="grid2 g3d">
        ${mapCanvasCard("La consulta dentro de la base de datos (vista 3D)")}
        <div class="col-stack">
          ${mapSideCard()}
          <div class="card"><h3>Normalización</h3>
            <p class="plain small">Todas las flechas se ajustan al mismo largo (1) para que solo importe <b>hacia dónde apuntan</b>.</p>
            <div class="meter"><span>‖q‖</span><div class="bar"><div id="m-raw"></div></div><b id="m-raw-v">0</b></div>
            <div class="eq">q̂ = q / ‖q‖ &nbsp;⇒&nbsp; cada componente ÷ ${fmt(t.norma_cruda, 3)}</div>
            <div class="meter"><span>‖q̂‖</span><div class="bar"><div id="m-unit"></div></div><b id="m-unit-v">–</b></div>
            <h3 style="margin-top:14px">Pitágoras en 768 dimensiones</h3>
            <div class="eq small">‖q̂‖² = q̂₁² + q̂₂² + ⋯ + q̂₇₆₈²</div>
            <div class="meter"><span>Σq̂ᵢ²</span><div class="bar"><div id="m-sum"></div></div><b id="m-sum-v">0</b></div>
            <p class="plain small">Componentes sumadas: <b id="m-count">0</b> / ${n}</p>
          </div>
        </div>
      </div>
      ${theory(`<p>Un <b>embedding</b> es una función f: texto → ℝ⁷⁶⁸. Frases con significado parecido quedan en <b>direcciones</b> parecidas.</p>
        <p>Norma: ‖q‖ = √(q·q) = √(Σ qᵢ²). Al dividir entre ella se obtiene el vector unitario q̂ (sobre la esfera de radio 1).</p>
        <p>El dibujo 3D proyecta los vectores con PCA (3 componentes principales). La comparación real se hace en ℝ⁷⁶⁸:
        sᵢ = q̂·vᵢ para cada plato, y se toman los 5 mayores (k-NN).</p>`)}`;
    bindReplay();
    startMap(t, run, true);

    const scale = Math.ceil(t.norma_cruda);
    $("m-raw").style.width = (t.norma_cruda / scale) * 100 + "%";
    $("m-raw-v").textContent = fmt(t.norma_cruda, 4);
    let sum = 0, i = 0;
    await tween(1800, (p) => {
      const upto = Math.floor(p * n);
      for (; i < upto; i++) sum += q[i] * q[i];
      $("m-sum").style.width = Math.min(100, sum * 100) + "%";
      $("m-sum-v").textContent = fmt(sum, 4);
      $("m-count").textContent = i;
    }, run);
    await tween(1100, (p) => {
      const val = t.norma_cruda + (1 - t.norma_cruda) * p;
      $("m-unit").style.width = (val / scale) * 100 + "%";
      $("m-unit-v").textContent = fmt(val, 4);
    }, run);
    $("m-unit-v").textContent = fmt(t.norma, 6);
  }

  // =================================================================
  //  2. PRODUCTO PUNTO Y ÁNGULO
  // =================================================================
  async function secDot(t, run) {
    const r = t.resultados[state.pick] || t.resultados[0];
    const q = t.vector, v = vecOf(r.id), n = q.length, SHOW = 10;
    const d = dot(q, v);
    const cos = Math.max(-1, Math.min(1, d));
    const theta = Math.acos(cos);
    const euclid = Math.sqrt(q.reduce((s, x, i) => s + (x - v[i]) ** 2, 0));
    $("stage").innerHTML = head("2 · Producto punto: ¿qué tan parecidos son?",
      "Se multiplican las componentes una a una y se suman. Como ‖q‖ = ‖v‖ = 1, el resultado es directamente cos θ: cerca de 1 = muy parecidos, cerca de 0 = sin relación.") +
      pickChips(t) + `
      <div class="grid2">
        <div class="card"><h3>q · v = Σ qᵢ vᵢ, término a término</h3>
          <table class="terms"><thead><tr><th>i</th><th>qᵢ</th><th>vᵢ (${esc(r.nombre)})</th><th>qᵢ · vᵢ</th><th>suma parcial</th></tr></thead>
          <tbody>${Array.from({ length: SHOW }, (_, i) =>
            `<tr id="tr${i}"><td>${i + 1}</td><td>${fmt(q[i], 4)}</td><td>${fmt(v[i], 4)}</td><td class="p">${fmt(q[i] * v[i], 5)}</td><td id="ps${i}"></td></tr>`).join("")}
            <tr class="more" id="trmore"><td colspan="5">⋮ ${n - SHOW} términos más ⋮</td></tr></tbody></table>
          <div class="running"><span class="muted">q · v =</span><b id="run">0</b></div>
        </div>
        <div class="card"><h3>El ángulo θ entre q y v</h3><svg id="angle" class="angle" viewBox="0 0 380 280"></svg></div>
      </div>
      <div class="metrics">
        <div class="metric"><span>Producto punto</span><b class="g">${fmt(d, 4)}</b><small>q · v = ‖q‖‖v‖ cos θ = cos θ</small></div>
        <div class="metric"><span>Ángulo</span><b>${fmt(deg(theta), 2)}°</b><small>θ = arccos(q · v)</small></div>
        <div class="metric"><span>Distancia coseno (ChromaDB)</span><b>${fmt(1 - d, 4)}</b>
          <small class="ok">Chroma devolvió ${fmt(r.distancia_chroma, 4)} ✓</small></div>
        <div class="metric"><span>Distancia euclidiana</span><b>${fmt(euclid, 4)}</b>
          <small>√(2 − 2cos θ) = ${fmt(Math.sqrt(Math.max(0, 2 - 2 * d)), 4)} ✓</small></div>
      </div>
      ${theory(`<p><b>Definición:</b> q·v = Σ qᵢvᵢ. <b>Interpretación geométrica:</b> q·v = ‖q‖‖v‖ cos θ. Si ambos son unitarios, q·v = cos θ.</p>
        <p><b>Proyección:</b> la sombra de q sobre v es proj<sub>v</sub>q = (q·v / v·v) v = (q·v) v. Su longitud es cos θ: mientras más larga, más parecidos.</p>
        <p><b>Ley de cosenos:</b> ‖q − v‖² = ‖q‖² + ‖v‖² − 2 q·v = 2 − 2cos θ. Por eso, con vectores normalizados, ordenar por
        similitud coseno o por distancia euclidiana da <i>exactamente el mismo ranking</i>.</p>`)}`;
    bindReplay(); bindPick();

    const el = $("angle"), O = [40, 240], L = 290;
    const P = (len, ang) => [O[0] + len * Math.cos(ang), O[1] - len * Math.sin(ang)];
    svg("circle", { cx: O[0], cy: O[1], r: L, fill: "none", stroke: "rgba(255,255,255,.06)", "stroke-dasharray": "3 5" }, el);
    svg("line", { x1: O[0] - 20, y1: O[1], x2: O[0] + L + 20, y2: O[1], stroke: "rgba(255,255,255,.12)" }, el);
    const arc = svg("path", { fill: "rgba(139,123,255,.18)", stroke: "#8b7bff", "stroke-width": 1.5 }, el);
    const proj = svg("line", { stroke: "#3ddc97", "stroke-width": 5, "stroke-linecap": "round", opacity: 0 }, el);
    const perp = svg("line", { stroke: "rgba(255,255,255,.5)", "stroke-dasharray": "4 4", opacity: 0 }, el);
    const qa = svg("line", { x1: O[0], y1: O[1], x2: O[0], y2: O[1], stroke: "#8b7bff", "stroke-width": 3, "stroke-linecap": "round" }, el);
    const va = svg("line", { x1: O[0], y1: O[1], x2: O[0], y2: O[1], stroke: "#3dd6d0", "stroke-width": 3, "stroke-linecap": "round" }, el);
    const ql = svg("text", { fill: "#8b7bff", "font-size": 15, "font-weight": 700, opacity: 0 }, el); ql.textContent = "q̂";
    const vl = svg("text", { fill: "#3dd6d0", "font-size": 15, "font-weight": 700, opacity: 0 }, el); vl.textContent = "v̂";
    const tl = svg("text", { fill: "#cfc8ff", "font-size": 13, opacity: 0 }, el);
    const pl = svg("text", { fill: "#3ddc97", "font-size": 12, opacity: 0 }, el);

    let partial = 0;
    const qDraw = tween(500, (p) => { const [x, y] = P(L * p, 0); qa.setAttribute("x2", x); qa.setAttribute("y2", y); }, run);
    for (let i = 0; i < SHOW; i++) {
      const row = $("tr" + i);
      row.classList.add("show", "now");
      partial += q[i] * v[i];
      $("ps" + i).textContent = fmt(partial, 5);
      $("run").textContent = fmt(partial, 5);
      await sleep(260, run);
      row.classList.remove("now");
    }
    await qDraw;
    ql.setAttribute("x", O[0] + L + 6); ql.setAttribute("y", O[1] + 5); ql.setAttribute("opacity", 1);
    $("trmore").classList.add("show");
    const start = partial;
    await tween(1300, (p) => {
      $("run").textContent = fmt(start + (d - start) * p, 5);
      const ang = theta * p, [x, y] = P(L, ang);
      va.setAttribute("x2", x); va.setAttribute("y2", y);
      const [ax, ay] = P(70, ang);
      arc.setAttribute("d", `M${O[0] + 70},${O[1]} A70,70 0 0 0 ${ax},${ay} L${O[0]},${O[1]} Z`);
    }, run);
    $("run").textContent = fmt(d, 5);
    const [vx, vy] = P(L + 16, theta);
    vl.setAttribute("x", vx); vl.setAttribute("y", vy); vl.setAttribute("opacity", 1);
    const [tx, ty] = P(88, theta / 2);
    tl.setAttribute("x", tx); tl.setAttribute("y", ty + 4); tl.textContent = `θ = ${fmt(deg(theta), 1)}°`; tl.setAttribute("opacity", 1);
    await sleep(300, run);
    // proyección de q sobre v
    const [px, py] = P(L * cos, theta);
    perp.setAttribute("x1", O[0] + L); perp.setAttribute("y1", O[1]);
    await tween(600, (p) => {
      perp.setAttribute("opacity", 1);
      perp.setAttribute("x2", O[0] + L + (px - O[0] - L) * p); perp.setAttribute("y2", O[1] + (py - O[1]) * p);
    }, run);
    proj.setAttribute("x1", O[0]); proj.setAttribute("y1", O[1]); proj.setAttribute("opacity", 0.85);
    await tween(700, (p) => { proj.setAttribute("x2", O[0] + (px - O[0]) * p); proj.setAttribute("y2", O[1] + (py - O[1]) * p); }, run);
    pl.setAttribute("x", (O[0] + px) / 2 + 14 * Math.sin(theta)); pl.setAttribute("y", (O[1] + py) / 2 + 22 * Math.cos(theta));
    pl.textContent = `proyección = cos θ = ${fmt(cos, 3)}`; pl.setAttribute("opacity", 1);
    for (const m of document.querySelectorAll(".metric")) { m.classList.add("show"); await sleep(180, run); }
  }

  // =================================================================
  //  3. LA BASE DE DATOS ES UNA MATRIZ: s = X·q
  // =================================================================
  async function secMatrix(t, run) {
    const { filas, X } = state.matrix, N = filas.length, D = state.matrix.dimensiones, ROW = 14;
    const scores = filas.map((f) => t.scores[f.id]);
    const allowed = new Set(t.candidatos);
    const smin = Math.min(...scores), smax = Math.max(...scores);
    const topIds = t.resultados.map((r) => r.id);
    $("stage").innerHTML = head("3 · Búsqueda exacta: la base de datos es una matriz",
      `Los ${N} platos forman una matriz X de ${N}×${D} (una fila por plato). Buscar es UNA multiplicación matriz·vector: s = X·q.
       Es repetir el producto punto del paso 2 con cada fila.`) + `
      <div class="counters">
        <div class="counter">Productos punto <b id="c-dots">0</b> / ${N}</div>
        <div class="counter">Multiplicaciones <b id="c-mul">0</b></div>
        <div class="counter">Sumas <b id="c-add">0</b></div>
        ${t.filtros ? `<div class="counter">Filtro de metadata <b class="mono" style="font-size:12px">${esc(JSON.stringify(t.filtros))}</b></div>` : ""}
      </div>
      <div class="mx">
        <div class="mx-label">qᵀ &nbsp;(1 × ${D})</div><div></div>
        <canvas id="mx-q" width="${D}" height="1" style="height:16px"></canvas><div></div>
        <div class="mx-label">X &nbsp;(${N} × ${D}) — fila i = vector del plato i</div>
        <div class="mx-label">s = X·q &nbsp;(${N} × 1) <span class="muted" style="font-size:11px">· barra entre mín. y máx.</span></div>
        <canvas id="mx-X" width="${D}" height="${N * ROW}" style="height:${N * ROW}px"></canvas>
        <div id="slist" class="slist" style="height:${N * ROW}px"></div>
      </div>
      <p class="muted" style="font-size:12.5px;margin:8px 0 0">¿Ves las rayas verticales? Son dimensiones en las que <b>todos</b> los platos tienen casi el mismo valor.
        Esas componentes comunes hacen que todos los vectores apunten hacia una zona parecida (un "cono"), y por eso todos los cosenos quedan entre ~0,75 y ~0,90.</p>
      <div id="mx-result" class="result-line"></div>
      ${theory(`<p>Si apilamos los vectores de los platos como filas, obtenemos X ∈ ℝ<sup>${N}×${D}</sup>. El producto matriz-vector
        s = Xq calcula de una vez <b>todos</b> los productos punto: s<sub>i</sub> = x<sub>i</sub> · q = cos θ<sub>i</sub>.</p>
        <p>Costo: ${N} filas × ${D} columnas = <b>${int(N * D)} multiplicaciones</b> y ${int(N * (D - 1))} sumas. Con un millón de platos serían
        768 millones de multiplicaciones por consulta: por eso las bases vectoriales usan índices aproximados como HNSW (pestaña 4).</p>
        <p>Después se ordena s de mayor a menor y se toman los k primeros: eso es <b>k-NN</b> (k vecinos más cercanos).</p>`)}`;
    bindReplay();

    const qc = $("mx-q").getContext("2d"), q = t.vector, mq = Math.max(...q.map(Math.abs));
    q.forEach((x, j) => { qc.fillStyle = divColor(x, mq); qc.fillRect(j, 0, 1, 1); });
    const base = document.createElement("canvas"); base.width = D; base.height = N * ROW;
    const bc = base.getContext("2d");
    X.forEach((row, i) => row.forEach((x, j) => { bc.fillStyle = divColor(x, 0.12); bc.fillRect(j, i * ROW, 1, ROW - 2); }));
    const xc = $("mx-X").getContext("2d");
    const drawX = (hi) => {
      xc.drawImage(base, 0, 0);
      filas.forEach((f, i) => { if (!allowed.has(f.id)) { xc.fillStyle = "rgba(11,14,26,.7)"; xc.fillRect(0, i * ROW, D, ROW); } });
      if (hi != null) {
        xc.fillStyle = "rgba(255,209,102,.22)"; xc.fillRect(0, hi * ROW, D, ROW - 2);
        xc.strokeStyle = "#ffd166"; xc.lineWidth = 2; xc.strokeRect(1, hi * ROW, D - 2, ROW - 2);
      }
    };
    drawX(null);

    $("slist").innerHTML = filas.map((f, i) =>
      `<div class="srow ${allowed.has(f.id) ? "" : "out"}" id="s-${f.id}" style="top:${i * ROW}px">
        <span class="rk"></span><span class="nm" title="${esc(f.nombre)}">${esc(f.nombre)}</span>
        <div class="sb"><div></div></div><span class="sv">·</span></div>`).join("");

    for (let i = 0; i < N; i++) {
      drawX(i);
      const row = $("s-" + filas[i].id);
      row.classList.add("done");
      row.querySelector(".sv").textContent = fmt(scores[i], 3);
      row.querySelector(".sb div").style.width = 8 + ((scores[i] - smin) / (smax - smin || 1)) * 92 + "%";
      $("c-dots").textContent = i + 1;
      $("c-mul").textContent = int((i + 1) * D);
      $("c-add").textContent = int((i + 1) * (D - 1));
      await sleep(70, run);
    }
    drawX(null);
    await sleep(400, run);
    const order = filas.map((f, i) => i).sort((a, b) =>
      (allowed.has(filas[b].id) - allowed.has(filas[a].id)) || scores[b] - scores[a]);
    order.forEach((i, rank) => {
      const row = $("s-" + filas[i].id);
      row.style.top = rank * ROW + "px";
      row.querySelector(".rk").textContent = allowed.has(filas[i].id) ? rank + 1 : "";
    });
    await sleep(1000, run);
    topIds.forEach((id) => $("s-" + id)?.classList.add("top"));
    const ours = order.filter((i) => allowed.has(filas[i].id)).slice(0, topIds.length).map((i) => filas[i].id);
    const same = ours.every((id, k) => id === topIds[k]);
    const res = $("mx-result");
    res.innerHTML = `<b>Top-${topIds.length}</b> calculado aquí con s = X·q: ${ours.map((id) => esc(rowOf(id).nombre)).join(", ")}<br>
      ${same ? "✓ Idéntico al resultado que devolvió ChromaDB." : "≈ ChromaDB devolvió: " + topIds.map((id) => esc(rowOf(id).nombre)).join(", ")}`;
    res.classList.add("show");
  }

  // =================================================================
  //  4. HNSW: navegar un grafo por capas
  // =================================================================
  async function secHnsw(t, run) {
    const G = state.graph, H = t.hnsw, layers = G.capas.length;
    const pts = {}; state.pca.puntos.forEach((p) => (pts[p.id] = p));
    const xs = state.pca.puntos.map((p) => p.xy[0]).concat(t.consulta_xy[0]);
    const ys = state.pca.puntos.map((p) => p.xy[1]).concat(t.consulta_xy[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const W = 760, Wp = 540, skew = 120, Dp = 92, gap = 30, left = 70, top0 = 14;
    const Hs = top0 + layers * (Dp + gap);
    const uv = (xy) => [(xy[0] - x0) / (x1 - x0 || 1), 1 - (xy[1] - y0) / (y1 - y0 || 1)];
    const planeTop = (L) => top0 + (layers - 1 - L) * (Dp + gap);
    const pos = (xy, L) => { const [u, v] = uv(xy); return [left + skew * (1 - v) + u * Wp, planeTop(L) + v * Dp]; };
    const posId = (id, L) => pos(pts[id].xy, L);
    const exact = Object.entries(t.scores).sort((a, b) => b[1] - a[1]).slice(0, H.resultado.length).map(([id]) => id);

    $("stage").innerHTML = head("4 · Simulación HNSW: así busca ChromaDB sin revisar todo",
      "Un grafo por capas: arriba pocos nodos y saltos largos, abajo todos los nodos. Se baja \"acercándose\" a q en cada capa, " +
      "como buscar una dirección: primero el país, luego la ciudad, luego la calle.") + `
      <div class="counters">
        <div class="counter">Distancias calculadas (HNSW) <b id="h-calc">0</b></div>
        <div class="counter">Búsqueda exacta <b>${H.total_vectores}</b></div>
        <div class="counter">Capa actual <b id="h-layer">–</b></div>
      </div>
      <div class="hn">
        <div class="card" style="padding:8px"><svg id="hn" class="hnsw" viewBox="0 0 ${W} ${Hs}"></svg></div>
        <div>
          <div class="log" id="h-log"></div>
          <div class="compare" id="h-cmp"></div>
        </div>
      </div>
      ${theory(`<p>Cada plato es un <b>nodo</b>; cada nodo se conecta con sus M = ${G.M} vecinos más cercanos (por distancia coseno 1 − q·v).
        Cada nodo recibe un nivel aleatorio L = ⌊−ln(U)·m<sub>L</sub>⌋, así que la cantidad de nodos decrece exponencialmente con la altura.</p>
        <p><b>Búsqueda voraz:</b> en cada capa, desde el nodo actual se calcula q·v con sus vecinos y se salta al más parecido;
        cuando ningún vecino mejora, se baja una capa. En la capa 0 se mantiene una lista de los ef mejores candidatos.</p>
        <p>Costo aproximado O(log N) en vez de O(N). Con 41 platos el ahorro es pequeño; con 1.000.000 de vectores HNSW calcula unos pocos miles de distancias.
        El precio: el resultado es <b>aproximado</b> (a veces se escapa un vecino).</p>
        <p class="muted">Simulación educativa construida con los mismos 41 vectores (M=${G.M}, ef=5). ChromaDB usa la misma idea con su propio grafo interno.</p>`)}`;
    bindReplay();

    const el = $("hn"), nodeEls = {};
    for (let L = layers - 1; L >= 0; L--) {
      const T = planeTop(L);
      svg("polygon", { class: "plane", points: `${left + skew},${T} ${left + skew + Wp},${T} ${left + Wp},${T + Dp} ${left},${T + Dp}` }, el);
      svg("text", { class: "lbl strong", x: 4, y: T + 14 }, el).textContent = `Capa ${L}`;
      svg("text", { class: "lbl", x: 4, y: T + 29 }, el).textContent = `${G.capas[L].nodos.length} nodos`;
      G.capas[L].aristas.forEach(([a, b]) => {
        const [ax, ay] = posId(a, L), [bx, by] = posId(b, L);
        svg("line", { class: "edge", x1: ax, y1: ay, x2: bx, y2: by }, el);
      });
    }
    Object.entries(G.niveles).forEach(([id, lvl]) => {
      for (let L = lvl; L > 0; L--) {
        const [ax, ay] = posId(id, L), [bx, by] = posId(id, L - 1);
        svg("line", { class: "vlink", x1: ax, y1: ay, x2: bx, y2: by }, el);
      }
    });
    const ghosts = [];
    for (let L = layers - 1; L >= 0; L--) {
      const [qx, qy] = pos(t.consulta_xy, L), s = 6;
      ghosts[L] = svg("polygon", { class: "ghost", points: `${qx},${qy - s} ${qx + s},${qy} ${qx},${qy + s} ${qx - s},${qy}` }, el);
      G.capas[L].nodos.forEach((id) => {
        const [x, y] = posId(id, L);
        const c = svg("circle", { class: "node", cx: x, cy: y, r: L ? 5.5 : 4, fill: catColor(pts[id].categoria) }, el);
        svg("title", {}, c).textContent = `${pts[id].nombre} (capa ${L})`;
        nodeEls[id + "@" + L] = c;
      });
    }
    const fx = svg("g", {}, el);
    const tokenEl = svg("circle", { class: "token", r: 7, cx: -20, cy: -20 }, el);

    const log = $("h-log"), say = (html, cls = "") => { const d = document.createElement("div"); d.className = cls; d.innerHTML = html; log.appendChild(d); log.scrollTop = log.scrollHeight; };
    const name = (id) => esc(pts[id].nombre);
    let calc = 0;
    const bump = () => ($("h-calc").textContent = ++calc);
    const moveToken = (from, to, ms = 450) =>
      tween(ms, (p) => { tokenEl.setAttribute("cx", from[0] + (to[0] - from[0]) * p); tokenEl.setAttribute("cy", from[1] + (to[1] - from[1]) * p); }, run);

    say(`Entrada fija del índice: <b>${name(H.entrada)}</b>`);
    let at = posId(H.entrada, layers - 1), prevId = H.entrada;
    tokenEl.setAttribute("cx", at[0]); tokenEl.setAttribute("cy", at[1]);
    for (const capa of H.capas) {
      const L = capa.capa;
      $("h-layer").textContent = L;
      ghosts.forEach((g) => g.classList.remove("on")); ghosts[L].classList.add("on");
      say(`Capa ${L}${L === 0 ? " (todos los nodos, lista de ef = 5 candidatos)" : " (búsqueda voraz)"}`, "l");
      bump();
      for (const paso of capa.pasos) {
        const to = posId(paso.nodo, L);
        if (paso.nodo !== prevId) {
          svg("line", { class: "path", x1: at[0], y1: at[1], x2: to[0], y2: to[1] }, fx);
          await moveToken(at, to);
        }
        at = to; prevId = paso.nodo;
        const ring = svg("circle", { class: "ring", cx: to[0], cy: to[1], r: 9 }, fx);
        say(`En <b>${name(paso.nodo)}</b> · cos = ${fmt(paso.similitud, 3)}`);
        for (const [nb, sim] of paso.evaluados) {
          const [nx, ny] = posId(nb, L);
          const ln = svg("line", { class: "flash", x1: to[0], y1: to[1], x2: nx, y2: ny }, fx);
          const ne = nodeEls[nb + "@" + L]; ne?.setAttribute("r", 8);
          bump();
          say(`&nbsp;&nbsp;↳ compara con ${name(nb)}: <span class="mono">${fmt(sim, 3)}</span>`);
          await sleep(170, run);
          ln.remove(); ne?.setAttribute("r", L ? 5.5 : 4);
        }
        ring.remove();
      }
      if (L > 0) {
        say(`Ningún vecino mejora → baja a la capa ${L - 1} desde <b>${name(capa.sale)}</b>`);
        const down = posId(capa.sale, L - 1);
        await moveToken(posId(capa.sale, L), down, 500);
        at = down; prevId = capa.sale;
      }
    }
    $("h-calc").textContent = H.calculos_distancia;
    H.resultado.forEach((r, k) => {
      const [x, y] = posId(r.id, 0);
      svg("circle", { class: "ring", cx: x, cy: y, r: 9 }, fx);
      svg("text", { class: "lbl strong", x: x + 10, y: y - 8 }, fx).textContent = `${k + 1}`;
    });
    const hits = H.resultado.filter((r) => exact.includes(r.id)).length;
    say(`Fin: ${H.calculos_distancia} distancias calculadas en vez de ${H.total_vectores}.`, "l");
    $("h-cmp").innerHTML = `<table><tr><th>#</th><th>HNSW (aprox.)</th><th>Exacta (X·q)</th></tr>
      ${H.resultado.map((r, k) => `<tr><td>${k + 1}</td><td>${exact.includes(r.id) ? "✓" : "✗"} ${name(r.id)}</td><td>${name(exact[k])}</td></tr>`).join("")}
      </table><p style="margin:6px 0 0">Recall: <b class="g">${hits}/${exact.length}</b> vecinos exactos encontrados.</p>`;
  }

  // =================================================================
  //  5. MAPA PCA (2D)
  // =================================================================
  async function secPca(t, run) {
    const P = state.pca, [v1, v2] = P.varianza_explicada;
    $("stage").innerHTML = head("5 · Mapa del menú: de ℝ⁷⁶⁸ a ℝ² con PCA",
      "Proyectamos los vectores sobre las 2 direcciones donde más varían. Es una \"sombra\" del espacio real: platos parecidos quedan cerca.") + `
      <div class="grid2">
        <div class="card rel" style="padding:8px"><svg id="pca" class="pca"></svg><div id="pca-tip" class="tip" hidden></div>
          <div class="legend">${Object.entries(CAT).map(([c, col]) => `<span><i style="background:${col}"></i>${c}</span>`).join("")}
          <span><i style="background:var(--violet);border-radius:2px;transform:rotate(45deg)"></i>tu consulta</span></div></div>
        <div class="card"><h3>Varianza explicada</h3>
          <div class="vbar"><span>PC1</span><div class="bar"><div id="vb1"></div></div><b class="mono">${fmt(v1 * 100, 1)}%</b></div>
          <div class="vbar"><span>PC2</span><div class="bar"><div id="vb2"></div></div><b class="mono">${fmt(v2 * 100, 1)}%</b></div>
          <p class="muted" style="font-size:12.5px">Las dos componentes conservan el ${fmt((v1 + v2) * 100, 1)}% de la variación total.
          El resto vive en las otras 766 direcciones: por eso dos puntos cercanos en el mapa no siempre son los más parecidos en ℝ⁷⁶⁸.</p>
          <h3 style="margin-top:14px">La consulta en 2D</h3>
          <div class="eq small">(x, y) = ( (q − μ)·w₁ , (q − μ)·w₂ )</div>
          <div class="eq">= (${fmt(t.consulta_xy[0], 3)} , ${fmt(t.consulta_xy[1], 3)})</div>
        </div>
      </div>
      ${theory(`<p>1. Se centra la matriz: X<sub>c</sub> = X − μ (μ = vector promedio de los platos).</p>
        <p>2. Matriz de covarianza: C = X<sub>c</sub>ᵀX<sub>c</sub> / (n − 1) ∈ ℝ<sup>768×768</sup>. Sus <b>vectores propios</b> w₁, w₂ con mayores
        valores propios λ₁ ≥ λ₂ son las direcciones de máxima varianza (se calculan con la SVD: X<sub>c</sub> = UΣVᵀ).</p>
        <p>3. Proyección: W = [w₁ w₂] ∈ ℝ<sup>768×2</sup> y cada punto pasa a (x − μ)W. Es una <b>transformación lineal</b> de ℝ⁷⁶⁸ en ℝ².</p>
        <p>Varianza explicada de la componente i: λ<sub>i</sub> / Σλ<sub>j</sub>.</p>`)}`;
    bindReplay();
    setTimeout(() => { $("vb1").style.width = "90%"; $("vb2").style.width = (v2 / v1) * 90 + "%"; }, 50);

    const el = $("pca"), W = Math.max(360, el.clientWidth || 600), H = Math.round(Math.min(470, Math.max(320, W * 0.66))), pad = 34;
    el.setAttribute("viewBox", `0 0 ${W} ${H}`); el.style.height = H + "px";
    const all = P.puntos.map((p) => p.xy).concat([t.consulta_xy]);
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const sx = (x) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * pad - 110);
    const sy = (y) => H - pad - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
    for (let k = 1; k < 6; k++) {
      svg("line", { class: "grid", x1: pad + (k * (W - 2 * pad)) / 6, x2: pad + (k * (W - 2 * pad)) / 6, y1: pad, y2: H - pad }, el);
      svg("line", { class: "grid", y1: pad + (k * (H - 2 * pad)) / 6, y2: pad + (k * (H - 2 * pad)) / 6, x1: pad, x2: W - pad }, el);
    }
    svg("text", { class: "axis", x: W - 8, y: H - 10, "text-anchor": "end" }, el).textContent = "PC1 →";
    svg("text", { class: "axis", x: 8, y: 18 }, el).textContent = "↑ PC2";
    const hits = new Set(t.vecinos), [qx, qy] = [sx(t.consulta_xy[0]), sy(t.consulta_xy[1])];
    const links = t.resultados.map((r) => svg("line", { class: "link", x1: qx, y1: qy, x2: sx(r.xy[0]), y2: sy(r.xy[1]) }, el));
    const tip = $("pca-tip");
    P.puntos.forEach((p, i) => {
      const hit = hits.has(p.id), cx = sx(p.xy[0]), cy = sy(p.xy[1]);
      const c = svg("circle", { class: "dot" + (hit ? " hit" : ""), cx, cy, r: hit ? 7 : 5.5, fill: catColor(p.categoria), style: `animation-delay:${(i * 25) / speed}ms` }, el);
      c.addEventListener("mouseenter", () => { tip.textContent = `${p.nombre} · cos = ${fmt(t.scores[p.id], 3)}`; tip.style.left = cx + 18 + "px"; tip.style.top = cy - 18 + "px"; tip.hidden = false; });
      c.addEventListener("mouseleave", () => (tip.hidden = true));
    });
    await sleep(41 * 25 + 300, run);
    const s = 9, star = svg("polygon", { points: `${qx},${qy - s} ${qx + s},${qy} ${qx},${qy + s} ${qx - s},${qy}`, fill: "#8b7bff", stroke: "#fff", "stroke-width": 1.5, class: "dot" }, el);
    svg("text", { class: "lbl q", x: qx + 12, y: qy - 10 }, el).textContent = "consulta";
    await sleep(350, run);
    links.forEach((l) => l.classList.add("on"));
    await sleep(700, run);
    const placed = [];
    t.resultados.forEach((r) => {
      const x = sx(r.xy[0]) + 10, y = sy(r.xy[1]) + 4, w = r.nombre.length * 6.8;
      for (const dy of [0, -15, 15, -30, 30, -45, 45]) {
        const b = { x, y: y - 11 + dy, w, h: 14 };
        if (!placed.some((o) => b.x < o.x + o.w && b.x + b.w > o.x && b.y < o.y + o.h && b.y + b.h > o.y) || dy === 45) {
          placed.push(b); svg("text", { class: "lbl", x, y: y + dy }, el).textContent = r.nombre; break;
        }
      }
    });
    star.parentNode.appendChild(star);
  }

  // =================================================================
  //  6. NÚMEROS COMPLEJOS
  // =================================================================
  async function secComplex(t, run) {
    const r = t.resultados[state.pick] || t.resultados[0];
    const [a, b] = t.consulta_xy, [c, d] = r.xy;
    const rq = Math.hypot(a, b), rv = Math.hypot(c, d);
    const phq = Math.atan2(b, a), phv = Math.atan2(d, c);
    // números complejos unitarios (normalizados, como q / ||q||)
    const [ua, ub, uc, ud] = [a / rq, b / rq, c / rv, d / rv];
    const re = ua * uc + ub * ud, im = ua * ud - ub * uc, th = Math.atan2(im, re);
    const C = (x, y, n = 4) => `${fmt(x, n)} ${y < 0 ? "−" : "+"} ${fmt(Math.abs(y), n)}i`;
    const sub = (s, k, cls = "") => `<b class="${cls}">${s}<sub>${k}</sub></b>`;
    $("stage").innerHTML = head("6 · El mismo ángulo con números complejos",
      "En el plano PCA cada punto (x, y) es un número complejo z = x + iy. Multiplicar complejos multiplica longitudes y suma ángulos: eso <b>gira</b> flechas.") +
      pickChips(t) + `
      <div class="grid2">
        <div class="card"><svg id="cx" class="cplx"></svg></div>
        <div class="card"><h3>Paso a paso</h3><ol class="steps">
          <li><span class="muted">1. Consulta y plato como complejos (forma binómica y polar)</span>
            <div class="eq">${sub("z", "q", "v")} = ${C(a, b)} = ${fmt(rq, 4)}·e<sup>i·${fmt(deg(phq), 1)}°</sup></div>
            <div class="eq">${sub("z", "v", "c")} = ${C(c, d)} = ${fmt(rv, 4)}·e<sup>i·${fmt(deg(phv), 1)}°</sup></div></li>
          <li><span class="muted">2. Normalizar: ẑ = z / |z| (igual que q / ‖q‖) → quedan sobre la circunferencia unitaria</span>
            <div class="eq">ẑ<sub>q</sub> = ${C(ua, ub)} &nbsp; ẑ<sub>v</sub> = ${C(uc, ud)}</div></li>
          <li><span class="muted">3. Conjugado: reflejar sobre el eje real (cambia el signo del ángulo)</span>
            <div class="eq">ẑ̄<sub>q</sub> = ${C(ua, -ub)} = e<sup>−i·${fmt(deg(phq), 1)}°</sup></div></li>
          <li><span class="muted">4. Producto w = ẑ̄<sub>q</sub> · ẑ<sub>v</sub> = e<sup>i(φ<sub>v</sub> − φ<sub>q</sub>)</sup> = e<sup>iθ</sup></span>
            <div class="eq"><b class="g">w</b> = (ac + bd) + (ad − bc)i = ${C(re, im)}</div>
            <div class="eq small">Re(w) = cos θ = <b class="g">producto punto</b> de las direcciones = ${fmt(re, 4)}</div>
            <div class="eq small">Im(w) = sen θ = det[q v] (área con signo) = ${fmt(im, 4)}</div></li>
          <li><span class="muted">5. Multiplicar por e<sup>iθ</sup> = rotar θ = ${fmt(deg(th), 1)}°: lleva ẑ<sub>q</sub> exactamente a ẑ<sub>v</sub></span>
            <div class="eq small">cos θ<sub>2D</sub> = ${fmt(re, 4)} &nbsp; vs &nbsp; cos θ real en ℝ⁷⁶⁸ = ${fmt(r.similitud, 4)}</div></li>
        </ol></div>
      </div>
      ${theory(`<p><b>Fórmula de Euler:</b> e<sup>iθ</sup> = cos θ + i sen θ. Todo complejo se escribe z = r·e<sup>iφ</sup> (forma polar) y
        r₁e<sup>iφ₁</sup> · r₂e<sup>iφ₂</sup> = r₁r₂·e<sup>i(φ₁+φ₂)</sup>: <b>se multiplican las longitudes y se suman los ángulos</b>.</p>
        <p>Por eso z̄<sub>q</sub>·z<sub>v</sub> = |z<sub>q</sub>||z<sub>v</sub>|·e<sup>i(φ<sub>v</sub>−φ<sub>q</sub>)</sup>, y su parte real
        |z<sub>q</sub>||z<sub>v</sub>| cos θ es el <b>producto punto</b> de los vectores (a, b) y (c, d). Con complejos unitarios queda solo e<sup>iθ</sup>.</p>
        <p><b>Producto interno en ℂⁿ:</b> ⟨u, v⟩ = Σ ū<sub>k</sub> v<sub>k</sub>. El conjugado hace que ⟨u, u⟩ = Σ|u<sub>k</sub>|² sea real y positivo
        (una longitud al cuadrado). Nuestros embeddings son reales: ū = u y vuelve a ser el producto punto, cos θ = Re⟨u, v⟩ / (‖u‖‖v‖).</p>
        <p class="muted">θ<sub>2D</sub> es el ángulo en la "sombra" PCA; el ángulo verdadero entre q y v vive en 768 dimensiones (pestaña 2).</p>`)}`;
    bindReplay(); bindPick();

    const el = $("cx"), W = 460, H = 400, O = [W / 2, H / 2], R = 150;
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const P = (x, y) => [O[0] + x * R, O[1] - y * R];
    for (let k = -4; k <= 4; k++) {
      svg("line", { class: "gridl", x1: O[0] + (k * R) / 2, x2: O[0] + (k * R) / 2, y1: 10, y2: H - 10 }, el);
      svg("line", { class: "gridl", y1: O[1] + (k * R) / 2, y2: O[1] + (k * R) / 2, x1: 10, x2: W - 10 }, el);
    }
    svg("line", { class: "axis", x1: 10, y1: O[1], x2: W - 10, y2: O[1] }, el);
    svg("line", { class: "axis", x1: O[0], y1: 10, x2: O[0], y2: H - 10 }, el);
    // subTxt: subíndice; si empieza con "^" es exponente
    const txt = (x, y, main, subTxt, col) => {
      const e = svg("text", { class: "lbl", x, y, style: `fill:${col}` }, el);
      e.appendChild(document.createTextNode(main));
      if (subTxt) {
        const sup = subTxt.startsWith("^");
        const s = svg("tspan", { dy: sup ? -7 : 4, "font-size": 10 }, e);
        s.textContent = sup ? subTxt.slice(1) : subTxt;
      }
      return e;
    };
    txt(W - 26, O[1] - 8, "Re", "", "#8d93b3"); txt(O[0] + 8, 22, "Im", "", "#8d93b3");
    txt(O[0] + 6, O[1] - R - 4, "i", "", "#8d93b3");
    const unit = svg("circle", { class: "circle", cx: O[0], cy: O[1], r: R, opacity: 0 }, el);
    const defs = svg("defs", {}, el);
    [["av", "#8b7bff"], ["ac", "#3dd6d0"], ["ag", "#3ddc97"], ["aw", "#ffffff"]].forEach(([id, col]) => {
      const m = svg("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
      svg("path", { d: "M0,0 L10,5 L0,10 z", fill: col }, m);
    });
    const arrow = (col, mk, dash) => svg("line", { x1: O[0], y1: O[1], x2: O[0], y2: O[1], stroke: col, "stroke-width": 2.6, "marker-end": `url(#${mk})`, ...(dash ? { "stroke-dasharray": "5 4" } : {}) }, el);
    const setTip = (ln, [x, y]) => { ln.setAttribute("x2", x); ln.setAttribute("y2", y); };
    const polar = (ang, len) => P((Math.cos(ang) * len) / R, (Math.sin(ang) * len) / R);
    const labelAt = (ang, len, main, subTxt, col) => { const [x, y] = polar(ang, len + 22); return txt(x - 10, y + 5, main, subTxt, col); };
    const steps = [...document.querySelectorAll(".steps li")];
    const show = (i) => steps[i].classList.add("show");

    // 1. vectores originales (escalados para verse)
    show(0);
    const S = (R * 0.55) / Math.max(rq, rv), lenQ0 = rq * S, lenV0 = rv * S;
    const zq = arrow("#8b7bff", "av"), zv = arrow("#3dd6d0", "ac");
    await tween(600, (p) => setTip(zq, polar(phq, lenQ0 * p)), run);
    await tween(600, (p) => setTip(zv, polar(phv, lenV0 * p)), run);
    const lq = labelAt(phq, lenQ0, "z", "q", "#8b7bff"), lv = labelAt(phv, lenV0, "z", "v", "#3dd6d0");
    await sleep(500, run);

    // 2. normalizar: llegan a la circunferencia unitaria
    show(1);
    lq.remove(); lv.remove();
    unit.setAttribute("opacity", 1);
    await tween(900, (p) => {
      setTip(zq, polar(phq, lenQ0 + (R - lenQ0) * p));
      setTip(zv, polar(phv, lenV0 + (R - lenV0) * p));
    }, run);
    labelAt(phq, R, "ẑ", "q", "#8b7bff"); labelAt(phv, Math.abs(phv - phq) < 0.35 ? R + 22 : R, "ẑ", "v", "#3dd6d0");
    await sleep(400, run);

    // 3. conjugado
    show(2);
    const cj = arrow("#b9b0ff", "av", true);
    await tween(800, (p) => setTip(cj, P(ua, ub - 2 * ub * p)), run);
    const cjl = labelAt(-phq, R, "ẑ̄", "q", "#b9b0ff");
    await sleep(400, run);

    // 4. w = conj(ẑq)·ẑv = e^{iθ}
    show(3);
    const w = svg("line", { x1: O[0], y1: O[1], x2: O[0], y2: O[1], stroke: "#3ddc97", "stroke-width": 3, "marker-end": "url(#ag)" }, el);
    await tween(700, (p) => setTip(w, P(re * p, im * p)), run);
    cj.setAttribute("opacity", 0.3); cjl.remove();
    labelAt(th, R, "w = e", "^iθ", "#3ddc97");
    const [wx, wy] = P(re, im);
    svg("line", { x1: wx, y1: wy, x2: wx, y2: O[1], stroke: "rgba(255,255,255,.45)", "stroke-dasharray": "3 3" }, el);
    const reSeg = svg("line", { x1: O[0], y1: O[1], x2: O[0], y2: O[1], stroke: "#3ddc97", "stroke-width": 7, opacity: 0.4, "stroke-linecap": "round" }, el);
    await tween(600, (p) => reSeg.setAttribute("x2", O[0] + (wx - O[0]) * p), run);
    txt(re >= 0 ? O[0] - 128 : O[0] + 10, O[1] + 20, re >= 0 ? "Re(w) = cos θ →" : "← Re(w) = cos θ", "", "#3ddc97");
    await sleep(500, run);

    // 5. rotar ẑq multiplicando por e^{iθ}
    show(4);
    const rot = arrow("#ffffff", "aw", true), arcEl = svg("path", { fill: "none", stroke: "#ffd166", "stroke-width": 2.2 }, el);
    const rr = R * 0.42;
    await tween(1500, (p) => {
      const ea = phq + th * p, large = Math.abs(th * p) > Math.PI ? 1 : 0, sweep = th > 0 ? 0 : 1;
      setTip(rot, polar(ea, R));
      arcEl.setAttribute("d", `M${O[0] + rr * Math.cos(phq)},${O[1] - rr * Math.sin(phq)} A${rr},${rr} 0 ${large} ${sweep} ${O[0] + rr * Math.cos(ea)},${O[1] - rr * Math.sin(ea)}`);
    }, run);
    txt(16, 26, `θ = ${fmt(deg(th), 1)}°  (arco dorado)`, "", "#ffd166");
  }

  // =================================================================
  //  ESPACIO 3D (estilo Embedding Projector), usado en la pestaña 1
  // =================================================================
  function mapCanvasCard(title) {
    return `<div class="card map-card">
        ${title ? `<h3>${title}</h3>` : ""}
        <div class="rel"><canvas id="map3d"></canvas><div id="map-status" class="map-status"></div></div>
        <div class="map-tools">
          <label><input type="checkbox" id="map-names"> Ver todos los nombres</label>
          <label><input type="checkbox" id="map-spin" checked> Girar solo</label>
          <span class="muted">Arrastra para girar · rueda para acercar · toca un plato</span>
        </div>
        <div class="legend">${Object.entries(CAT).map(([c, col]) => `<span><i style="background:${col}"></i>${c}</span>`).join("")}
          <span><i style="background:#fff;border-radius:2px;transform:rotate(45deg)"></i>tu consulta</span></div>
      </div>`;
  }
  function mapSideCard() {
    const [v1, v2, v3] = state.pca.varianza_explicada;
    return `<div class="card map-side">
        <h3 id="nn-title">Toca un plato</h3>
        <div id="nn-list" class="nn-list"><p class="plain small">Haz clic en cualquier punto para ver los platos más parecidos a él.</p></div>
        <p class="plain small" style="margin-top:10px">El dibujo es una proyección PCA a 3D de los vectores de ℝ⁷⁶⁸ (conserva el
          ${fmt((v1 + v2 + v3) * 100, 0)}% de la varianza). Girarlo es multiplicar cada punto por una <b>matriz de rotación</b>.</p>
      </div>`;
  }

  function nearestTo(id) {
    const v = vecOf(id);
    return state.matrix.filas.filter((f) => f.id !== id)
      .map((f) => ({ id: f.id, nombre: f.nombre, sim: dot(v, vecOf(f.id)) }))
      .sort((a, b) => b.sim - a.sim).slice(0, 5);
  }

  // Con animate = true se anima la búsqueda: la consulta entra al espacio,
  // se compara con cada plato y se iluminan los 5 más parecidos.
  function startMap(t, run, animate = false) {
    const cv = $("map3d"), ctx = cv.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; ctx.scale(dpr, dpr);

    const pts = state.pca.puntos.map((p) => ({ ...p, v: p.xyz }));
    const qv = t ? t.consulta_xyz : null;
    const R = Math.max(...pts.map((p) => Math.hypot(...p.v)), qv ? Math.hypot(...qv) : 0) || 1;
    let yaw = 0.7, pitch = -0.3, zoom = 1, hover = null, drag = null, moved = false, last = null, el = 0;
    let selected = t ? "__q" : null, anim = !!(t && animate), shownTop = -1;
    const opt = { names: false, spin: true };
    $("map-names").addEventListener("change", (e) => (opt.names = e.target.checked));
    $("map-spin").addEventListener("change", (e) => (opt.spin = e.target.checked));

    const T_INTRO = 900, T_FLY = 1300, T_SCAN = 2600, T_RANK = 1600;
    const tFly = T_INTRO, tScan = tFly + T_FLY, tRank = tScan + T_SCAN, tEnd = tRank + T_RANK;
    const clamp = (x) => Math.max(0, Math.min(1, x));
    const scores = t ? t.scores : {};
    const sVals = Object.values(scores), sMin = Math.min(...sVals), sMax = Math.max(...sVals);
    const scanOrder = pts.map((p) => p.id);

    const neighbors = () => {
      if (!selected) return [];
      if (selected === "__q") return t.resultados.map((r) => ({ id: r.id, nombre: r.nombre, sim: r.similitud }));
      return nearestTo(selected);
    };
    function renderSide(count) {
      if (!selected) return;
      const list = neighbors().slice(0, count ?? 5);
      $("nn-title").innerHTML = selected === "__q" ? `Más parecidos a “${esc(t.consulta)}”` : `Más parecidos a ${esc(rowOf(selected).nombre)}`;
      $("nn-list").innerHTML = (list.length ? list.map((n, i) => `<div class="nn-row"><span class="rk">${i + 1}</span>
        <span class="nm">${esc(n.nombre)}</span><div class="sb"><div style="width:${Math.max(0, (n.sim - 0.7) / 0.3) * 100}%"></div></div>
        <span class="sv">${fmt(n.sim, 3)}</span></div>`).join("") : `<p class="plain small">Buscando…</p>`) +
        `<p class="plain small" style="margin-top:8px">cos θ calculado con los 768 números (no con el dibujo).</p>`;
    }
    if (anim) renderSide(0); else renderSide();

    const rotate = ([x, y, z]) => {
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      return [x1, cp * y - sp * z1, sp * y + cp * z1];
    };
    const project = (v, k = 1) => {
      const r = rotate([(v[0] / R) * k, (v[1] / R) * k, (v[2] / R) * k]);
      const persp = 3 / (3 + r[2]);
      const s = Math.min(W, H) * 0.56 * zoom * persp;
      return { X: W / 2 + r[0] * s, Y: H / 2 - r[1] * s, z: r[2], persp };
    };
    const label = (text, X, Y, color, bold) => {
      ctx.font = `${bold ? "700 " : ""}12px Segoe UI, system-ui, sans-serif`;
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(11,14,26,.9)"; ctx.strokeText(text, X + 9, Y - 7);
      ctx.fillStyle = color; ctx.fillText(text, X + 9, Y - 7);
    };
    const status = (html) => { const s = $("map-status"); if (s && s.innerHTML !== html) s.innerHTML = html; };

    function frame(now) {
      if (run !== token || !document.body.contains(cv)) return;
      const dt = last == null ? 0 : Math.min(500, now - last);
      last = now;
      el += dt * speed;
      if (opt.spin && !drag) yaw += 0.00018 * dt * speed * (anim ? 0.5 : 1);
      ctx.clearRect(0, 0, W, H);

      const k = ease(clamp(el / T_INTRO));
      const pFly = anim ? ease(clamp((el - tFly) / T_FLY)) : 1;
      const pScan = anim ? clamp((el - tScan) / T_SCAN) : 1;
      const pRank = anim ? clamp((el - tRank) / T_RANK) : 1;
      if (anim && el >= tEnd) { anim = false; status(""); renderSide(); }

      [[1, 0, 0, "PC1"], [0, 1, 0, "PC2"], [0, 0, 1, "PC3"]].forEach(([x, y, z, nm]) => {
        const a = project([-x * R, -y * R, -z * R], k), b = project([x * R, y * R, z * R], k);
        ctx.strokeStyle = "rgba(255,255,255,.09)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(a.X, a.Y); ctx.lineTo(b.X, b.Y); ctx.stroke();
        ctx.fillStyle = "rgba(255,255,255,.25)"; ctx.font = "11px Segoe UI, sans-serif"; ctx.fillText(nm, b.X + 4, b.Y);
      });

      const proj = pts.map((p) => ({ p, ...project(p.v, k) }));
      const byId = Object.fromEntries(proj.map((o) => [o.p.id, o]));
      let qp = null;
      if (qv && (!anim || el >= tFly)) {
        const start = [qv[0], qv[1] + 2.2 * R, qv[2]];
        qp = project(start.map((s0, i) => s0 + (qv[i] - s0) * pFly), k);
      }

      const scanned = anim ? Math.floor(pScan * scanOrder.length) : scanOrder.length;
      const scannedSet = new Set(scanOrder.slice(0, scanned));
      const topCount = anim ? (el < tRank ? 0 : Math.min(5, Math.ceil(pRank * 5))) : 5;
      if (anim && topCount !== shownTop) { shownTop = topCount; renderSide(topCount); }
      if (anim && el >= tScan && el < tRank) status(`Calculando q·vᵢ con cada plato: <b>${Math.min(scanned, 41)}</b> / 41`);
      else if (anim && el >= tRank) status(`Top-<b>${topCount}</b>: los platos más parecidos`);
      else if (anim && el >= tFly) status("La consulta entra a la base de datos…");

      const near = anim ? (selected === "__q" ? neighbors().slice(0, topCount) : []) : neighbors();
      const nearSet = new Set(near.map((n) => n.id));
      const src = selected === "__q" ? qp : selected ? byId[selected] : null;

      // rayos de comparación (producto punto con cada plato)
      if (anim && qp && el >= tScan && el < tRank) {
        for (let j = Math.max(0, scanned - 4); j < scanned; j++) {
          const o = byId[scanOrder[j]], fade = (j - (scanned - 4)) / 4;
          ctx.strokeStyle = `rgba(61,214,208,${0.15 + 0.7 * fade})`; ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.moveTo(qp.X, qp.Y); ctx.lineTo(o.X, o.Y); ctx.stroke();
        }
      }
      if (src) near.forEach((n) => {
        const o = byId[n.id];
        const g = ctx.createLinearGradient(src.X, src.Y, o.X, o.Y);
        g.addColorStop(0, "rgba(255,255,255,.9)"); g.addColorStop(1, "rgba(255,209,102,.5)");
        ctx.strokeStyle = g; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(src.X, src.Y); ctx.lineTo(o.X, o.Y); ctx.stroke();
      });

      proj.sort((a, b) => b.z - a.z).forEach((o) => {
        const id = o.p.id, isN = nearSet.has(id), isS = selected === id, isH = hover === id;
        let alpha = Math.max(0.45, Math.min(1, 1.1 - o.z * 0.45)), rad = (isN || isS ? 7 : 5) * o.persp;
        if (t && selected === "__q" && scannedSet.has(id) && !isN) {
          const s = (scores[id] - sMin) / (sMax - sMin || 1); // ya comparado: brilla según su cos θ
          alpha = anim && el < tRank ? 0.2 + 0.8 * s : 0.25;
          rad = (4 + 3 * s) * o.persp;
        } else if (src && !isN && !isS) alpha = 0.25;
        ctx.globalAlpha = alpha;
        ctx.shadowColor = catColor(o.p.categoria); ctx.shadowBlur = isN || isS || isH ? 16 : 6;
        ctx.fillStyle = catColor(o.p.categoria);
        ctx.beginPath(); ctx.arc(o.X, o.Y, rad, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
        if (isS || isN) { ctx.strokeStyle = isS ? "#fff" : "#ffd166"; ctx.lineWidth = 2; ctx.stroke(); }
        ctx.globalAlpha = 1;
      });

      if (qp) {
        const s = 9 * qp.persp;
        if (anim && pFly < 1) {
          const tail = project([qv[0], qv[1] + 2.2 * R * (1 - pFly) + 0.5 * R, qv[2]], k);
          const g = ctx.createLinearGradient(tail.X, tail.Y, qp.X, qp.Y);
          g.addColorStop(0, "rgba(139,123,255,0)"); g.addColorStop(1, "rgba(139,123,255,.9)");
          ctx.strokeStyle = g; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(tail.X, tail.Y); ctx.lineTo(qp.X, qp.Y); ctx.stroke();
        }
        const pulse = anim ? 1 + 0.25 * Math.sin(el / 120) : 1;
        ctx.shadowColor = "#8b7bff"; ctx.shadowBlur = 24; ctx.fillStyle = "#fff";
        ctx.beginPath(); ctx.moveTo(qp.X, qp.Y - s * pulse); ctx.lineTo(qp.X + s * pulse, qp.Y); ctx.lineTo(qp.X, qp.Y + s * pulse); ctx.lineTo(qp.X - s * pulse, qp.Y); ctx.closePath(); ctx.fill();
        ctx.shadowBlur = 0;
        label("consulta q", qp.X, qp.Y, "#cfc8ff", true);
      }
      proj.forEach((o) => {
        const showName = opt.names || nearSet.has(o.p.id) || selected === o.p.id || hover === o.p.id;
        if (showName) label(o.p.nombre, o.X, o.Y, selected === o.p.id ? "#fff" : nearSet.has(o.p.id) ? "#ffd166" : "#c9cce0", selected === o.p.id);
      });
      requestAnimationFrame(frame);
    }

    const pick = (e) => {
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = null, bd = 14;
      pts.forEach((p) => { const o = project(p.v), d = Math.hypot(o.X - mx, o.Y - my); if (d < bd) { bd = d; best = p.id; } });
      if (qv) { const o = project(qv); if (Math.hypot(o.X - mx, o.Y - my) < bd) best = "__q"; }
      return best;
    };
    cv.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY }; moved = false; });
    cv.addEventListener("mousemove", (e) => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        yaw += dx * 0.008; pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.008));
        drag = { x: e.clientX, y: e.clientY };
      } else {
        hover = pick(e); cv.style.cursor = hover ? "pointer" : "grab";
      }
    });
    cv.addEventListener("mouseup", (e) => {
      if (!moved) {
        const id = pick(e);
        if (id) { anim = false; status(""); selected = id; renderSide(); }
      }
      drag = null;
    });
    cv.addEventListener("mouseleave", () => { drag = null; hover = null; });
    cv.addEventListener("wheel", (e) => { e.preventDefault(); zoom = Math.max(0.5, Math.min(3, zoom * (e.deltaY < 0 ? 1.1 : 0.9))); }, { passive: false });
    requestAnimationFrame(frame);
  }

  const SECTIONS = { vector: secVector, dot: secDot, matrix: secMatrix, hnsw: secHnsw, pca: secPca, complex: secComplex };

  // ---------------- API pública ----------------
  async function show(traces, i = 0, tab = "vector") {
    if (!traces || !traces.length) return;
    await ready; // los datos del menú (matriz, PCA, grafo) deben estar cargados
    state.traces = traces; state.idx = i; state.pick = 0; state.tab = tab;
    renderTraceTabs(); renderTabs(); lightPipeline(); play();
  }

  async function explore(text) {
    text = text.trim(); if (!text) return;
    $("explorer-input").value = text;
    const res = await fetch("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ consulta: text }) });
    const tr = await res.json(); tr.directa = true;
    show([tr], 0, "vector");
  }

  async function init() {
    renderPipeline(); renderTabs();
    $("speed").addEventListener("change", (e) => (speed = +e.target.value));
    $("explorer").addEventListener("submit", (e) => { e.preventDefault(); explore($("explorer-input").value); });
    await loadData();
    play();
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (!cur() || state.tab === "vector" || state.tab === "pca") play(); }, 300); });
  }

  const ready = init();
  return { show, explore };
})();
