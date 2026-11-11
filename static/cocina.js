// Panel de cocina: muestra los pedidos por estado y se actualiza cada 3 segundos.
const money = (n) => "$" + Number(n).toLocaleString("es-CO");
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const LABELS = { recibido: "Recibidos", preparando: "Preparando", listo: "Listos", entregado: "Entregados" };
const seen = new Set();
let firstLoad = true;

async function load() {
  const { estados, pedidos } = await (await fetch("/api/orders")).json();
  const board = document.getElementById("board");
  board.innerHTML = estados.map((estado, idx) => {
    const list = pedidos.filter((p) => p.estado === estado);
    const next = estados[idx + 1];
    const cards = list.map((p) => {
      const isNew = !firstLoad && !seen.has(p.id);
      const items = p.items.map((i) =>
        `<li>${i.cantidad} × ${escapeHtml(i.nombre)}${i.nota ? `<br><span class="note">↳ ${escapeHtml(i.nota)}</span>` : ""}</li>`).join("");
      return `<div class="order${isNew ? " new" : ""}">
        <div class="top"><span>#${p.id} · ${escapeHtml(p.cliente)}</span><span class="time">${p.creado.slice(11, 16)}</span></div>
        <ul>${items}</ul>
        <div class="bottom"><b>${money(p.total)}</b>
          ${next ? `<button data-id="${p.id}" data-next="${next}">${LABELS[next].replace(/s$/, "")} →</button>` : ""}</div>
      </div>`;
    }).join("");
    return `<section class="col" data-estado="${estado}">
      <h2><span>${LABELS[estado]}</span><span>${list.length}</span></h2>
      ${cards || '<p class="empty">Sin pedidos</p>'}</section>`;
  }).join("");
  pedidos.forEach((p) => seen.add(p.id));
  firstLoad = false;
}

document.getElementById("board").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-id]");
  if (!btn) return;
  await fetch("/api/orders/" + btn.dataset.id, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ estado: btn.dataset.next }),
  });
  load();
});

load();
setInterval(load, 3000);
