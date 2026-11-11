// ===================== Utilidades =====================
const $ = (id) => document.getElementById(id);
const money = (n) => "$" + Number(n).toLocaleString("es-CO");
const now = () => new Date().toLocaleTimeString("es-CO", { hour: "2-digit", minute: "2-digit" });
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Formato estilo WhatsApp: *negrita*, _cursiva_, viñetas y saltos de línea
function formatText(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\*(\S(?:.*?\S)?)\*/g, "<b>$1</b>")
    .replace(/(^|\s)_(\S(?:.*?\S)?)_(?=\s|$)/g, "$1<i>$2</i>")
    .replace(/^\s*[-*]\s+/gm, "• ")
    .replace(/\n/g, "<br>");
}

// ===================== Sesión =====================
function getSessionId() {
  let id = null;
  try { id = localStorage.getItem("session_id"); } catch (e) {}
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random());
    try { localStorage.setItem("session_id", id); } catch (e) {}
  }
  return id;
}
let sessionId = getSessionId();

// ===================== Chat =====================
const messagesEl = $("messages");

function addMessage(text, who, extraHtml = "") {
  const div = document.createElement("div");
  div.className = "msg " + who;
  const ticks = who === "out" ? '<span class="ticks">✓✓</span>' : "";
  div.innerHTML = formatText(text) + extraHtml + `<span class="meta">${now()}${ticks}</span>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function addNote(text, onClick) {
  const div = document.createElement("div");
  div.className = "msg search-note";
  div.textContent = text;
  if (onClick) div.addEventListener("click", onClick);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function ticketHtml(p) {
  const rows = p.items.map((i) =>
    `<div><span>${i.cantidad} × ${escapeHtml(i.nombre)}</span><span>${money(i.subtotal)}</span></div>`).join("");
  return `<div class="ticket"><b>🧾 Pedido #${p.id} · ${escapeHtml(p.cliente)}</b>${rows}
    <div><b>Total</b><b>${money(p.total)}</b></div><div class="muted">Estado: ${p.estado}</div></div>`;
}

function welcome() {
  messagesEl.innerHTML = "";
  addMessage("¡Hola! 👋 Bienvenido a *La Fonda de la Abuela*.\nCuéntame qué se te antoja (por ejemplo: \"algo calientito con pollo\") y te ayudo a armar tu pedido.", "in");
}

async function send(text) {
  text = text.trim();
  if (!text) return;
  $("input").value = "";
  $("send").disabled = true;
  $("chips").classList.add("hidden");
  addMessage(text, "out");
  const typing = addMessage("escribiendo…", "in typing");
  $("status").textContent = "escribiendo…";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const data = await res.json();
    typing.remove();
    if (data.vector_trace && data.vector_trace.length) {
      data.vector_trace.forEach((t, i) =>
        addNote(`🔎 Búsqueda semántica: “${t.consulta}”  → ver en el laboratorio`, () => Lab.show(data.vector_trace, i)));
      Lab.show(data.vector_trace, 0);
    }
    if (data.error) addMessage(data.error, "error");
    else addMessage(data.reply || "…", "in", data.pedido ? ticketHtml(data.pedido) : "");
    renderCart(data.cart);
  } catch (e) {
    typing.remove();
    addMessage("No hay conexión con el servidor.", "error");
  } finally {
    $("status").textContent = "en línea";
    $("send").disabled = false;
    $("input").focus();
  }
}

$("composer").addEventListener("submit", (e) => { e.preventDefault(); send($("input").value); });
document.querySelectorAll("#chips button").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
$("reset").addEventListener("click", async () => {
  await fetch("/api/reset/" + encodeURIComponent(sessionId), { method: "POST" });
  try { localStorage.removeItem("session_id"); } catch (e) {}
  sessionId = getSessionId();
  renderCart({ items: [], total: 0 });
  $("chips").classList.remove("hidden");
  welcome();
});

// ===================== Carrito =====================
function renderCart(cart) {
  const count = cart.items.reduce((s, i) => s + i.cantidad, 0);
  $("cart").classList.toggle("hidden", count === 0);
  $("cart-count").textContent = count;
  $("cart-total").textContent = money(cart.total);
  $("cart-items").innerHTML = cart.items.map((i) =>
    `<li><span>${i.cantidad} × ${escapeHtml(i.nombre)}${i.nota ? `<br><span class="note">${escapeHtml(i.nota)}</span>` : ""}</span>
     <span>${money(i.subtotal)}</span></li>`).join("");
}
$("cart-toggle").addEventListener("click", () => $("cart-items").classList.toggle("hidden"));

// ===================== Inicio =====================
welcome();
fetch("/api/info").then((r) => r.json()).then((i) => { $("restaurant-name").textContent = i.restaurante; });
