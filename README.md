# La Fonda de la Abuela: RAG de pedidos con búsqueda semántica

Asistente de pedidos para un restaurante colombiano ficticio, con un chat que imita WhatsApp. El cliente escribe en lenguaje natural ("algo calientito con pollo") y el sistema encuentra los platos con **búsqueda semántica** en una **base de datos vectorial** (ChromaDB). Luego una IA conversacional (Google Gemini) responde y arma el pedido. Junto al chat, un **Laboratorio vectorial** muestra con animaciones el álgebra lineal de cada búsqueda: el vector de la consulta, el producto punto, la búsqueda como multiplicación matriz·vector, una simulación de HNSW (el índice que usa ChromaDB), un mapa PCA y el mismo ángulo visto con números complejos.

---

## 1. Instalación y ejecución

Requisitos: Python 3.10 o superior y una clave de Gemini ([Google AI Studio](https://aistudio.google.com/apikey)).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # y pega tu GEMINI_API_KEY dentro de .env
python run.py
```

- Chat: <http://localhost:8000>
- Cocina: <http://localhost:8000/cocina>

La primera ejecución descarga el modelo de embeddings (~1,1 GB) y vectoriza el menú. **Hazlo antes de la demo**: después funciona sin descargar nada más. Solo Gemini necesita internet.

Pruebas: `python -m pytest`

### Modelo de Gemini y cuotas gratuitas

Por defecto se usa `GEMINI_MODEL=gemini-3.5-flash-lite`: responde en 2–3 s y maneja bien las herramientas. Cada mensaje del chat gasta entre 2 y 3 solicitudes a la API (una por cada vez que Gemini llama a una herramienta, más la respuesta final).

En el plan gratuito, algunos modelos tienen cuotas muy bajas. Por ejemplo, `gemini-3.5-flash` permite **20 solicitudes al día**, unos 7 mensajes. Si el chat dice que se agotó la cuota, cambia `GEMINI_MODEL` en `.env` y reinicia, o revisa tus límites en <https://aistudio.google.com/rate-limit>. El **buscador del laboratorio no usa Gemini**, así que la parte de álgebra funciona aunque no haya cuota ni internet.

---

## 2. ¿Qué es un RAG?

**R**etrieval-**A**ugmented **G**eneration: en vez de que la IA "recuerde" el menú (y se invente platos), en cada consulta se **recupera** la información relevante de una base de datos y se le entrega como contexto.

```
 Cliente: "quiero algo calientito con pollo"
        │
        ▼
 ┌──────────────┐  decide buscar   ┌──────────────────────────────┐
 │    Gemini    │ ───────────────▶ │ buscar_platos("sopa caliente │
 │ (generación) │                  │   con pollo")                │
 └──────────────┘                  └──────────────┬───────────────┘
        ▲                                         │ 1. texto → vector q ∈ ℝ⁷⁶⁸
        │                                         │    (sentence-transformer)
        │                                         ▼
        │                          ┌──────────────────────────────┐
        │   top-5 platos +         │   ChromaDB (41 vectores)     │
        └──── similitud ◀───────── │ 2. los k vectores v más      │
              (aumentación)        │    cercanos a q (coseno)     │
                                   └──────────────────────────────┘
        │
        ▼
 "¡Te recomiendo el ajiaco santafereño ($32.000)…!"  → carrito → cocina
```

Gemini usa *function calling*: tiene herramientas (`buscar_platos`, `agregar_al_carrito`, `quitar_del_carrito`, `ver_carrito`, `confirmar_pedido`) y decide cuándo llamarlas. Solo puede agregar al carrito platos con un `id` que exista en el menú, así que no puede inventar platos.

---

## 3. El álgebra lineal detrás

### 3.1 Embeddings: texto → vector

Un **embedding** es una función que convierte un texto en un vector de números reales:

$$f: \text{texto} \rightarrow \mathbb{R}^{768}$$

Usamos el modelo `intfloat/multilingual-e5-base`. Es una red neuronal entrenada para que **textos con significado parecido queden como vectores que apuntan en direcciones parecidas**. "Sopa caliente de pollo" y "ajiaco con gallina" no comparten casi palabras, pero sus vectores son cercanos.

E5 está entrenado específicamente para **búsqueda**: la consulta lleva el prefijo `query: ` y cada documento el prefijo `passage: `. Comparamos tres modelos con las mismas consultas, y E5 fue el único que ubicó primero el plato correcto en todas ("ajiaco", "bandeja paisa", "sopa de pollo con papa y mazorca"…). Los modelos *paraphrase*, entrenados para comparar frases parecidas entre sí, fallaban con nombres propios.

Cada plato del menú se describe con un texto (nombre, categoría, descripción, ingredientes y características). Ese texto se convierte en un vector $v_i \in \mathbb{R}^{768}$ y se guarda en ChromaDB. Código: [app/menu.py](app/menu.py) (`dish_to_document`) y [app/embeddings.py](app/embeddings.py).

### 3.2 Producto punto y norma

Para $a, b \in \mathbb{R}^n$:

$$a \cdot b = \sum_{i=1}^{n} a_i b_i \qquad\qquad \|a\| = \sqrt{a \cdot a} = \sqrt{\sum_{i=1}^{n} a_i^2}$$

### 3.3 Similitud coseno

El ángulo $\theta$ entre dos vectores cumple:

$$\cos\theta = \frac{a \cdot b}{\|a\|\,\|b\|}$$

- $\cos\theta = 1$: misma dirección, significado muy parecido.
- $\cos\theta = 0$: vectores ortogonales, sin relación.
- $\cos\theta = -1$: direcciones opuestas.

La similitud coseno **ignora la longitud** del vector y solo mira la dirección, que es donde el modelo codifica el significado.

**Observación (anisotropía):** con E5, *todos* los cosenos salen entre ~0,78 y ~0,90, incluso para "sushi", que no está en el menú. Eso pasa porque los vectores del modelo no se reparten por toda la esfera, sino que ocupan un **cono estrecho** alrededor de una dirección común. Por eso lo importante no es el valor absoluto del coseno, sino el **orden** (ranking) y la **diferencia** entre el primero y el resto. Si al vector le restamos el promedio $\mu$ (centrado, como hace el PCA de la sección 3.7), las diferencias se vuelven mucho más visibles.

**Ejemplo a mano** en $\mathbb{R}^3$: $q = (1, 2, 2)$ y $v = (2, 1, 2)$.

- $q \cdot v = 1\cdot2 + 2\cdot1 + 2\cdot2 = 8$
- $\|q\| = \sqrt{1+4+4} = 3$ y $\|v\| = \sqrt{4+1+4} = 3$
- $\cos\theta = \dfrac{8}{3 \cdot 3} = 0{,}889$, es decir, $\theta \approx 27{,}3°$: son vectores parecidos.

### 3.4 Normalización: por qué ‖q‖ = 1

Antes de guardarlos, todos los vectores se **normalizan**: $\hat v = v / \|v\|$, de modo que $\|\hat v\| = 1$ (quedan sobre la esfera unitaria de $\mathbb{R}^{768}$). Entonces:

$$\cos\theta = \frac{\hat q \cdot \hat v}{1 \cdot 1} = \hat q \cdot \hat v$$

Es decir, **la similitud coseno es simplemente el producto punto**. El laboratorio (pestaña 1) muestra la norma real que sale de la red, ‖q‖ ≈ 16, y cómo al dividir queda en 1,000000. También suma $\hat q_1^2 + \dots + \hat q_{768}^2$ en vivo hasta llegar a 1: el teorema de Pitágoras en 768 dimensiones.

Una consecuencia de la ley de cosenos: con vectores unitarios, $\|q - v\|^2 = \|q\|^2 + \|v\|^2 - 2\,q\cdot v = 2 - 2\cos\theta$. Por eso ordenar por similitud coseno o por distancia euclidiana da **el mismo ranking** (pestaña 2).

### 3.5 Distancia coseno en ChromaDB

La colección se crea con `{"hnsw:space": "cosine"}`, y Chroma reporta una **distancia** (menor = más parecido):

$$d(q, v) = 1 - \cos\theta$$

Buscar los platos más parecidos equivale a encontrar los **k vecinos más cercanos** (k-NN) de $q$:

$$\text{top-}k = \underset{i}{\operatorname{arg\,min}_k}\; d(q, v_i)$$

En [app/vector_store.py](app/vector_store.py), la función `search` recalcula el coseno a mano con numpy y lo compara con lo que devuelve Chroma. Los tests (`tests/test_vector_math.py`) verifican que $\cos_{\text{numpy}} = 1 - d_{\text{Chroma}}$ con un error menor a $10^{-4}$.

### 3.6 La base de datos es una matriz: búsqueda exacta

Si apilamos los 41 vectores del menú como filas, obtenemos $X \in \mathbb{R}^{41 \times 768}$. Buscar es **una sola multiplicación matriz·vector**:

$$s = X q \in \mathbb{R}^{41}, \qquad s_i = x_i \cdot q = \cos\theta_i$$

Después se ordena $s$ y se toman los $k$ mayores. Esto cuesta $41 \times 768 = 31.488$ multiplicaciones. La pestaña 3 recorre la matriz fila por fila con contadores y luego anima el ordenamiento. Al final comprueba que el top-5 obtenido así es idéntico al que devolvió ChromaDB.

En esa matriz se ven **rayas verticales**: dimensiones donde todos los platos tienen casi el mismo valor. Son la causa de la anisotropía de la sección 3.3.

### 3.7 HNSW: cómo busca ChromaDB sin revisar todo

Con un millón de vectores, $Xq$ serían 768 millones de multiplicaciones por consulta. Chroma usa **HNSW** (*Hierarchical Navigable Small World*):

- Cada vector es un nodo conectado con sus $M$ vecinos más cercanos.
- Cada nodo recibe un nivel aleatorio $L = \lfloor -\ln(U)\, m_L \rfloor$, así que hay pocos nodos arriba y todos abajo.
- La búsqueda entra por la capa superior y salta de forma **voraz** al vecino más parecido a $q$. Cuando ningún vecino mejora, baja una capa. En la capa 0 mantiene una lista de los $ef$ mejores candidatos.

El costo es cercano a $O(\log N)$ en vez de $O(N)$, a cambio de que el resultado sea **aproximado**.

[app/hnsw_sim.py](app/hnsw_sim.py) construye un HNSW educativo con los mismos 41 vectores ($M = 4$, $ef = 5$) y registra cada paso para animarlo (pestaña 4). Ese grafo es nuestro; Chroma usa la misma idea con su propio grafo interno. Medido con 10 consultas de prueba:

- calcula unas **30 distancias en vez de 41**;
- acierta el primer resultado **10 de 10** veces;
- recupera el **94 %** de los 5 vecinos exactos (*recall*).

Con 41 platos el ahorro es pequeño; con millones de vectores es enorme.

### 3.8 Filtros por metadata

Además del vector, cada plato guarda metadata (`categoria`, `vegetariano`, `precio`). La búsqueda puede combinar similitud semántica con filtros exactos, por ejemplo "bebidas de máximo $7.000":

```python
col.query(query_embeddings=[q], n_results=5,
          where={"$and": [{"categoria": "bebidas"}, {"precio": {"$lte": 7000}}]})
```

### 3.9 PCA: ver ℝ⁷⁶⁸ en 3D

No podemos dibujar 768 dimensiones, así que el mapa 3D del laboratorio (pestaña 5, al estilo del Embedding Projector) usa **Análisis de Componentes Principales (PCA)**, una proyección lineal:

1. Matriz de datos $X \in \mathbb{R}^{41 \times 768}$ (una fila por plato).
2. Se centra: $X_c = X - \mu$, donde $\mu$ es el vector promedio.
3. Descomposición en valores singulares (SVD): $X_c = U \Sigma V^T$.
4. Las dos primeras filas de $V^T$ ($w_1, w_2$, ortonormales) son las direcciones de **máxima varianza**. Equivalen a los **vectores propios** de la matriz de covarianza $C = X_c^T X_c / (n-1)$ con mayores valores propios.
5. Cada vector se proyecta: $(x, y) = \big((v - \mu)\cdot w_1,\; (v - \mu)\cdot w_2\big)$.

La **varianza explicada** de cada componente es $\sigma_i^2 / \sum_j \sigma_j^2$. Con este menú, las tres primeras componentes explican cerca del 23 %, así que el dibujo es una "sombra" aproximada. Girar el mapa en pantalla es multiplicar cada punto por matrices de rotación 3×3. El ranking real siempre se calcula en las 768 dimensiones. Aun así, se ve que los platos de la misma categoría tienden a agruparse. Código: [app/vector_math.py](app/vector_math.py).

### 3.10 Números complejos: el mismo ángulo

En el plano PCA cada punto $(x, y)$ se puede ver como un número complejo $z = x + iy = r\,e^{i\varphi}$. Por la **fórmula de Euler**, $e^{i\theta} = \cos\theta + i\,\mathrm{sen}\,\theta$, y al multiplicar complejos **se multiplican las longitudes y se suman los ángulos**.

Para la consulta $z_q = a + bi$ y un plato $z_v = c + di$:

$$\bar z_q \cdot z_v = (ac + bd) + (ad - bc)\,i = |z_q||z_v|\,e^{i(\varphi_v - \varphi_q)}$$

- La **parte real** $ac + bd$ es el producto punto de $(a, b)$ y $(c, d)$.
- La **parte imaginaria** $ad - bc$ es el determinante (el área con signo).
- Si normalizamos ($\hat z = z/|z|$, igual que $q/\|q\|$), queda exactamente $e^{i\theta}$, y multiplicar por $e^{i\theta}$ **rota** $\hat z_q$ hasta $\hat z_v$.

En $\mathbb{C}^n$ el producto interno es $\langle u, v\rangle = \sum \bar u_k v_k$. El conjugado hace que $\langle u, u\rangle = \sum |u_k|^2$ sea real y positivo. Como los embeddings son reales, vuelve a ser el producto punto de siempre.

La pestaña 6 anima la normalización, el conjugado, el producto y la rotación. El ángulo 2D es el de la "sombra" PCA, no el ángulo real en $\mathbb{R}^{768}$.

---

## 4. Estructura del proyecto

```
data/menu.json        41 platos colombianos (sopas, platos fuertes, entradas, bebidas, postres)
app/embeddings.py     texto → vector normalizado (sentence-transformers)
app/vector_store.py   ChromaDB: indexación del menú, búsqueda semántica y filtros
app/vector_math.py    coseno, norma y PCA hechos a mano con numpy
app/hnsw_sim.py       simulación educativa de HNSW (grafo por capas + traza de la búsqueda)
app/agent.py          Gemini + function calling (el RAG)
app/cart.py           carrito por sesión
app/orders.py         pedidos en SQLite (data/pedidos.sqlite3)
app/main.py           API (FastAPI)
static/chat.*         chat estilo WhatsApp
static/lab.*          Laboratorio vectorial (6 secciones animadas, SVG/canvas sin librerías)
static/cocina.*       panel de cocina
tests/                pruebas automáticas (pytest)
```

---

## 5. Guion sugerido para la demo

1. Abrir el chat y, en otra pestaña, `/cocina`. Si quieres ir más despacio al explicar, pon la velocidad en **🐢 Lento**.
2. Escribir en el chat **"algo calientito con pollo"**. Gemini decide buscar y el laboratorio se activa solo:
   - **1 · Vector:** la frase se vuelve un vector de 768 números con ‖q‖ ≈ 16, se divide entre su norma y Σq̂ᵢ² llega a 1.
   - **2 · Producto punto:** q·v término a término, el ángulo θ, la proyección y que 1 − cos θ es la distancia que dio Chroma.
   - **3 · Matriz X·q:** las 41 filas multiplicadas por q, 31.488 multiplicaciones, el ordenamiento y el top-5 igual al de Chroma.
   - **4 · HNSW:** el recorrido por capas, cuántas distancias se ahorran y el recall frente a la búsqueda exacta.
   - **5 · Mapa 3D:** la nube de platos girando, tu frase como rombo blanco unido a sus 5 vecinos; toca cualquier plato (por ejemplo, la limonada de coco) para ver los suyos.
   - **6 · Complejos:** z̄_q·z_v = e^{iθ} y la rotación.
3. **"¿Tienen sushi?"**: no hay ningún plato realmente parecido y la IA responde que no lo tienen y sugiere mariscos.
4. **"Agrega dos ajiacos y una limonada de coco"**: el carrito se actualiza.
5. **"Confirmo, a nombre de Ana"**: llega el ticket, el pedido aparece en la cocina y se avanza su estado.
6. Sin internet o sin cuota de Gemini, el buscador del laboratorio ("Busca directo en ChromaDB") sigue mostrando toda la matemática.
