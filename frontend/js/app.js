/* ============================================================
   MindWell – Stress Activity Recommender — Frontend Logic
   ============================================================ */

const API = "http://localhost:5000/api";

// ---- State ----
const state = {
  users:       [],
  currentUser: null,
  activeTab:   "hybrid",
  evalChart:   null,
};

// ---- Wellness Category Emojis ----
const CATEGORY_EMOJI = {
  "Mindfulness": "🧘",
  "Physical":    "💪",
  "Creative":    "🎨",
  "Social":      "👥",
  "Rest":        "😴",
  "Cognitive":   "🧠",
  "Lifestyle":   "🌿",
};

// ---- Individual Activity Emojis ----
const ACTIVITY_EMOJI = {
  "Guided Body Scan Meditation":    "🧘",
  "Mindful Breathing Exercise":     "🌬️",
  "Progressive Muscle Relaxation":  "💆",
  "Loving-Kindness Meditation":     "💝",
  "Mindfulness App Session":        "📱",
  "4-7-8 Breathing Technique":      "🫁",
  "Grounding 5-4-3-2-1 Technique":  "🌱",
  "Morning Mindfulness Routine":    "🌅",
  "Mindful Walking":                "🚶",
  "Zen Coloring / Mandala":         "🎨",
  "Yoga Nidra (Sleep Yoga)":        "🌙",
  "Visualization Meditation":       "✨",
  "Yoga (Beginner)":                "🧘",
  "Yoga (Intermediate)":            "🧘",
  "Morning Walk / Jog":             "🏃",
  "Stretching Routine":             "🤸",
  "Dance Exercise / Zumba":         "💃",
  "Swimming":                       "🏊",
  "Cycling Outdoors":               "🚴",
  "HIIT Workout":                   "⚡",
  "Nature Hike":                    "🥾",
  "Tai Chi / Qigong":               "☯️",
  "Gratitude Journaling":           "📔",
  "Art Therapy":                    "🖌️",
  "Music Listening (Playlist)":     "🎵",
  "Creative Writing":               "✍️",
  "Photography Walk":               "📷",
  "DIY Craft Project":              "🔨",
  "Playing Musical Instrument":     "🎸",
  "Cooking a New Recipe":           "🍳",
  "Call a Friend / Family":         "📞",
  "Group Support Session":          "🤝",
  "Volunteer Community Work":       "❤️",
  "Family Game Night":              "🎲",
  "Join a Club or Group":           "🏅",
  "Coffee Chat with Colleague":     "☕",
  "Attend a Social Event":          "🎉",
  "Online Community Forum":         "💬",
  "Power Nap (20 min)":             "💤",
  "Digital Detox Hour":             "📵",
  "Reading Fiction":                "📚",
  "Hot Bath / Shower Relaxation":   "🛁",
  "Forest Bathing":                 "🌲",
  "Aromatherapy Session":           "🕯️",
  "Gentle Stretching Before Bed":   "🌙",
  "Sleep Hygiene Routine":          "😴",
  "CBT Journaling":                 "📝",
  "Problem-Solving Workshop":       "🔑",
  "Time Management Planning":       "📅",
  "Positive Affirmations":          "💪",
  "Worry Time Scheduling":          "⏰",
  "Thought Record Exercise":        "🗂️",
  "Values Clarification":           "🧭",
  "Behavioral Activation":          "✅",
  "Goal Setting Session":           "🎯",
  "Self-Compassion Exercise":       "🤗",
  "Healthy Meal Prep":              "🥗",
  "Herbal Tea Ritual":              "🍵",
  "Sunlight Exposure Walk":         "☀️",
  "Free-Form Journaling":           "📓",
};

// ---- Severity → colour mapping (matches DASS-21 bands) ----
const SEVERITY_COLOR = {
  "Normal":           { bg: "#f0fdf4", color: "#15803d" },
  "Mild":             { bg: "#fef9c3", color: "#854d0e" },
  "Moderate":         { bg: "#fff7ed", color: "#c2410c" },
  "Severe":           { bg: "#fef2f2", color: "#b91c1c" },
  "Extremely Severe": { bg: "#fdf4ff", color: "#86198f" },
};

const ALGO_COLORS = {
  "MindMatch Hybrid":   "#22c55e",
  "SVD-CF (Surprise)":  "#14b8a6",
  "Content-Based":      "#f97316",
  "User-Based CF":      "#3b82f6",
  "Item-Based CF":      "#8b5cf6",
  "Neural CF":          "#ec4899",
};

const COMPARE_MODEL_COLORS = {
  "Hybrid":        "#22c55e",
  "SVD-CF":        "#14b8a6",
  "Content-Based": "#f97316",
  "User-CF":       "#3b82f6",
  "Item-CF":       "#8b5cf6",
  "Neural CF":     "#ec4899",
};

function emoji(name, category) {
  return ACTIVITY_EMOJI[name] || CATEGORY_EMOJI[category] || "🧘";
}

// ---- Fetch helpers ----
async function apiFetch(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function showToast(msg) {
  const container = document.getElementById("toastContainer");
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

// ---- Stars ----
function stars(rating) {
  const full = Math.floor(rating);
  const half = rating % 1 >= 0.5 ? 1 : 0;
  return "★".repeat(full) + (half ? "½" : "") + "☆".repeat(5 - full - half);
}

// ---- Initialise ----
async function init() {
  try {
    const [usersData, statsData] = await Promise.all([
      apiFetch("/users"),
      apiFetch("/stats"),
    ]);
    state.users = usersData;
    renderStats(statsData);
    populateUserSelect(usersData);

    const saved = localStorage.getItem("mw_user");
    const defaultUser = saved && usersData.find(u => u.user_id === saved)
      ? saved
      : usersData[0].user_id;
    document.getElementById("userSelect").value = defaultUser;
    await selectUser(defaultUser);
  } catch (e) {
    showToast("Cannot reach backend. Start the Flask server first.");
    console.error(e);
  }
}

function populateUserSelect(users) {
  const sel = document.getElementById("userSelect");
  sel.innerHTML = users.map(u =>
    `<option value="${u.user_id}">${u.name} (${u.user_id})</option>`
  ).join("");
  sel.addEventListener("change", e => selectUser(e.target.value));
}

function renderStats(stats) {
  document.getElementById("statUsers").textContent    = stats.total_users.toLocaleString();
  document.getElementById("statProducts").textContent = stats.total_products.toLocaleString();
  document.getElementById("statRatings").textContent  = stats.total_ratings.toLocaleString();
  document.getElementById("statSparsity").textContent = stats.sparsity + "%";
}

async function selectUser(userId) {
  state.currentUser = state.users.find(u => u.user_id === userId);
  localStorage.setItem("mw_user", userId);
  _compareData   = null;   // invalidate comparison cache on user change
  _compareUserId = null;
  renderUserProfile(state.currentUser);
  await Promise.all([
    loadHistory(userId),
    state.activeTab === "compare"
      ? loadComparison()
      : loadRecommendations(state.activeTab, userId),
  ]);
}

// ---- User Profile with DASS-21 scores ----
function renderUserProfile(user) {
  document.getElementById("profileAvatar").textContent = user.name[0];
  document.getElementById("profileName").textContent   = user.name;
  document.getElementById("profileMeta").textContent   =
    `${user.user_id} · ${user.age_group} · ${user.occupation}`;

  // DASS-21 score panel
  const dass = document.getElementById("profileDass");
  const dims = [
    { key: "dass_depression", sev: user.depression_severity, label: "Depression" },
    { key: "dass_anxiety",    sev: user.anxiety_severity,    label: "Anxiety"    },
    { key: "dass_stress",     sev: user.stress_severity,     label: "Stress"     },
  ];
  dass.innerHTML = dims.map(d => {
    const sc = SEVERITY_COLOR[d.sev] || { bg: "#f8fafc", color: "#64748b" };
    return `
      <div class="dass-score" style="background:${sc.bg};border-color:${sc.color}40;">
        <span class="dass-score-label">${d.label}</span>
        <span class="dass-score-value" style="color:${sc.color}">${user[d.key]}</span>
        <span class="dass-score-max">/42 · ${d.sev}</span>
      </div>
    `;
  }).join("");

  const tags = document.getElementById("profileTags");
  tags.innerHTML = `
    <span class="tag tag-occ">${user.occupation}</span>
    <span class="tag tag-age">${user.age_group}</span>
  `;
  document.getElementById("headerBadge").textContent = `🧘 ${user.name}`;
}

// ---- Activity History ----
async function loadHistory(userId) {
  const list = document.getElementById("historyList");
  list.innerHTML = `<div class="loading"><div class="spinner"></div></div>`;
  try {
    const data = await apiFetch(`/user/${userId}/history`);
    if (!data.length) {
      list.innerHTML = `<div class="empty-state">No activity history yet</div>`;
      return;
    }
    list.innerHTML = data.slice(0, 20).map(item => `
      <div class="history-item">
        <span class="history-emoji">${emoji(item.name, item.category)}</span>
        <div class="history-info">
          <div class="history-name">${item.name}</div>
          <div class="history-cat">${item.category}</div>
        </div>
        <span class="history-stars">${stars(item.rating)}</span>
      </div>
    `).join("");
  } catch (e) {
    list.innerHTML = `<div class="empty-state">Failed to load history</div>`;
  }
}

// ---- Tabs ----
function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeTab = tab;

      const meta = TAB_META[tab] || {};
      document.getElementById("algoLabel").textContent = meta.label || "";
      document.getElementById("algoDesc").textContent  = meta.desc  || "";

      if (tab === "evaluation") {
        await loadEvaluation();
      } else if (tab === "condition") {
        showConditionSection();
      } else if (tab === "fairness") {
        await loadFairness();
      } else if (tab === "compare") {
        await loadComparison();
      } else if (state.currentUser) {
        await loadRecommendations(tab, state.currentUser.user_id);
      }
    });
  });
}

const TAB_META = {
  "hybrid":     { label: "MindMatch Hybrid",       desc: "score = 0.5 × Content-Based + 0.5 × SVD Collaborative — the primary MindMatch model." },
  "svd-cf":     { label: "SVD-CF (Surprise)",       desc: "Latent-factor SVD from the Surprise library — collaborative filtering via user–activity interaction patterns." },
  "content":    { label: "Content-Based",           desc: "TF-IDF profile built from your rated activities, matched to new activities via cosine similarity." },
  "user-cf":    { label: "User-Based CF",           desc: "Finds users with similar stress profiles and surfaces activities they found helpful." },
  "item-cf":    { label: "Item-Based CF",           desc: "Recommends activities similar to ones you have already rated highly." },
  "neural-cf":  { label: "Neural CF (NeuMF)",       desc: "NeuMF: Generalized Matrix Factorization (GMF) + MLP branches trained on implicit feedback via negative sampling." },
  "popular":    { label: "Popularity-Based",        desc: "Most-engaged activities ranked by Bayesian average rating across all users." },
  "compare":    { label: "Model Comparison",        desc: "Side-by-side comparison of all 6 models — consensus table shows which activities multiple models agree on." },
  "evaluation": { label: "Evaluation Metrics",      desc: "Precision@5, Recall@5, NDCG@5, Coverage, Diversity per algorithm on a held-out test set." },
  "condition":  { label: "My Condition",            desc: "Enter your DASS scores (0–42 each) and get activities matched to your specific condition and severity." },
  "fairness":   { label: "Fairness Analysis",        desc: "Demographic Parity: measures whether NDCG@5 is equal across age groups and occupations. Disparate Impact ≥ 0.8 is considered fair." },
};

const TAB_ENDPOINT = {
  "hybrid":     uid => `/recommend/hybrid/${uid}`,
  "svd-cf":     uid => `/recommend/svd-cf/${uid}`,
  "content":    uid => `/recommend/content-based/${uid}`,
  "user-cf":    uid => `/recommend/user-cf/${uid}`,
  "item-cf":    uid => `/recommend/item-cf/${uid}`,
  "neural-cf":  uid => `/recommend/neural-cf/${uid}`,
  "popular":    _   => `/recommend/popular`,
};

async function loadRecommendations(tab, userId) {
  const meta = TAB_META[tab] || {};
  document.getElementById("algoLabel").textContent = meta.label || "";
  document.getElementById("algoDesc").textContent  = meta.desc  || "";

  const grid    = document.getElementById("productsGrid");
  const section = document.getElementById("recSection");
  const evalSec = document.getElementById("evalSection");

  section.style.display = "block";
  evalSec.style.display  = "none";
  document.getElementById("conditionSection").style.display = "none";
  document.getElementById("fairnessSection").style.display  = "none";
  document.getElementById("compareSection").style.display   = "none";
  grid.innerHTML = `<div class="loading"><div class="spinner"></div><span>Loading recommendations…</span></div>`;

  const endpoint = TAB_ENDPOINT[tab];
  if (!endpoint) return;

  try {
    const data = await apiFetch(endpoint(userId));
    const recs  = data.recommendations || [];
    document.getElementById("recCount").textContent = `${recs.length} activities`;
    grid.innerHTML = recs.length
      ? recs.map(renderProductCard).join("")
      : `<div class="empty-state"><div class="big-emoji">🤷</div><div>No recommendations found for this user.</div></div>`;
  } catch (e) {
    grid.innerHTML = `<div class="empty-state"><div class="big-emoji">⚠️</div><div>Failed to load. Is the backend running?</div></div>`;
  }
}

// ---- Activity Card ----
function renderProductCard(item) {
  const maxScore = 1.0;
  const pct = Math.min(100, Math.round((item.score / maxScore) * 100));

  const effClass = {
    "high":     "badge-eff-high",
    "moderate": "badge-eff-moderate",
    "low":      "badge-eff-low",
  }[item.effectiveness] || "badge-eff-low";

  const targetLabel = item.stress_target
    ? (item.stress_target === "all"
        ? "All conditions"
        : item.stress_target.charAt(0).toUpperCase() + item.stress_target.slice(1))
    : "";

  const durationBadge = item.duration_minutes
    ? `<span class="badge badge-duration">⏱ ${item.duration_minutes} min</span>`
    : "";

  const targetBadge = targetLabel
    ? `<span class="badge badge-target">🎯 ${targetLabel}</span>`
    : "";

  const effBadge = item.effectiveness
    ? `<span class="badge ${effClass}">${item.effectiveness} effectiveness</span>`
    : "";

  const ratingDisplay = item.avg_rating
    ? `<span class="score-value">${item.avg_rating}/5</span>`
    : `<span class="score-value">${item.score.toFixed(3)}</span>`;

  return `
    <div class="product-card" onclick="showSimilar('${item.product_id}','${item.name}','${item.category}')">
      <div class="product-emoji">${emoji(item.name, item.category)}</div>
      <div class="product-name">${item.name}</div>
      <div class="product-category">${item.category}</div>
      <div class="product-badges">
        ${durationBadge}${targetBadge}${effBadge}
      </div>
      <div class="product-reason">${item.reason}</div>
      <div class="product-score">
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:${pct}%"></div>
        </div>
        ${ratingDisplay}
      </div>
    </div>
  `;
}

// ---- Similar Activities Modal ----
async function showSimilar(productId, productName, productCategory) {
  try {
    const data = await apiFetch(`/similar/${productId}?top_n=5`);
    const cbItems = data.content_based  || [];
    const cfItems = data.collaborative  || [];
    if (!cbItems.length && !cfItems.length) { showToast("No similar activities found."); return; }

    const modal = document.createElement("div");
    modal.style.cssText = `
      position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:500;
      display:flex;align-items:center;justify-content:center;padding:20px;
    `;
    const box = document.createElement("div");
    box.style.cssText = `
      background:#fff;border-radius:16px;padding:24px;max-width:640px;width:100%;
      max-height:80vh;overflow-y:auto;box-shadow:0 20px 40px rgba(0,0,0,.2);
    `;
    const renderItems = items => items.map(p => `
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px;font-size:13px;display:flex;align-items:center;gap:8px;">
        <span style="font-size:22px;">${emoji(p.name, p.category)}</span>
        <div>
          <div style="font-weight:600;">${p.name}</div>
          <div style="color:#64748b;font-size:11px;">${p.category || ""} · similarity ${p.similarity.toFixed(2)}</div>
        </div>
      </div>
    `).join("");

    box.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <h3 style="font-size:16px;font-weight:700;">Similar to ${emoji(productName, productCategory)} ${productName}</h3>
        <button id="closeModal" style="border:none;background:none;font-size:22px;cursor:pointer;color:#64748b;">✕</button>
      </div>
      <div style="margin-bottom:16px;">
        <div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:8px;">Content-Based Similar</div>
        <div style="display:flex;flex-direction:column;gap:6px;">${renderItems(cbItems)}</div>
      </div>
      <div>
        <div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:8px;">Collaborative Similar</div>
        <div style="display:flex;flex-direction:column;gap:6px;">${renderItems(cfItems)}</div>
      </div>
    `;
    modal.appendChild(box);
    document.body.appendChild(modal);
    modal.addEventListener("click", e => { if (e.target === modal) modal.remove(); });
    document.getElementById("closeModal").addEventListener("click", () => modal.remove());
  } catch (e) {
    showToast("Failed to load similar activities.");
  }
}

// ---- Evaluation ----
async function loadEvaluation() {
  const section = document.getElementById("recSection");
  const evalSec = document.getElementById("evalSection");
  section.style.display = "none";
  evalSec.style.display  = "block";
  document.getElementById("conditionSection").style.display = "none";
  document.getElementById("fairnessSection").style.display  = "none";
  document.getElementById("compareSection").style.display   = "none";

  const container = document.getElementById("evalContainer");
  container.innerHTML = `<div class="loading"><div class="spinner"></div><span>Running evaluation… this may take ~10 seconds.</span></div>`;

  try {
    const data = await apiFetch("/evaluation");
    renderEvaluation(data);
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="big-emoji">⚠️</div><div>Evaluation failed. Check the backend.</div></div>`;
  }
}

function renderEvaluation(data) {
  const container = document.getElementById("evalContainer");
  const algos = Object.keys(data);

  const cardsHtml = algos.map(algo => {
    const m = data[algo];
    const color = ALGO_COLORS[algo] || "#64748b";
    const metrics = [
      { label: "Precision@5",  value: m.precision_at_k, max: 1 },
      { label: "Recall@5",     value: m.recall_at_k,    max: 1 },
      { label: "NDCG@5",       value: m.ndcg_at_k,      max: 1 },
      { label: "Coverage",     value: m.coverage,       max: 1 },
      { label: "Diversity",    value: m.diversity,      max: 1 },
    ];
    if (m.rmse != null) metrics.push({ label: "RMSE (↓ better)", value: m.rmse, max: 2, invert: true });
    if (m.mae  != null) metrics.push({ label: "MAE (↓ better)",  value: m.mae,  max: 2, invert: true });

    return `
      <div class="eval-card">
        <div class="eval-algo-name">
          <div class="algo-color-dot" style="background:${color}"></div>
          ${algo}
        </div>
        <div class="eval-metrics">
          ${metrics.map(metric => {
            const pct = metric.invert
              ? Math.max(0, 100 - (metric.value / metric.max) * 100)
              : Math.min(100, (metric.value / metric.max) * 100);
            return `
              <div class="eval-metric">
                <span class="eval-metric-label">${metric.label}</span>
                <div class="eval-metric-bar">
                  <div class="eval-metric-fill" style="width:${pct.toFixed(1)}%;background:${color}"></div>
                </div>
                <span class="eval-metric-value" style="color:${color}">${metric.value}</span>
              </div>
            `;
          }).join("")}
          <div style="font-size:11px;color:#94a3b8;margin-top:4px;">
            Evaluated on ${m.n_users_evaluated} users
          </div>
        </div>
      </div>
    `;
  }).join("");

  container.innerHTML = `
    <div class="eval-grid">${cardsHtml}</div>
    <div class="chart-card">
      <div class="chart-title">Algorithm Comparison — Precision / Recall / NDCG @10</div>
      <div class="chart-wrap">
        <canvas id="evalChart"></canvas>
      </div>
    </div>
  `;

  const ctx = document.getElementById("evalChart").getContext("2d");
  if (state.evalChart) state.evalChart.destroy();

  state.evalChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: algos,
      datasets: [
        {
          label: "Precision@5",
          data: algos.map(a => data[a].precision_at_k),
          backgroundColor: "#3b82f6cc",
          borderRadius: 4,
        },
        {
          label: "Recall@5",
          data: algos.map(a => data[a].recall_at_k),
          backgroundColor: "#22c55ecc",
          borderRadius: 4,
        },
        {
          label: "NDCG@5",
          data: algos.map(a => data[a].ndcg_at_k),
          backgroundColor: "#f97316cc",
          borderRadius: 4,
        },
        {
          label: "Coverage",
          data: algos.map(a => data[a].coverage || 0),
          backgroundColor: "#8b5cf6cc",
          borderRadius: 4,
        },
        {
          label: "Diversity",
          data: algos.map(a => data[a].diversity || 0),
          backgroundColor: "#14b8a6cc",
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          grid: { color: "#e2e8f0" },
          ticks: { font: { size: 11 } },
        },
        x: {
          grid: { display: false },
          ticks: { font: { size: 11 } },
        },
      },
    },
  });
}

// ---- My Condition tab ----

const SEVERITY_THRESHOLDS = {
  depression: [[10,"Mild"],[14,"Moderate"],[21,"Severe"],[28,"Extremely Severe"]],
  anxiety:    [[ 8,"Mild"],[10,"Moderate"],[15,"Severe"],[20,"Extremely Severe"]],
  stress:     [[15,"Mild"],[19,"Moderate"],[26,"Severe"],[34,"Extremely Severe"]],
};

// Scores extracted by NLP, shared between NLP path and recommendation call
const nlpState = { depression: 0, anxiety: 0, stress: 0 };

function getSeverity(score, dim) {
  let label = "Normal";
  for (const [threshold, lbl] of SEVERITY_THRESHOLDS[dim]) {
    if (score >= threshold) label = lbl;
  }
  return label;
}

function updateSevBadge(badgeEl, severity) {
  const sc = SEVERITY_COLOR[severity] || { bg: "#f8fafc", color: "#64748b" };
  badgeEl.textContent      = severity;
  badgeEl.dataset.sev      = severity;
  badgeEl.style.background = sc.bg;
  badgeEl.style.color      = sc.color;
}

function showConditionSection() {
  document.getElementById("recSection").style.display       = "none";
  document.getElementById("evalSection").style.display      = "none";
  document.getElementById("fairnessSection").style.display  = "none";
  document.getElementById("compareSection").style.display   = "none";
  document.getElementById("conditionSection").style.display = "block";
}

// ── Mode toggle ──────────────────────────────────────────────────────────────

function setupModeToggle() {
  const nlpBtn     = document.getElementById("modeNlpBtn");
  const slidersBtn = document.getElementById("modeSlidersBtn");
  const nlpPanel     = document.getElementById("nlpPanel");
  const slidersPanel = document.getElementById("slidersPanel");

  nlpBtn.addEventListener("click", () => {
    nlpBtn.classList.add("active");
    slidersBtn.classList.remove("active");
    nlpPanel.style.display     = "block";
    slidersPanel.style.display = "none";
    document.getElementById("conditionResults").style.display = "none";
  });

  slidersBtn.addEventListener("click", () => {
    slidersBtn.classList.add("active");
    nlpBtn.classList.remove("active");
    slidersPanel.style.display = "block";
    nlpPanel.style.display     = "none";
    document.getElementById("conditionResults").style.display = "none";
  });
}

// ── NLP path ─────────────────────────────────────────────────────────────────

function setupNlpPanel() {
  const textarea  = document.getElementById("nlpInput");
  const charCount = document.getElementById("charCount");

  textarea.addEventListener("input", () => {
    charCount.textContent = textarea.value.length;
  });

  document.getElementById("analyzeBtn").addEventListener("click", analyzeCondition);

  document.getElementById("getRecsFromNlp").addEventListener("click", () => {
    fetchRecommendations(nlpState.depression, nlpState.anxiety, nlpState.stress);
  });

  document.getElementById("resetNlp").addEventListener("click", () => {
    textarea.value                = "";
    charCount.textContent         = "0";
    document.getElementById("nlpResult").style.display          = "none";
    document.getElementById("conditionResults").style.display   = "none";
  });
}

async function analyzeCondition() {
  const text = document.getElementById("nlpInput").value.trim();
  if (!text) { showToast("Please describe how you are feeling first."); return; }

  const btn = document.getElementById("analyzeBtn");
  btn.textContent = "Analyzing…";
  btn.disabled    = true;

  try {
    const res = await fetch(`${API}/nlp/parse-condition`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderNlpResult(data);
  } catch (e) {
    showToast("Analysis failed — is the backend running?");
  } finally {
    btn.textContent = "Analyze My Condition";
    btn.disabled    = false;
  }
}

function renderNlpResult(data) {
  // Save for recommendation call
  nlpState.depression = data.depression || 0;
  nlpState.anxiety    = data.anxiety    || 0;
  nlpState.stress     = data.stress     || 0;

  // Explanation
  document.getElementById("nlpExplanation").textContent = data.explanation || "";

  // Symptom chips
  const syms = data.detected_symptoms || [];
  document.getElementById("nlpSymptoms").innerHTML = syms.length
    ? syms.map(s => `<span class="symptom-chip">${s}</span>`).join("")
    : `<span style="color:var(--muted);font-size:12px;">No specific signals detected</span>`;

  // Score cards
  const dims = [
    { key: "depression", label: "Depression", score: data.depression },
    { key: "anxiety",    label: "Anxiety",    score: data.anxiety    },
    { key: "stress",     label: "Stress",     score: data.stress     },
  ];
  document.getElementById("nlpScores").innerHTML = dims.map(d => {
    const sev = (data.severity && data.severity[d.key]) || getSeverity(d.score, d.key);
    const sc  = SEVERITY_COLOR[sev] || { bg: "#f8fafc", color: "#64748b" };
    return `
      <div class="nlp-score-card" style="background:${sc.bg};border-color:${sc.color}40;">
        <div class="nlp-score-label">${d.label}</div>
        <div class="nlp-score-value" style="color:${sc.color}">${d.score}</div>
        <div class="nlp-score-max">/42</div>
        <div class="nlp-score-sev" style="color:${sc.color}">${sev}</div>
      </div>
    `;
  }).join("");

  // Source badge
  const isGemini = data.source === "gemini";
  document.getElementById("nlpSourceBadge").innerHTML = isGemini
    ? `<span class="source-badge source-gemini">✨ Powered by Gemini AI</span>`
    : `<span class="source-badge source-keyword">⚙️ Keyword analysis (set GEMINI_API_KEY for AI)</span>`;

  document.getElementById("nlpResult").style.display = "block";
}

// ── Sliders path ─────────────────────────────────────────────────────────────

function setupConditionSliders() {
  const sliders = [
    { slider: "depSlider", valEl: "depVal", sevEl: "depSev", dim: "depression" },
    { slider: "anxSlider", valEl: "anxVal", sevEl: "anxSev", dim: "anxiety"    },
    { slider: "strSlider", valEl: "strVal", sevEl: "strSev", dim: "stress"     },
  ];

  sliders.forEach(({ slider, valEl, sevEl, dim }) => {
    const input    = document.getElementById(slider);
    const valSpan  = document.getElementById(valEl);
    const sevBadge = document.getElementById(sevEl);
    input.addEventListener("input", () => {
      const v = parseInt(input.value, 10);
      valSpan.textContent = v;
      updateSevBadge(sevBadge, getSeverity(v, dim));
    });
  });

  document.getElementById("getConditionRecs").addEventListener("click", () => {
    const dep = parseInt(document.getElementById("depSlider").value, 10);
    const anx = parseInt(document.getElementById("anxSlider").value, 10);
    const str = parseInt(document.getElementById("strSlider").value, 10);
    fetchRecommendations(dep, anx, str);
  });

  document.getElementById("resetCondition").addEventListener("click", () => {
    sliders.forEach(({ slider, valEl, sevEl }) => {
      document.getElementById(slider).value      = 0;
      document.getElementById(valEl).textContent = 0;
      updateSevBadge(document.getElementById(sevEl), "Normal");
    });
    document.getElementById("conditionResults").style.display = "none";
  });
}

// ── Shared recommendation fetcher ─────────────────────────────────────────────

const REC_PAGE_SIZE  = 5;
const REC_FETCH_SIZE = 20;

// Store last DASS scores so model-selector change can re-fetch without re-typing
const _lastDass = { depression: 0, anxiety: 0, stress: 0 };

function setupModelSelector() {
  document.getElementById("conditionModelSel").addEventListener("change", () => {
    if (document.getElementById("conditionResults").style.display !== "none") {
      fetchRecommendations(_lastDass.depression, _lastDass.anxiety, _lastDass.stress);
    }
  });
}

async function fetchRecommendations(depression, anxiety, stress) {
  // Remember for re-fetch on model change
  _lastDass.depression = depression;
  _lastDass.anxiety    = anxiety;
  _lastDass.stress     = stress;

  const model      = document.getElementById("conditionModelSel").value;
  const resultsDiv = document.getElementById("conditionResults");
  const grid       = document.getElementById("conditionGrid");
  const banner     = document.getElementById("nearestUserBanner");

  resultsDiv.style.display = "block";
  banner.style.display     = "none";
  grid.innerHTML = `<div class="loading"><div class="spinner"></div><span>Finding activities…</span></div>`;
  resultsDiv.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const res = await fetch(`${API}/recommend/by-condition`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ depression, anxiety, stress, top_n: REC_FETCH_SIZE, model }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Show nearest-user banner for ML models
    if (data.nearest_user) {
      const u = data.nearest_user;
      const sc = SEVERITY_COLOR[getSeverity(u.dass_depression, "depression")] || {};
      banner.style.display = "block";
      banner.innerHTML = `
        <span class="nu-icon">👤</span>
        <div class="nu-info">
          <span class="nu-label">Using nearest matching user:</span>
          <strong>${u.name}</strong>
          <span class="nu-meta">${u.age_group} · ${u.occupation}</span>
          <span class="nu-dass">D:${u.dass_depression} A:${u.dass_anxiety} S:${u.dass_stress}</span>
        </div>
        <span class="nu-dist" title="Euclidean distance on DASS scores">DASS distance: ${u.distance}</span>
      `;
    }

    renderConditionRecs(data.recommendations || []);
  } catch (e) {
    grid.innerHTML = `<div class="empty-state"><div class="big-emoji">⚠️</div><div>Failed to reach backend.</div></div>`;
  }
}

function renderConditionRecs(recs, showAll = false) {
  const grid      = document.getElementById("conditionGrid");
  const countEl   = document.getElementById("conditionCount");
  const visible   = showAll ? recs : recs.slice(0, REC_PAGE_SIZE);
  const remaining = recs.length - visible.length;

  countEl.textContent = `Showing ${visible.length} of ${recs.length} activities`;

  if (!recs.length) {
    grid.innerHTML = `<div class="empty-state"><div class="big-emoji">🤷</div><div>No activities found.</div></div>`;
    return;
  }

  const cards = visible.map(renderProductCard).join("");

  const showMoreBtn = (!showAll && remaining > 0)
    ? `<div class="show-more-wrap">
         <button class="show-more-btn" onclick="renderConditionRecs(window._lastRecs, true)">
           Show ${remaining} more activities
         </button>
       </div>`
    : "";

  // Store for the show-more callback
  window._lastRecs = recs;

  grid.innerHTML = cards + showMoreBtn;
}

// ---- Fairness Analysis ----

let _fairnessData = null;

async function loadFairness() {
  document.getElementById("recSection").style.display       = "none";
  document.getElementById("evalSection").style.display      = "none";
  document.getElementById("conditionSection").style.display = "none";
  document.getElementById("compareSection").style.display   = "none";
  document.getElementById("fairnessSection").style.display  = "block";

  if (_fairnessData) { renderFairness(_fairnessData); return; }

  const container = document.getElementById("fairnessContainer");
  container.innerHTML = `<div class="loading"><div class="spinner"></div><span>Running fairness evaluation… ~10 seconds</span></div>`;

  try {
    _fairnessData = await apiFetch("/fairness");
    renderFairness(_fairnessData);
    document.getElementById("fairnessModelSel").addEventListener("change", () => renderFairness(_fairnessData));
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="big-emoji">⚠️</div><div>Fairness evaluation failed. Check backend.</div></div>`;
  }
}

function renderFairness(data) {
  const model     = document.getElementById("fairnessModelSel").value;
  const modelData = data[model];
  const container = document.getElementById("fairnessContainer");

  if (!modelData) {
    container.innerHTML = `<div class="empty-state">No data for ${model}.</div>`;
    return;
  }

  const overallNDCG = modelData.overall_ndcg;
  const nUsers      = modelData.n_users_evaluated;

  container.innerHTML = `
    <div class="fairness-overview">
      <div class="fairness-kpi">
        <span class="fairness-kpi-value">${overallNDCG}</span>
        <span class="fairness-kpi-label">Overall NDCG@5</span>
      </div>
      <div class="fairness-kpi">
        <span class="fairness-kpi-value">${nUsers}</span>
        <span class="fairness-kpi-label">Users Evaluated</span>
      </div>
    </div>
    <div class="fairness-groups">
      ${renderGroupBlock("By Age Group",   modelData.age_group)}
      ${renderGroupBlock("By Occupation",  modelData.occupation)}
    </div>
    <div class="fairness-note">
      <strong>Disparate Impact ≥ 0.8</strong> is the standard fairness threshold (4/5ths rule).
      A gap of 0 means all groups receive equally good recommendations.
    </div>
  `;
}

function renderGroupBlock(title, groupData) {
  const scores  = groupData.scores || {};
  const gap     = groupData.ndcg_gap;
  const di      = groupData.disparate_impact;
  const isFair  = groupData.is_fair;

  const maxScore = Math.max(...Object.values(scores), 0.001);

  const bars = Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .map(([grp, ndcg]) => {
      const pct = Math.round((ndcg / maxScore) * 100);
      return `
        <div class="fairness-bar-row">
          <span class="fairness-bar-label">${grp}</span>
          <div class="fairness-bar-track">
            <div class="fairness-bar-fill" style="width:${pct}%"></div>
          </div>
          <span class="fairness-bar-value">${ndcg.toFixed(4)}</span>
        </div>
      `;
    }).join("");

  const diColor  = isFair ? "#15803d" : "#b91c1c";
  const diIcon   = isFair ? "✅" : "⚠️";
  const gapColor = gap <= 0.05 ? "#15803d" : gap <= 0.15 ? "#c2410c" : "#b91c1c";

  return `
    <div class="fairness-group-card">
      <div class="fairness-group-title">${title}</div>
      <div class="fairness-bars">${bars}</div>
      <div class="fairness-metrics-row">
        <div class="fairness-metric">
          <span class="fairness-metric-label">NDCG Gap</span>
          <span class="fairness-metric-value" style="color:${gapColor}">${gap.toFixed(4)}</span>
        </div>
        <div class="fairness-metric">
          <span class="fairness-metric-label">Disparate Impact ${diIcon}</span>
          <span class="fairness-metric-value" style="color:${diColor}">${di.toFixed(4)}</span>
        </div>
      </div>
    </div>
  `;
}

// ---- Model Comparison ----

let _compareData   = null;
let _compareUserId = null;

async function loadComparison() {
  document.getElementById("recSection").style.display       = "none";
  document.getElementById("evalSection").style.display      = "none";
  document.getElementById("conditionSection").style.display = "none";
  document.getElementById("fairnessSection").style.display  = "none";
  document.getElementById("compareSection").style.display   = "block";

  if (!state.currentUser) return;
  const userId = state.currentUser.user_id;

  // Re-fetch when user changes
  if (_compareUserId === userId && _compareData) {
    renderComparison(_compareData);
    return;
  }

  const container = document.getElementById("compareContainer");
  container.innerHTML = `<div class="loading"><div class="spinner"></div><span>Fetching all 6 model recommendations…</span></div>`;

  try {
    _compareData   = await apiFetch(`/compare-all/${userId}?top_n=10`);
    _compareUserId = userId;
    renderComparison(_compareData);
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="big-emoji">⚠️</div><div>Comparison failed. Is the backend running?</div></div>`;
  }
}

function hexToRgb(hex) {
  const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return r ? [parseInt(r[1], 16), parseInt(r[2], 16), parseInt(r[3], 16)] : [100, 116, 139];
}

function renderComparison(data) {
  const container  = document.getElementById("compareContainer");
  const allModels  = Object.keys(data.models);
  const consensus  = data.consensus  || [];
  const agreement  = data.agreement  || {};
  const topN       = data.top_n      || 10;

  const uniqueItems    = consensus.length;
  const fullConsensus  = consensus.filter(c => c.n_models === allModels.length).length;
  const avgOverlap     = uniqueItems
    ? (consensus.reduce((s, c) => s + c.n_models, 0) / uniqueItems).toFixed(1)
    : "0";

  // ── KPI row ──
  const overviewHtml = `
    <div class="compare-overview">
      <div class="compare-kpi">
        <span class="compare-kpi-value">${uniqueItems}</span>
        <span class="compare-kpi-label">Unique Activities</span>
      </div>
      <div class="compare-kpi">
        <span class="compare-kpi-value">${fullConsensus}</span>
        <span class="compare-kpi-label">Full Consensus (${allModels.length}/${allModels.length} models)</span>
      </div>
      <div class="compare-kpi">
        <span class="compare-kpi-value">${avgOverlap}</span>
        <span class="compare-kpi-label">Avg Models per Item</span>
      </div>
    </div>
  `;

  // ── Consensus table ──
  const legendHtml = allModels.map(m => `
    <span style="display:flex;align-items:center;gap:4px;font-size:11px;white-space:nowrap;">
      <span style="width:8px;height:8px;border-radius:50%;background:${COMPARE_MODEL_COLORS[m]||'#64748b'};display:inline-block;flex-shrink:0;"></span>
      ${m}
    </span>
  `).join("");

  const consensusRows = consensus.map(item => {
    const dots = allModels.map(m => {
      const on    = item.models.includes(m);
      const color = on ? (COMPARE_MODEL_COLORS[m] || "#64748b") : "#e2e8f0";
      const title = on ? `In ${m}` : `Not in ${m}`;
      return `<span title="${title}" style="display:inline-block;width:11px;height:11px;border-radius:50%;background:${color};flex-shrink:0;"></span>`;
    }).join("");

    const pct      = Math.round((item.n_models / allModels.length) * 100);
    const barColor = item.n_models === allModels.length ? "#22c55e"
                   : item.n_models >= 4                 ? "#14b8a6"
                   : item.n_models >= 2                 ? "#f97316"
                                                        : "#94a3b8";
    return `
      <tr class="compare-row">
        <td>
          <div class="compare-item-cell">
            <span class="compare-item-emoji">${emoji(item.name, item.category)}</span>
            <div>
              <div class="compare-item-name">${item.name}</div>
              <div class="compare-item-cat">${item.category}</div>
            </div>
          </div>
        </td>
        <td>
          <div class="compare-model-dots">${dots}</div>
          <div class="compare-consensus-bar-wrap">
            <div class="compare-consensus-bar" style="width:${pct}%;background:${barColor};"></div>
          </div>
          <span class="compare-n-models" style="color:${barColor}">${item.n_models}/${allModels.length} models</span>
        </td>
        <td class="compare-score-cell">${item.avg_score.toFixed(3)}</td>
      </tr>
    `;
  }).join("");

  // ── Agreement matrix ──
  const matrixHead = `<tr><th></th>${allModels.map(m =>
    `<th style="color:${COMPARE_MODEL_COLORS[m]||'#64748b'};font-size:11px;">${m}</th>`
  ).join("")}</tr>`;

  const matrixRows = allModels.map(la => {
    const [r, g, b] = hexToRgb(COMPARE_MODEL_COLORS[la] || "#64748b");
    const cells = allModels.map(lb => {
      const isdiag = la === lb;
      const count  = agreement[la]?.[lb] ?? 0;
      const alpha  = isdiag ? 0.18 : Math.min(0.65, (count / topN) * 0.65);
      const bg     = `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
      return `<td style="background:${bg};text-align:center;padding:8px 10px;font-size:13px;font-weight:${isdiag?700:500};">${isdiag ? "—" : count}</td>`;
    }).join("");
    return `<tr><th style="color:${COMPARE_MODEL_COLORS[la]||'#64748b'};font-size:11px;padding:8px 12px;">${la}</th>${cells}</tr>`;
  }).join("");

  container.innerHTML = `
    ${overviewHtml}

    <div class="compare-section-title">Consensus Recommendations</div>
    <p style="font-size:12px;color:var(--muted);margin-bottom:12px;">
      Sorted by how many models recommend each activity. Colored dots show model coverage.
    </p>
    <div class="compare-table-wrap">
      <table class="compare-table">
        <thead>
          <tr>
            <th style="min-width:220px;">Activity</th>
            <th>
              <div class="compare-legend">${legendHtml}</div>
              Model Agreement
            </th>
            <th style="white-space:nowrap;">Avg Score</th>
          </tr>
        </thead>
        <tbody>${consensusRows}</tbody>
      </table>
    </div>

    <div class="compare-section-title" style="margin-top:28px;">Agreement Matrix</div>
    <p style="font-size:12px;color:var(--muted);margin-bottom:12px;">
      Number of activities appearing in both models' top-${topN} lists. Darker = more overlap.
    </p>
    <div class="compare-table-wrap" style="overflow-x:auto;">
      <table class="compare-matrix">
        <thead>${matrixHead}</thead>
        <tbody>${matrixRows}</tbody>
      </table>
    </div>
  `;
}

// ---- Primary Navigation (3-section top nav) ----
function setupPrimaryNav() {
  const algoNav    = document.getElementById("algoNav");
  const analysisNav = document.getElementById("analysisNav");
  const infoBar    = document.getElementById("algoInfoBar");

  document.querySelectorAll(".nav-primary-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const section = btn.dataset.section;
      document.querySelectorAll(".nav-primary-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      if (section === "recommendations") {
        algoNav.style.display = "";
        analysisNav.style.display = "none";
        infoBar.style.display = "";
        (algoNav.querySelector(".tab-btn.active") || algoNav.querySelector(".tab-btn")).click();

      } else if (section === "condition") {
        algoNav.style.display = "none";
        analysisNav.style.display = "none";
        infoBar.style.display = "none";
        state.activeTab = "condition";
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        showConditionSection();
        const meta = TAB_META["condition"] || {};
        document.getElementById("algoLabel").textContent = meta.label || "";
        document.getElementById("algoDesc").textContent  = meta.desc  || "";

      } else if (section === "analysis") {
        algoNav.style.display = "none";
        analysisNav.style.display = "";
        infoBar.style.display = "";
        (analysisNav.querySelector(".tab-btn.active") || analysisNav.querySelector(".tab-btn")).click();
      }
    });
  });
}

// ---- Boot ----
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupModeToggle();
  setupNlpPanel();
  setupConditionSliders();
  setupModelSelector();
  setupPrimaryNav();
  init();
});
