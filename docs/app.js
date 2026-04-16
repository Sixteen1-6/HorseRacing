/* SharpLine Racing — v4 Frontend */

const REPO = "Sixteen1-6/HorseRacing";
const BRANCH = "main";
const CARD_DIR = "HorseRacing";
const EDGE_THRESHOLD = 1.10;
const POLL_INTERVAL = 5000;
const POLL_TIMEOUT = 120000;

const state = {
  pat: localStorage.getItem("sharpline_pat") || "",
  cards: [],
  currentCard: null,
  currentRace: null,
  races: {},
  edits: { scratches: new Set(), overrides: {}, jockeys: {}, added: [] },
  predictions: null,
};

/* ── GitHub API ── */

async function ghFetch(path, opts = {}) {
  const headers = { Accept: "application/vnd.github.v3+json" };
  if (state.pat) headers.Authorization = `token ${state.pat}`;
  const res = await fetch(`https://api.github.com/${path}`, { ...opts, headers });
  if (!res.ok) throw new Error(`GitHub ${res.status}: ${await res.text()}`);
  return res.json();
}

async function ghContents(path) {
  // Try raw URL first (no size limit, no auth needed)
  try {
    const rawUrl = `https://raw.githubusercontent.com/${REPO}/${BRANCH}/${path}`;
    const res = await fetch(rawUrl);
    if (res.ok) return await res.text();
  } catch (e) { /* fall through */ }
  // Fallback to API
  const data = await ghFetch(`repos/${REPO}/contents/${path}?ref=${BRANCH}`);
  return atob(data.content);
}

async function ghDispatch(payload) {
  if (!state.pat) { document.getElementById("patDialog").showModal(); return; }
  await fetch(`https://api.github.com/repos/${REPO}/dispatches`, {
    method: "POST",
    headers: { Authorization: `token ${state.pat}`, "Content-Type": "application/json" },
    body: JSON.stringify({ event_type: "predict", client_payload: payload }),
  });
}

/* ── CSV Parsing ── */

function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map(line => {
    const vals = parseCsvLine(line);
    const obj = {};
    headers.forEach((h, i) => obj[h] = (vals[i] || "").trim());
    return obj;
  });
}

function parseCsvLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { inQuotes = !inQuotes; continue; }
    if (ch === ',' && !inQuotes) { result.push(current.trim()); current = ""; continue; }
    current += ch;
  }
  result.push(current.trim());
  return result;
}

/* ── Card Discovery ── */

async function discoverCards() {
  // Hardcode known card — avoids API rate limits
  state.cards = [{ name: "KEE 2026-04-16", path: "test_data.csv" }];
}

/* ── Load Card ── */

async function loadCard(path) {
  setStatus("running", "Loading...");
  try {
    console.log("Loading card:", path);
    const csv = await ghContents(path);
    console.log("CSV loaded, length:", csv.length, "first 100:", csv.substring(0, 100));
    const rows = parseCSV(csv);
    console.log("Parsed rows:", rows.length);
    state.races = {};
    state.edits = { scratches: new Set(), overrides: {}, jockeys: {}, added: [] };

    rows.forEach(row => {
      const rn = row.race_number;
      if (!state.races[rn]) state.races[rn] = { entries: [], meta: {} };
      state.races[rn].entries.push(row);
      state.races[rn].meta = {
        race_type: row.race_type || "",
        surface: row.surface || "",
        distance: row.distance || "",
        distance_unit: row.distance_unit || "",
        purse: row.purse || "",
        track_condition: row.track_condition || "",
      };
    });

    state.currentCard = path;
    renderRaceTabs();
    const firstRace = Object.keys(state.races).sort((a, b) => a - b)[0];
    if (firstRace) selectRace(firstRace);

    // Try loading existing predictions
    await loadPredictions();
    setStatus("done", "Loaded");
  } catch (e) {
    setStatus("error", e.message);
    console.error("loadCard error:", e);
    document.getElementById("raceTitle").textContent = "Error: " + e.message;
  }
}

/* ── Load Predictions ── */

async function loadPredictions() {
  try {
    const csv = await ghContents("race_predictions/all_race_predictions.csv");
    const rows = parseCSV(csv);
    state.predictions = {};
    rows.forEach(row => {
      const rn = row.race_number;
      if (!state.predictions[rn]) state.predictions[rn] = [];
      state.predictions[rn].push(row);
    });
    // Also load exotics
    try {
      const exCsv = await ghContents("race_predictions/all_race_predictions_exotics.csv");
      state.exotics = {};
      parseCSV(exCsv).forEach(row => {
        const rn = row.race_number;
        if (!state.exotics[rn]) state.exotics[rn] = [];
        state.exotics[rn].push(row);
      });
    } catch (e) { state.exotics = {}; }
  } catch (e) {
    state.predictions = null;
    state.exotics = {};
  }
}

/* ── Render Race Tabs ── */

function renderRaceTabs() {
  const nav = document.getElementById("raceTabs");
  nav.innerHTML = "";
  Object.keys(state.races).sort((a, b) => a - b).forEach(rn => {
    const tab = document.createElement("div");
    tab.className = "race-tab" + (rn === state.currentRace ? " active" : "");
    tab.textContent = `R${rn}`;
    // Check if this race has value bets
    if (state.predictions && state.predictions[rn]) {
      const hasEdge = state.predictions[rn].some(p => parseFloat(p.edge) >= EDGE_THRESHOLD);
      if (hasEdge) tab.classList.add("has-edge");
    }
    tab.onclick = () => selectRace(rn);
    nav.appendChild(tab);
  });
}

/* ── Select Race ── */

function selectRace(rn) {
  state.currentRace = rn;
  renderRaceTabs();
  renderEntries();
  renderResults();
  renderExotics();
  renderBets();
}

/* ── Render Entries ── */

function renderEntries() {
  const rn = state.currentRace;
  const race = state.races[rn];
  if (!race) return;

  // Header
  const meta = race.meta;
  document.getElementById("raceTitle").textContent = `Race ${rn} - ${meta.race_type}`;
  document.getElementById("raceMeta").textContent =
    `${meta.distance} ${meta.distance_unit === 'F' ? 'Furlongs' : meta.distance_unit} | ${meta.surface === 'D' ? 'Dirt' : 'Turf'} | ${meta.track_condition} | $${Number(meta.purse).toLocaleString()}`;

  // Pace narrative from predictions
  const paceEl = document.getElementById("paceNarrative");
  if (state.predictions && state.predictions[rn]) {
    const pn = state.predictions[rn].find(p => p.pace_narrative && p.pace_narrative !== "" && p.pace_narrative !== "nan");
    paceEl.textContent = pn ? pn.pace_narrative : "";
  } else { paceEl.textContent = ""; }

  // Entries
  const tbody = document.getElementById("entriesBody");
  tbody.innerHTML = "";
  race.entries.forEach(entry => {
    const key = `${rn}::${entry.horse_name}`;
    const scratched = state.edits.scratches.has(key);
    const tr = document.createElement("tr");
    if (scratched) tr.classList.add("scratched");

    let style = entry.running_style || state.getPredStyle(rn, entry.horse_name) || "U";
    if (style === "nan" || style === "NaN" || style === "") style = "U";
    const odds = state.edits.overrides[key] ?? entry.dollar_odds ?? "";
    const jockey = state.edits.jockeys[key] ?? entry.jockey ?? "";

    tr.innerHTML = `
      <td class="col-scratch"><input type="checkbox" ${scratched ? "checked" : ""} data-key="${key}"></td>
      <td class="col-pp">${entry.post_position || ""}</td>
      <td class="col-horse">${entry.horse_name}</td>
      <td><input class="jockey-input" value="${jockey}" data-key="${key}" data-field="jockey"></td>
      <td>${entry.trainer || ""}</td>
      <td class="col-style"><span class="style-badge style-${style}">${style}</span></td>
      <td class="col-odds"><input class="odds-input" type="number" value="${odds}" min="0" step="0.5" data-key="${key}" data-field="odds"></td>
    `;
    tbody.appendChild(tr);
  });

  // Event listeners
  tbody.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.onchange = () => {
      const key = cb.dataset.key;
      if (cb.checked) state.edits.scratches.add(key);
      else state.edits.scratches.delete(key);
      renderEntries();
    };
  });
  tbody.querySelectorAll('.odds-input').forEach(input => {
    input.onchange = () => { state.edits.overrides[input.dataset.key] = input.value; };
  });
  tbody.querySelectorAll('.jockey-input').forEach(input => {
    input.onchange = () => { state.edits.jockeys[input.dataset.key] = input.value; };
  });
}

// Helper to get running style from predictions
state.getPredStyle = function(rn, horseName) {
  if (!state.predictions || !state.predictions[rn]) return null;
  const p = state.predictions[rn].find(r => r.horse_name === horseName);
  return p ? p.running_style : null;
};

/* ── Render Results ── */

function renderResults() {
  const rn = state.currentRace;
  const section = document.getElementById("resultsSection");
  const tbody = document.getElementById("resultsBody");
  const signal = document.getElementById("raceSignal");

  if (!state.predictions || !state.predictions[rn]) {
    section.classList.add("hidden");
    return;
  }

  section.classList.remove("hidden");
  const preds = state.predictions[rn]
    .filter(p => !state.edits.scratches.has(`${rn}::${p.horse_name}`))
    .sort((a, b) => a.predicted_rank - b.predicted_rank);

  const hasEdge = preds.some(p => parseFloat(p.edge) >= EDGE_THRESHOLD);
  signal.className = `signal ${hasEdge ? "edge" : "pass"}`;
  signal.textContent = hasEdge ? "EDGE" : "PASS";

  tbody.innerHTML = "";
  preds.forEach(p => {
    const isValue = parseFloat(p.edge) >= EDGE_THRESHOLD;
    const tr = document.createElement("tr");
    if (isValue) tr.classList.add("value-bet");
    const kelly = parseFloat(p.kelly_bet) || 0;
    tr.innerHTML = `
      <td>${p.predicted_rank}</td>
      <td>${p.horse_name} ${isValue ? '<span class="edge-tag">VALUE</span>' : ''}</td>
      <td>${(parseFloat(p.win_probability) * 100).toFixed(1)}%</td>
      <td>${(parseFloat(p.top3_probability) * 100).toFixed(1)}%</td>
      <td>${p.odds || p.dollar_odds || ""}</td>
      <td>${parseFloat(p.edge) >= 1.0 ? parseFloat(p.edge).toFixed(2) + 'x' : ''}</td>
      <td>${kelly > 0 ? '$' + kelly.toFixed(0) : ''}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* ── Render Exotics ── */

function renderExotics() {
  const rn = state.currentRace;
  const section = document.getElementById("exoticsSection");

  if (!state.exotics || !state.exotics[rn]) {
    section.classList.add("hidden");
    return;
  }

  section.classList.remove("hidden");
  const exotics = state.exotics[rn];

  const exactas = exotics.filter(e => e.type === "EXACTA");
  const trifectas = exotics.filter(e => e.type === "TRIFECTA");

  document.getElementById("exactaList").innerHTML = exactas.map(e =>
    `<div class="exotic-combo"><span>${e.combo}</span><span class="exotic-fair">Fair $${parseFloat(e.fair_odds).toFixed(0)}</span></div>`
  ).join("");

  document.getElementById("trifectaList").innerHTML = trifectas.map(e =>
    `<div class="exotic-combo"><span>${e.combo}</span><span class="exotic-fair">Fair $${parseFloat(e.fair_odds).toFixed(0)}</span></div>`
  ).join("");
}

/* ── Render Bets ── */

function renderBets() {
  const rn = state.currentRace;
  const section = document.getElementById("betsSection");
  const list = document.getElementById("betsList");
  const total = document.getElementById("betsTotal");

  if (!state.predictions || !state.predictions[rn]) {
    section.classList.add("hidden");
    return;
  }

  const bets = state.predictions[rn]
    .filter(p => parseFloat(p.edge) >= EDGE_THRESHOLD && !state.edits.scratches.has(`${rn}::${p.horse_name}`))
    .sort((a, b) => parseFloat(b.edge) - parseFloat(a.edge));

  if (bets.length === 0) {
    section.classList.add("hidden");
    return;
  }

  section.classList.remove("hidden");
  let totalKelly = 0;
  list.innerHTML = bets.map(b => {
    const kelly = parseFloat(b.kelly_bet) || 0;
    totalKelly += kelly;
    return `<div class="bet-card">
      <div>
        <div class="bet-horse">${b.horse_name}</div>
        <div class="bet-details">Win: ${(parseFloat(b.win_probability)*100).toFixed(1)}% vs Market: ${(parseFloat(b.implied_prob)*100).toFixed(1)}% | Edge: ${parseFloat(b.edge).toFixed(2)}x | Odds: ${b.odds}-1</div>
      </div>
      <div class="bet-amount">$${kelly.toFixed(0)}</div>
    </div>`;
  }).join("");
  total.textContent = `Total: $${totalKelly.toFixed(0)} on $10,000 bankroll`;
}

/* ── Run Model ── */

async function runModel() {
  if (!state.pat) { document.getElementById("patDialog").showModal(); return; }

  const runId = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  setStatus("running", "Dispatching...");

  try {
    const payload = {
      run_id: runId,
      track_code: "KEE",
      card_date: "2026-04-16",
      race_num: state.currentRace || "",
      card_path: "test_data.csv",
      edits: {
        scratches: [...state.edits.scratches],
        overrides: state.edits.overrides,
        jockeys: state.edits.jockeys,
        added: state.edits.added,
      },
    };

    console.log("Dispatching with payload:", JSON.stringify(payload, null, 2));

    const res = await fetch(`https://api.github.com/repos/${REPO}/dispatches`, {
      method: "POST",
      headers: {
        Authorization: `token ${state.pat}`,
        "Content-Type": "application/json",
        Accept: "application/vnd.github.v3+json",
      },
      body: JSON.stringify({ event_type: "predict", client_payload: payload }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Dispatch failed: ${res.status} ${text}`);
    }

    console.log("Dispatch sent, polling for results...");
    setStatus("running", "Running...");
    pollForResults(runId);
  } catch (e) {
    setStatus("error", e.message);
    console.error("runModel error:", e);
  }
}

async function pollForResults(runId) {
  const start = Date.now();
  const poll = async () => {
    if (Date.now() - start > POLL_TIMEOUT) {
      setStatus("error", "Timeout");
      return;
    }
    try {
      const pred = await ghFetch(`repos/${REPO}/contents/predictions/run_${runId}.json?ref=${BRANCH}`);
      const data = JSON.parse(atob(pred.content));
      state.predictions = {};
      for (const [key, race] of Object.entries(data.races)) {
        const rn = race.race_number.toString();
        state.predictions[rn] = race.horses.map((h, i) => ({
          ...h,
          predicted_rank: h.rank,
          win_probability: h.model_probability,
          top3_probability: h.top3_probability || 0,
          odds: h.odds,
          edge: h.edge || 0,
          kelly_bet: 0,
          horse_name: h.horse_name,
          implied_prob: h.market_probability || 0,
          value_bet: h.value_bet ? "YES" : "",
        }));
      }
      renderRaceTabs();
      renderResults();
      renderExotics();
      renderBets();
      setStatus("done", "Done");
    } catch (e) {
      setTimeout(poll, POLL_INTERVAL);
    }
  };
  setTimeout(poll, POLL_INTERVAL);
}

/* ── Status ── */

function setStatus(cls, text) {
  const el = document.getElementById("status");
  el.className = `status ${cls}`;
  el.textContent = text;
}

/* ── Init ── */

document.getElementById("btnRunModel").onclick = runModel;
document.getElementById("btnAddHorse").onclick = () => document.getElementById("addHorseDialog").showModal();
document.getElementById("addCancel").onclick = () => document.getElementById("addHorseDialog").close();
document.getElementById("addConfirm").onclick = () => {
  const entry = {
    horse_name: document.getElementById("addName").value,
    jockey: document.getElementById("addJockey").value,
    trainer: document.getElementById("addTrainer").value,
    post_position: document.getElementById("addPP").value,
    dollar_odds: document.getElementById("addOdds").value,
    running_style: "U",
  };
  if (entry.horse_name && state.currentRace) {
    state.races[state.currentRace].entries.push(entry);
    state.edits.added.push({ race: state.currentRace, ...entry });
    renderEntries();
  }
  document.getElementById("addHorseDialog").close();
};

document.getElementById("patSave").onclick = () => {
  state.pat = document.getElementById("patInput").value;
  localStorage.setItem("sharpline_pat", state.pat);
  document.getElementById("patDialog").close();
};
document.getElementById("patCancel").onclick = () => document.getElementById("patDialog").close();
document.getElementById("resetToken").onclick = (e) => {
  e.preventDefault();
  localStorage.removeItem("sharpline_pat");
  state.pat = "";
  alert("Token cleared. Click Run Model to enter a new one.");
  document.getElementById("patDialog").showModal();
};
document.getElementById("reloadCard").onclick = (e) => {
  e.preventDefault();
  if (state.currentCard) loadCard(state.currentCard);
};

// Boot
(async () => {
  await discoverCards();
  if (state.cards.length > 0) {
    const picker = document.getElementById("datePicker");
    state.cards.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.path;
      opt.textContent = c.name;
      picker.appendChild(opt);
    });
    picker.onchange = () => loadCard(picker.value);
    loadCard(state.cards[0].path);
  }
})();
