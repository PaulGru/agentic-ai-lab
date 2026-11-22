"use strict";

const byId = id => document.getElementById(id);
let snapshot = null;
let selectedStep = null;
let lastPayload = "";

function pdfLink(edition, label = "Ouvrir le PDF") {
  const link = document.createElement("a");
  link.href = edition.url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = label;
  return link;
}

function selectStep(step) {
  selectedStep = step;
  render();
}

function render() {
  const latest = snapshot.steps.at(-1);
  const active = snapshot.steps.find(step => step.step === selectedStep) || latest;
  const statuses = {waiting: "En attente", running: "Étape réservée", complete: "Terminée", blocked: "À examiner"};
  const runState = latest?.action === "autonomy_pause" ? "blocked" : latest?.status;
  byId("mission").textContent = snapshot.run;
  byId("run-status").textContent = statuses[runState] || "Journal en attente";
  byId("run-status").dataset.state = runState || "waiting";
  byId("step-count").textContent = `${snapshot.steps.length} jalon${snapshot.steps.length === 1 ? "" : "s"}`;
  byId("snapshot-title").textContent = selectedStep === null ? "État actuel" : `Jalon ${selectedStep}`;
  byId("back").hidden = selectedStep === null;
  byId("stage").textContent = active ? (active.status === "complete" ? "Mission terminée" : active.stage) : "En attente du journal…";
  byId("message").textContent = active?.message || "";
  byId("step-pdf").replaceChildren();
  const editionAtStep = snapshot.editions.find(edition => edition.step === active?.edition);
  if (editionAtStep?.status === "ready") byId("step-pdf").append(pdfLink(editionAtStep, "Ouvrir le PDF de ce jalon"));
  const highlights = active?.highlights || [];
  byId("snapshot-data").hidden = highlights.length === 0;
  byId("highlights").replaceChildren(...highlights.flatMap(item => {
    const title = document.createElement("dt");
    const description = document.createElement("dd");
    title.textContent = item.label;
    description.textContent = item.text;
    return [title, description];
  }));

  const steps = [...snapshot.steps].reverse().map(step => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    const id = document.createElement("span");
    const message = document.createElement("span");
    id.className = "step-id";
    id.textContent = `${step.step} · v${step.version}`;
    message.className = "message";
    message.textContent = step.message;
    button.type = "button";
    button.setAttribute("aria-pressed", String(step.step === active?.step));
    button.addEventListener("click", () => selectStep(step.step === latest?.step ? null : step.step));
    button.append(id, message);
    item.append(button);
    return item;
  });
  const stepList = byId("steps");
  const offset = stepList.scrollTop;
  stepList.replaceChildren(...steps);
  stepList.scrollTop = offset;

  const editions = [...snapshot.editions].reverse().map(edition => {
    const item = document.createElement("li");
    const label = document.createElement("strong");
    label.textContent = `v${edition.version} · ${edition.approved ? "Approuvé" : "Brouillon"}`;
    item.append(label);
    if (edition.status === "ready") {
      item.append(pdfLink(edition));
    } else {
      const status = document.createElement("span");
      status.textContent = edition.status === "error" ? "Compilation indisponible" : "PDF en attente";
      item.append(status);
      if (edition.message) {
        const message = document.createElement("small");
        message.textContent = edition.message;
        item.append(message);
      }
    }
    return item;
  });
  if (!editions.length) {
    const empty = document.createElement("li");
    empty.textContent = "Le premier PDF apparaîtra dès que le brouillon sera prêt.";
    editions.push(empty);
  }
  byId("editions").replaceChildren(...editions);
}

async function refresh() {
  try {
    const response = await fetch("/api/progress", {cache: "no-store"});
    if (!response.ok) throw new Error("Journal indisponible");
    const payload = await response.text();
    if (payload !== lastPayload) {
      snapshot = JSON.parse(payload);
      lastPayload = payload;
      render();
    }
    byId("connection").textContent = "Suivi local connecté";
  } catch {
    byId("connection").textContent = "Connexion interrompue · nouvelle tentative";
  } finally {
    window.setTimeout(refresh, 1500);
  }
}

byId("back").addEventListener("click", () => selectStep(null));
refresh();
