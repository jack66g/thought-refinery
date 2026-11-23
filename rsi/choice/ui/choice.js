"use strict";

const byId = (id) => document.getElementById(id);
const enabled = byId("enabled");
const open = byId("open");
const refresh = byId("retry");
const dialog = byId("round");
const confirm = byId("confirm");
const later = byId("later");
const options = byId("options");
let status;
let round;
let busy = false;

async function request(path, data) {
  const response = await fetch("/choice/api/" + path, {
    method: data === undefined ? "GET" : "POST",
    headers: data === undefined ? {} : { "Content-Type": "application/json" },
    body: data === undefined ? undefined : JSON.stringify(data),
    cache: "no-store",
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : "请求暂时无法完成。");
  }
  return body;
}

function readiness() {
  if (!status.enabled) return "开启后可进入答题窗口。";
  if (status.active_round) return "有一轮已保存，可以继续。";
  if (status.started_today >= status.rounds_per_day) return "今天的轮数已用完，明天可继续。";
  if (status.profile_updates < status.min_profile_updates) {
    return "画像还在积累，达到 " + status.min_profile_updates + " 次更新后可生成题目。";
  }
  if (status.memory_online === false) return "记忆服务暂时离线，可以稍后刷新。";
  if ((status.usable_memories ?? 0) < status.min_memories) {
    return "需要至少 " + status.min_memories + " 条可用记忆，稍后可以刷新。";
  }
  return "";
}

function dashboard() {
  enabled.checked = status.enabled;
  enabled.disabled = busy;
  open.hidden = !status.enabled;
  open.textContent = status.active_round ? "继续答题" : "进入答题";
  refresh.hidden = false;
  byId("status").textContent = readiness();
  byId("quota").textContent = status.started_today + " / " + status.rounds_per_day;
  byId("profile-count").textContent = status.profile_updates;
  byId("memory-count").textContent = status.usable_memories ?? "—";
  byId("total").textContent = status.total_answered + " 轮完整记录";
  const cards = status.estimates.map((axis) => {
    const card = document.createElement("article");
    card.className = "axis";
    const title = document.createElement("h3");
    title.textContent = axis.label;
    const evidence = document.createElement("p");
    evidence.textContent = axis.observations + " 轮 · " + axis.contexts + " 类情境";
    card.append(title, evidence);
    if (axis.a_share !== null) {
      const meter = document.createElement("meter");
      meter.min = 0;
      meter.max = 100;
      meter.value = axis.a_share;
      meter.setAttribute("aria-label", axis.label + " · " + axis.a + " " + axis.a_share + "%");
      const anchors = document.createElement("p");
      anchors.className = "anchors";
      for (const label of [axis.a, axis.b]) {
        const item = document.createElement("span");
        item.textContent = label;
        anchors.append(item);
      }
      card.append(meter, anchors);
    }
    return card;
  });
  byId("axes").replaceChildren(...cards);
}

function render(next) {
  round = next;
  const step = round?.step;
  const question = step === "decision" || step === "reason";
  const intro = !step;
  const completed = step === "complete";
  byId("progress").textContent = ({
    decision: "第 1 / 2 题 · 选择做法",
    reason: "第 2 / 2 题 · 选择理由",
    decision_saved: "第 2 / 2 题 · 继续",
    complete: "本轮已完成",
    skipped: "本轮已略过",
  })[step] ?? "情境选择";
  byId("scene-title").textContent = round?.title ?? "从一幕开始";
  byId("background").textContent = round?.background ??
    "一轮有两题。做完一轮后，你可以继续下一幕，或留到下次。";
  const previous = byId("previous");
  previous.hidden = !(round?.decision);
  const quote = (text) => "「" + text.trim().replace(/[。.!！?？]+$/u, "") + "」";
  previous.textContent = !round?.decision ? "" : "你选择了" + quote(round.decision) +
    (completed ? "。首要理由是" + quote(round.reason) + "。" : "。");
  byId("choices").hidden = !question;
  byId("question").textContent = round?.question ?? "";
  byId("notice").textContent = intro ? readiness() : completed ? "已生成一条人格预训练记录和一条人格后训练记录。" : "";
  confirm.hidden = step === "skipped";
  confirm.textContent = ({
    decision: "确认做法",
    reason: "确认理由",
    decision_saved: "重新生成理由",
    complete: "下一轮",
  })[step] ?? "开始答题";
  confirm.disabled = busy || question || ((intro || completed) &&
    (!status.enabled || status.started_today >= status.rounds_per_day ||
     status.profile_updates < status.min_profile_updates ||
     (status.usable_memories ?? 0) < status.min_memories));
  later.hidden = step !== "decision_saved";
  later.disabled = busy;
  options.replaceChildren(...(round?.options ?? []).map((option) => {
    const label = document.createElement("label");
    label.className = "option";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "choice";
    input.value = option.id;
    input.disabled = busy;
    input.addEventListener("change", () => { confirm.disabled = busy; });
    const caption = document.createElement("span");
    caption.textContent = option.text;
    label.append(input, caption);
    return label;
  }));
}

function openRound() {
  render(status.active_round);
  if (!dialog.open) dialog.showModal();
  byId("scene-title").focus();
}

async function operate(task) {
  if (busy) return;
  busy = true;
  enabled.disabled = true;
  open.disabled = true;
  refresh.disabled = true;
  confirm.disabled = true;
  later.disabled = true;
  options.querySelectorAll("input").forEach((input) => { input.disabled = true; });
  try {
    await task();
  } catch (error) {
    (dialog.open ? byId("notice") : byId("status")).textContent = error.message || "连接失败，请稍后重试。";
    refresh.hidden = false;
    if (status) enabled.checked = status.enabled;
  } finally {
    busy = false;
    enabled.disabled = !status;
    open.disabled = false;
    refresh.disabled = false;
    later.disabled = false;
    options.querySelectorAll("input").forEach((input) => { input.disabled = false; });
    if (dialog.open) {
      const step = round?.step;
      const answerStep = step === "decision" || step === "reason";
      const canStart = status.enabled && status.started_today < status.rounds_per_day &&
        status.profile_updates >= status.min_profile_updates && (status.usable_memories ?? 0) >= status.min_memories;
      confirm.disabled = answerStep ? !options.querySelector("input:checked") :
        step !== "decision_saved" && !canStart;
    }
  }
}

async function load(show) {
  status = await request("state");
  dashboard();
  if (show && status.enabled) openRound();
}

async function respond(step, option) {
  try {
    const result = await request("answer", {round_id: round.id, step, option});
    status.active_round = ["complete", "skipped"].includes(result.step) ? null : result;
    render(result);
    if (!status.active_round) await load(false);
  } catch (error) {
    const current = await request("state").catch(() => null);
    if (current) {
      status = current;
      dashboard();
      if (current.active_round) render(current.active_round);
    }
    throw error;
  }
}

enabled.addEventListener("change", () => operate(async () => {
  status = await request("settings", {enabled: enabled.checked, open_window: false});
  dashboard();
  if (status.enabled) openRound();
  else if (dialog.open) dialog.close();
}));
refresh.addEventListener("click", () => operate(() => load(false)));
open.addEventListener("click", openRound);
byId("close").addEventListener("click", () => dialog.close());
later.addEventListener("click", () => operate(() => respond("reason", "skip")));
confirm.addEventListener("click", () => operate(async () => {
  const step = round?.step;
  if (!step || step === "complete") {
    byId("notice").textContent = "正在生成情境…";
    const item = await request("start", {});
    status.active_round = item;
    status.started_today += 1;
    dashboard();
    render(item);
    byId("scene-title").focus();
    return;
  }
  const choice = options.querySelector("input:checked");
  if (step === "decision_saved" || choice) {
    byId("notice").textContent = step === "reason" ? "正在保存这一轮…" : "正在生成下一题…";
    await respond(step === "decision_saved" ? "decision" : step, choice?.value ?? round.retry);
    byId("scene-title").focus();
  }
}));
operate(() => load(true));
