let warningElement = null;
let lastText = "";

// Create / reuse warning UI
function showWarning(message, anchorElement) {
  if (!warningElement) {
    warningElement = document.createElement("div");
    warningElement.className = "pii-warning-banner";
    warningElement.style.position = "fixed";
    warningElement.style.background = "#facc15";
    warningElement.style.color = "#111";
    warningElement.style.padding = "8px 12px";
    warningElement.style.borderRadius = "8px";
    warningElement.style.fontSize = "13px";
    warningElement.style.fontFamily = "system-ui, sans-serif";
    warningElement.style.boxShadow = "0 4px 10px rgba(0,0,0,0.25)";
    warningElement.style.zIndex = "999999";
    document.body.appendChild(warningElement);
  }

  warningElement.textContent = message;

  const rect = anchorElement.getBoundingClientRect();
  warningElement.style.left = `${rect.left}px`;
  warningElement.style.top = `${rect.top - 40}px`;
  warningElement.style.display = "block";
}

function hideWarning() {
  if (warningElement) warningElement.style.display = "none";
}

function checkElement(el) {
  const text = el.value || el.innerText || "";
  if (text === lastText) return;
  lastText = text;

  if (!window.detectPII) {
    console.warn("detectPII not found — detector.js not loaded?");
    return;
  }

  const findings = window.detectPII(text);
  if (findings.length > 0) {
    const msg = `⚠️ Possible PII detected: ${findings.join(", ")}. Are you sure you want to share this?`;
    showWarning(msg, el);
  } else {
    hideWarning();
  }
}

// Attach listeners to inputs
function attachListeners() {
  const selector =
    'textarea, input[type="text"], input[type="email"], input[type="search"], [contenteditable="true"]';

  document.querySelectorAll(selector).forEach((el) => {
    if (el.dataset.piiAttached === "true") return;
    el.dataset.piiAttached = "true";

    el.addEventListener("input", () => checkElement(el));
    el.addEventListener("blur", hideWarning);
  });
}

// Run on load + dynamic pages like ChatGPT
const observer = new MutationObserver(() => attachListeners());
observer.observe(document.documentElement, { childList: true, subtree: true });

attachListeners();
console.log("PII Guardian content script loaded");
