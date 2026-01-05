// detector.js

const EMAIL_REGEX = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i;
const PHONE_REGEX = /\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{3,4}\b/;
const CREDIT_CARD_REGEX = /\b(?:\d[ -]*?){13,16}\b/;
const SSN_REGEX = /\b\d{3}-\d{2}-\d{4}\b/;

// attach to window so content.js can use it
window.detectPII = function(text) {
  const findings = [];

  if (EMAIL_REGEX.test(text)) findings.push("Email");
  if (PHONE_REGEX.test(text)) findings.push("Phone");
  if (CREDIT_CARD_REGEX.test(text)) findings.push("Credit Card");
  if (SSN_REGEX.test(text)) findings.push("SSN");

  return findings;
};
