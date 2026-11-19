# Reverse Engineer Anything (REA) — Installation, Configuration, and Verification Report

**Document Version:** 1.0.0  
**Date:** November 16, 2026  
**Repository Source:** `https://github.com/morluto/rea` (official repository)  
**Package:** `rea-agents` v6.9.0 (npm registry, user-isolated prefix)  
**Target Environment:** Linux (`x86_64`, Fedora/Linux kernel), Node.js v22.23.1, npm 10.9.8, Google Chrome 156 (Headless CDP)

---

## 1. Executive Summary

In accordance with Phase 5 of the research project roadmap, the official **Reverse Engineer Anything (REA)** project (`morluto/rea`) was inspected, installed in an isolated user-level environment, configured, and empirically verified against a harmless public website (`https://example.com`).

### Verification Outcome
* **Package Installed:** `rea-agents@6.9.0`
* **Binary Location:** `/home/chitta/.npm-global/bin/rea`
* **Verification Test:** Executed CDP inspection against headless Chrome visiting `https://example.com`.
* **Verification Result:** PASS (`14,785` bytes of normalized structured JSON returned covering DOM nodes, scripts, frames, security state, accessibility, network capture, and limitations).
* **Isolation Guarantee:** Zero modifications to existing agent configuration files, zero alterations to the Python machine learning environment, and zero installation of unnecessary native binary disassemblers (e.g. Ghidra/Hopper) that are not needed for web/affective computing workflow inspection.

---

## 2. Prerequisites and Compatibility

| Component | Required Version | Observed on System | Compatibility Status |
| :--- | :--- | :--- | :--- |
| **OS** | Linux x86_64 | Linux (Fedora release 41) | Compatible |
| **Runtime** | Node.js >= 18 | Node.js v22.23.1 | Fully Compatible |
| **Package Manager**| npm >= 9 | npm 10.9.8 | Fully Compatible |
| **Browser Engine** | Chrome / Chromium | Google Chrome 156.0.8078.73 | Fully Compatible |
| **User Space Isolation** | User prefix | `~/.npm-global` | Fully Isolated |

---

## 3. Installation Steps

REA was installed strictly in user space without requiring root privileges:
```bash
# Verify npm user prefix
npm config get prefix
# Output: /home/chitta/.npm-global

# Install official rea-agents package from npm
npm install -g rea-agents

# Verify CLI availability
which rea
# Output: /home/chitta/.npm-global/bin/rea

# Verify CLI version
rea --version
# Output: 6.9.0
```

---

## 4. Empirical Verification Workflow

To verify functionality safely without bypassing paywalls or access controls, a local CDP pipeline was tested against the official IANA test domain `https://example.com`:

### Step 1: Launch Headless Browser with CDP Endpoint
```bash
google-chrome --headless=new --remote-debugging-port=9222 --disable-gpu --no-sandbox https://example.com &
```

### Step 2: List Browser Targets via REA
```bash
rea list-browser-targets http://127.0.0.1:9222 --format json
```
*Output Summary:*
```json
{
  "evidence_id": "ev_aa2df8c53d05277094a47fe20e2a701d78ae55d7a87c12428e7cb318b803480f",
  "operation": "list_browser_targets",
  "normalized_result": {
    "browser": {
      "product": "Chrome/156.0.8078.73",
      "protocol_version": "1.3"
    },
    "targets": [
      {
        "target_id": "5283FBAC1234691596E2C0A00D9079D0",
        "type": "page",
        "title": "Example Domain",
        "url": "https://example.com/",
        "origin": "https://example.com"
      }
    ]
  }
}
```

### Step 3: Passive Page Inspection via REA
```bash
rea inspect-web-page http://127.0.0.1:9222 "5283FBAC1234691596E2C0A00D9079D0" --format json
```
*Output Summary:*
* **JSON Size:** 14,785 bytes
* **Normalized Result Keys:**
  `["accessibility", "browser", "capture_window", "completeness", "console", "dom", "frames", "limitations", "metadata", "network", "resources", "scripts", "storage", "target", "workers"]`
* **Scripts Detected:** 2 inline scripts.
* **DOM Integrity:** Fully reconstructed with accessibility tree and bounding box hierarchy.

---

## 5. REA Capabilities Relevant to Our Research

For our NLP and conversation analytics platform, REA provides three targeted capabilities:
1. **Passive Web Workflow Inspection:** Analyzing consumer chat applications' public documentation, landing pages, and export schemas (e.g., verifying Instagram or WhatsApp export structure without reverse-engineering mobile binaries).
2. **JavaScript Application Reconstruction:** Statically analyzing Electron or web-based conversation viewer bundles (`rea analyze-javascript-application`) to discover DOM layout techniques and chart data schemas.
3. **Structured Evidence Integrity:** Every observation output carries an immutable `evidence_id`, `limitations`, and provenance metadata, adhering to rigorous empirical research standards.

---

## 6. Limitations and Security Guardrails

* **Explicit IP Requirement:** REA strictly enforces `http://127.0.0.1:<port>` rather than `http://localhost:<port>` to prevent DNS-rebinding attacks on loopback debugging ports.
* **Passive Observation Only:** REA operates passively through CDP without modifying page behavior or injecting automated scripts.
* **Ethical Boundaries:** Under no circumstances is REA used to bypass authentication, crack proprietary model weights, or reverse-engineer paywalled services.

---
*Report completed and verified. REA v6.9.0 is fully functional and ready for authorized analysis workflows.*
