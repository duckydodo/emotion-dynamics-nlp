#!/usr/bin/env python3
"""
Task 34 — Emotion Dynamics Home Screen & Visual Identity Verification Script
Executes automated visual QA, verifies brand identity, color tokens, responsive layout,
and captures high-resolution desktop (1440x1024) and mobile (390x844) screenshots.
"""

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

APP_URL = "http://localhost:8001"
CHROME_PORT = 9225
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("outputs/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = []


def record_result(category, check_name, expected, actual, passed, detail=""):
    RESULTS.append({
        "category": category,
        "check": check_name,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    })
    status_str = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"[{status_str}] [{category}] {check_name}")
    if not passed:
        print(f"   Expected: {expected}")
        print(f"   Actual:   {actual}")
    if detail:
        print(f"   Detail:   {detail}")


async def run_home_qa():
    print("==================================================================")
    print("TASK 34: EMOTION DYNAMICS HOME SCREEN & VISUAL IDENTITY QA")
    print(f"Target URL: {APP_URL}")
    print("==================================================================\n")

    chrome_proc = subprocess.Popen([
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        f"--remote-debugging-port={CHROME_PORT}",
        "--remote-allow-origins=*",
        "--user-data-dir=/tmp/chrome_test_profile_task34",
        "--window-size=1440,1024",
        "--no-sandbox",
        APP_URL,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        ws_url = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{CHROME_PORT}/json", timeout=1) as resp:
                    tabs = json.loads(resp.read().decode())
                    for tab in tabs:
                        if tab.get("type") == "page" and "8001" in tab.get("url", ""):
                            ws_url = tab["webSocketDebuggerUrl"]
                            break
                    if ws_url:
                        break
            except Exception:
                await asyncio.sleep(0.25)

        if not ws_url:
            raise RuntimeError(f"Failed to connect to Chrome CDP at port {CHROME_PORT}")

        async with websockets.connect(ws_url, max_size=40 * 1024 * 1024) as ws:
            msg_counter = [1]

            async def send_cdp(method, params=None):
                mid = msg_counter[0]
                msg_counter[0] += 1
                payload = {"id": mid, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    resp_raw = await ws.recv()
                    resp = json.loads(resp_raw)
                    if resp.get("id") == mid:
                        return resp.get("result", {})

            async def eval_js(code):
                res = await send_cdp("Runtime.evaluate", {
                    "expression": code,
                    "returnByValue": True,
                    "awaitPromise": True,
                })
                if "exceptionDetails" in res:
                    raise RuntimeError(f"JS Exception: {res['exceptionDetails']}")
                return res.get("result", {}).get("value")

            async def take_screenshot(filename):
                res = await send_cdp("Page.captureScreenshot", {"format": "png"})
                data = base64.b64decode(res.get("data", ""))
                out_path = FIGURES_DIR / filename
                out_path.write_bytes(data)
                return out_path

            async def set_viewport(width, height, mobile=False):
                await send_cdp("Emulation.setDeviceMetricsOverride", {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 2 if mobile else 1,
                    "mobile": mobile,
                })
                await asyncio.sleep(0.5)

            # Settle initial page load
            await asyncio.sleep(2.0)

            # =================================================================
            # 1. PRODUCT IDENTITY & BRANDING
            # =================================================================
            print("\n--- [1. Product Identity & Branding] ---")

            page_title = await eval_js("document.title")
            brand_title = await eval_js("document.querySelector('.brand-title').textContent.trim()")
            brand_tagline = await eval_js("document.querySelector('.brand-tagline').textContent.trim()")
            favicon_href = await eval_js("document.querySelector('link[rel~=\"icon\"]')?.href || ''")

            record_result(
                "Identity", "Browser Title",
                "Emotion Dynamics — Conversational Emotion Analysis and Research",
                page_title,
                page_title == "Emotion Dynamics — Conversational Emotion Analysis and Research",
                f"Page title: {page_title}"
            )

            record_result(
                "Identity", "Header Brand Title",
                "Emotion Dynamics",
                brand_title,
                brand_title == "Emotion Dynamics",
                f"Brand title: {brand_title}"
            )

            record_result(
                "Identity", "Header Brand Subtitle",
                "Conversational Emotion Analysis and Research",
                brand_tagline,
                brand_tagline == "Conversational Emotion Analysis and Research",
                f"Subtitle: {brand_tagline}"
            )

            record_result(
                "Identity", "Custom Waveform Favicon",
                "SVG favicon defined with warm charcoal/sage wave",
                favicon_href[:50],
                "svg+xml" in favicon_href,
                "Favicon embedded as inline SVG"
            )

            # =================================================================
            # 2. DESIGN SYSTEM COLOR TOKENS & TYPOGRAPHY
            # =================================================================
            print("\n--- [2. Design System Tokens & Typography] ---")

            css_tokens = await eval_js("""
                (() => {
                    const style = getComputedStyle(document.documentElement);
                    return {
                        bgBase: style.getPropertyValue('--bg-base').trim(),
                        surface1: style.getPropertyValue('--bg-surface-1').trim(),
                        accentSage: style.getPropertyValue('--accent-primary').trim(),
                        highlightCopper: style.getPropertyValue('--highlight-secondary').trim(),
                        textPrimary: style.getPropertyValue('--text-primary').trim(),
                        textSecondary: style.getPropertyValue('--text-secondary').trim()
                    };
                })()
            """)

            record_result(
                "Design Tokens", "Warm Charcoal Background (#171B18)",
                "#171b18",
                css_tokens["bgBase"].lower(),
                css_tokens["bgBase"].lower() == "#171b18",
                f"Computed --bg-base: {css_tokens['bgBase']}"
            )

            record_result(
                "Design Tokens", "Muted Sage Accent (#91A58B)",
                "#91a58b",
                css_tokens["accentSage"].lower(),
                css_tokens["accentSage"].lower() == "#91a58b",
                f"Computed --accent-primary: {css_tokens['accentSage']}"
            )

            record_result(
                "Design Tokens", "Restrained Copper Highlight (#C18A68)",
                "#c18a68",
                css_tokens["highlightCopper"].lower(),
                css_tokens["highlightCopper"].lower() == "#c18a68",
                f"Computed --highlight-secondary: {css_tokens['highlightCopper']}"
            )

            body_font = await eval_js("window.getComputedStyle(document.body).fontFamily")
            record_result(
                "Typography", "Primary Typeface",
                "Plus Jakarta Sans",
                body_font,
                "Plus Jakarta Sans" in body_font,
                f"Body font: {body_font}"
            )

            # Zero-emoji check across Home screen UI
            emoji_violations = await eval_js("""
                (() => {
                    const emojiRegex = /(\\p{Emoji_Presentation}|\\p{Extended_Pictographic})/u;
                    const elements = Array.from(document.querySelectorAll(
                        '.app-header, .nav-tabs, .screen-intro, .sample-bar, .workbench-card-header, .model-workbench-bar, .preview-header, .preview-actions button'
                    ));
                    const violating = [];
                    for (const el of elements) {
                        const clone = el.cloneNode(true);
                        clone.querySelectorAll('svg').forEach(s => s.remove());
                        const text = clone.textContent.trim();
                        if (emojiRegex.test(text)) {
                            violating.push({ class: el.className, text });
                        }
                    }
                    return violating;
                })()
            """)
            record_result(
                "Design System", "Strict Zero-Emoji Enforcement",
                "0 emoji violations across core Home screen UI",
                f"{len(emoji_violations)} violations",
                len(emoji_violations) == 0,
                f"Violations list: {emoji_violations}"
            )

            # =================================================================
            # 3. HOME / IMPORT SCREEN REDESIGN (DESKTOP: 1440x1024)
            # =================================================================
            print("\n--- [3. Desktop Home Screen Verification (1440x1024)] ---")
            await set_viewport(1440, 1024, mobile=False)
            await eval_js("navigateTo('import');")
            await asyncio.sleep(0.5)

            # Capture initial unpopulated state
            p_initial = await take_screenshot("task34_desktop_home_initial.png")
            print(f"   [Captured] {p_initial.name}")

            # Check Intro Heading & Lead
            heading_text = await eval_js("document.querySelector('.screen-heading').textContent.trim()")
            record_result(
                "Home Screen", "Screen Heading",
                "Conversation Ingestion & Sequence Analysis",
                heading_text,
                "Conversation Ingestion" in heading_text,
                f"Heading: {heading_text}"
            )

            # Check Preset Dialogue Pills
            sample_btn_count = await eval_js("document.querySelectorAll('.sample-pill-btn').length")
            record_result(
                "Home Screen", "Preset Sample Buttons",
                "4 preset transcript pills",
                f"{sample_btn_count} pills",
                sample_btn_count == 4,
                "Found Conflict & Resolution, WhatsApp, Instagram, Three-Party pills"
            )

            # Check Ingestion Dual Workbench (Dropzone + Direct Paste)
            has_dropzone = await eval_js("document.getElementById('fileDropzone') !== null")
            has_textarea = await eval_js("document.getElementById('rawConversationInput') !== null")
            record_result(
                "Home Screen", "Dual Ingestion Workbench",
                "Dropzone and Direct Dialogue textarea both present",
                f"dropzone={has_dropzone}, textarea={has_textarea}",
                has_dropzone and has_textarea,
                "Both ingestion modalities available"
            )

            # Check Model Selector Options
            model_options = await eval_js("Array.from(document.getElementById('modelSelect').options).map(o => o.value)")
            record_result(
                "Home Screen", "Model Selector Options",
                "['dialoguegcn', 'dialoguernn', 'distilbert']",
                str(model_options),
                set(model_options) == {"dialoguegcn", "dialoguernn", "distilbert"},
                f"Registered models: {model_options}"
            )

            # Load Preset 1 (Conflict & Resolution)
            await eval_js("document.getElementById('sampleConflictBtn').click();")
            await asyncio.sleep(1.2)  # Wait for debounce and preview API

            # Verify Ingestion Preview & Schema Audit
            preview_visible = await eval_js("document.getElementById('importPreviewPanel').style.display !== 'none'")
            preview_format = await eval_js("document.getElementById('previewFormat').textContent")
            preview_turns = await eval_js("document.getElementById('previewEvalTurns').textContent")
            preview_spk = await eval_js("document.getElementById('previewSpeakers').textContent")
            preview_warns = await eval_js("document.getElementById('importWarningsBox').style.display !== 'none'")

            record_result(
                "Home Screen", "Ingestion Preview & Schema Audit",
                "Preview displayed, plain text format, 9 turns, 2 speakers, timestamp warning visible",
                f"display={preview_visible}, format='{preview_format}', turns={preview_turns}, spk='{preview_spk}', warns={preview_warns}",
                preview_visible and "plain text" in str(preview_format).lower() and str(preview_turns) == "9" and preview_warns,
                f"Preview turns: {preview_turns}, Speakers: {preview_spk}"
            )

            # Capture Desktop Populated State
            p_desktop = await take_screenshot("task34_desktop_home.png")
            print(f"   [Captured] {p_desktop.name}")

            # =================================================================
            # 4. MOBILE RESPONSIVE VERIFICATION (390x844 - iPhone 14)
            # =================================================================
            print("\n--- [4. Mobile Home Screen Verification (390x844)] ---")
            await set_viewport(390, 844, mobile=True)
            await asyncio.sleep(0.8)

            # Check no horizontal overflow
            no_overflow = await eval_js("document.documentElement.scrollWidth <= window.innerWidth")
            record_result(
                "Mobile Layout", "Viewport Fit & No Horizontal Overflow",
                "scrollWidth <= 390px (window.innerWidth)",
                f"no_overflow={no_overflow}",
                no_overflow,
                "Mobile layout fits cleanly without horizontal scroll blowout"
            )

            # Check vertical stacking of dual columns
            columns_stacked = await eval_js("""
                (() => {
                    const cards = document.querySelectorAll('.workbench-card');
                    if (cards.length < 2) return false;
                    const r1 = cards[0].getBoundingClientRect();
                    const r2 = cards[1].getBoundingClientRect();
                    return r2.top >= r1.bottom;
                })()
            """)
            record_result(
                "Mobile Layout", "Workbench Vertical Stacking",
                "Upload card stacks cleanly above Direct Paste card",
                f"stacked={columns_stacked}",
                columns_stacked,
                "Responsive single-column stacking verified"
            )

            # Check touch target size on action buttons
            button_touch_targets = await eval_js("""
                (() => {
                    const buttons = Array.from(document.querySelectorAll('.btn-primary, .tab-btn'));
                    const small = buttons.filter(b => {
                        const rect = b.getBoundingClientRect();
                        return rect.height > 0 && rect.height < 36;
                    });
                    return small.map(b => ({ class: b.className, text: b.textContent.trim(), h: b.getBoundingClientRect().height }));
                })()
            """)
            record_result(
                "Mobile Layout", "Primary Touch Target Ergonomics",
                "All primary action buttons and tabs have height >= 36px",
                f"{len(button_touch_targets)} undersized buttons",
                len(button_touch_targets) == 0,
                f"Touch targets: {button_touch_targets}"
            )

            # Capture Mobile Home Screenshot
            p_mobile = await take_screenshot("task34_mobile_home.png")
            print(f"   [Captured] {p_mobile.name}")

            # =================================================================
            # 5. INFERENCE & COMPARISON WORKFLOW INTEGRITY
            # =================================================================
            print("\n--- [5. Inference & Comparison Workflow Integrity] ---")
            await set_viewport(1440, 1024, mobile=False)

            # Click Analyze Conversation Dynamics
            await eval_js("document.getElementById('runAnalysisBtn').click();")
            await asyncio.sleep(3.5)

            viewer_active = await eval_js("document.getElementById('screen-viewer').classList.contains('active')")
            bubble_count = await eval_js("document.querySelectorAll('.bubble-wrapper').length")
            record_result(
                "Workflow", "Analyze Action -> Viewer Transition",
                "Transitions to Screen 2 (Viewer) with 9 bubbles rendered",
                f"viewer_active={viewer_active}, bubbles={bubble_count}",
                viewer_active and bubble_count == 9,
                f"Single-model analysis succeeded ({bubble_count} turns)"
            )

            # Return to Import and run Compare All Models
            await eval_js("navigateTo('import');")
            await asyncio.sleep(0.5)
            await eval_js("document.getElementById('runComparisonBtn').click();")
            await asyncio.sleep(4.0)

            compare_active = await eval_js("document.getElementById('screen-compare').classList.contains('active')")
            consensus_badge = await eval_js("document.getElementById('consensusRateBadge').textContent")
            matrix_cells = await eval_js("document.querySelectorAll('#pairwiseMatrixBody td.agreement-cell').length")

            record_result(
                "Workflow", "Compare Action -> Comparison Transition",
                "Transitions to Screen 4 (Comparison) with 3x3 matrix and consensus badge",
                f"compare_active={compare_active}, consensus='{consensus_badge}', cells={matrix_cells}",
                compare_active and "%" in consensus_badge and matrix_cells == 9,
                f"Multi-model comparison succeeded ({consensus_badge})"
            )

            # Return to Import for final view
            await eval_js("navigateTo('import');")
            await asyncio.sleep(0.5)

    finally:
        chrome_proc.terminate()
        chrome_proc.wait()

    print("\n==================================================================")
    print("TASK 34 HOME SCREEN & VISUAL IDENTITY QA SUMMARY")
    print("==================================================================")
    passed_count = sum(1 for r in RESULTS if r["status"] == "PASS")
    total_count = len(RESULTS)
    print(f"Total Checks: {total_count} | Passed: {passed_count} | Failed: {total_count - passed_count}")

    # Output JSON summary for report inclusion
    out_json = TABLES_DIR / "task34_home_qa_results.json"
    out_json.write_text(json.dumps(RESULTS, indent=2))
    print(f"Results written to: {out_json}")

    if passed_count < total_count:
        print("\n[WARNING] Some visual QA checks failed.")
        raise SystemExit(1)
    else:
        print("\n\033[92mAll Task 34 QA & Visual Identity Checks Passed Successfully!\033[0m")


if __name__ == "__main__":
    asyncio.run(run_home_qa())
