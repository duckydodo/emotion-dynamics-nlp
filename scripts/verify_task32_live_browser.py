#!/usr/bin/env python3
"""
Task 32 — Live Browser Automation and Verification Script
Uses headless Google Chrome via Chrome DevTools Protocol (CDP) to interact with and
verify the live Dockerized five-screen frontend and three-model backend on port 8001.
"""

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error
import websockets

APP_URL = "http://localhost:8001"
CHROME_PORT = 9223
SCREENSHOT_DIR = Path("outputs/figures")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = []


def record_result(screen, feature, expected, actual, passed, detail=""):
    RESULTS.append({
        "screen": screen,
        "feature": feature,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    })
    status_str = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"[{status_str}] [{screen}] {feature}")
    if not passed:
        print(f"   Expected: {expected}")
        print(f"   Actual:   {actual}")
    if detail:
        print(f"   Detail:   {detail}")


async def run_browser_verification():
    print("==================================================================")
    print("TASK 32: LIVE BROWSER AUTOMATION & COMPLIANCE VERIFICATION")
    print(f"Target URL: {APP_URL}")
    print("==================================================================\n")

    # 1. Launch Headless Google Chrome
    chrome_proc = subprocess.Popen([
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        f"--remote-debugging-port={CHROME_PORT}",
        "--remote-allow-origins=*",
        "--user-data-dir=/tmp/chrome_test_profile_task32",
        "--window-size=1440,1024",
        "--no-sandbox",
        APP_URL,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # 2. Connect to CDP WebSocket
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

        async with websockets.connect(ws_url, max_size=30 * 1024 * 1024) as ws:
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
                out_path = SCREENSHOT_DIR / filename
                out_path.write_bytes(data)
                return out_path

            # Allow DOM & health checks to settle
            await asyncio.sleep(2.0)

            # -------------------------------------------------------------
            # TEST 1: Page Title & Global Header
            # -------------------------------------------------------------
            page_title = await eval_js("document.title")
            record_result(
                "Global",
                "Page Title Verification",
                "Emotion Dynamics Analyzer | Multi-Model Conversation Intelligence",
                page_title,
                "Multi-Model Conversation Intelligence" in str(page_title),
                f"Served Title: {page_title}"
            )

            # -------------------------------------------------------------
            # TEST 2: System Status Pill & Backend Connectivity
            # -------------------------------------------------------------
            status_text = await eval_js("document.getElementById('systemStatusText').textContent")
            status_dot_offline = await eval_js("document.querySelector('.status-dot').classList.contains('offline')")
            record_result(
                "Global",
                "System Status Pill",
                "'3 Models Online' and status dot not offline",
                f"'{status_text}' (offline_class={status_dot_offline})",
                "3 Models Online" in str(status_text) and not status_dot_offline,
                "Confirmed live connection between frontend and backend on port 8001"
            )

            # -------------------------------------------------------------
            # TEST 3: Model Selector Registry Population
            # -------------------------------------------------------------
            model_options = await eval_js("""
                Array.from(document.getElementById('modelSelect').options).map(o => ({
                    value: o.value,
                    text: o.text,
                    disabled: o.disabled,
                    selected: o.selected
                }))
            """)
            opt_values = [o["value"] for o in model_options]
            selected_model = await eval_js("document.getElementById('modelSelect').value")
            record_result(
                "Home / Import",
                "Model Selector Options",
                "['distilbert', 'dialoguernn', 'dialoguegcn']",
                str(opt_values),
                set(opt_values) == {"distilbert", "dialoguernn", "dialoguegcn"},
                f"Options: {opt_values}, default selected: {selected_model}"
            )

            # -------------------------------------------------------------
            # TEST 4: Navigation Tabs & Ingestion Format Tags
            # -------------------------------------------------------------
            nav_tabs = await eval_js("""
                Array.from(document.querySelectorAll('.tab-btn')).map(t => t.textContent.trim())
            """)
            format_tags = await eval_js("""
                Array.from(document.querySelectorAll('.format-tag')).map(t => t.textContent.trim())
            """)
            record_result(
                "Home / Import",
                "Navigation Tabs & Format Tags",
                "5 navigation screen tabs and 4 format tags rendered",
                f"nav_tabs={len(nav_tabs)}, format_tags={format_tags}",
                len(nav_tabs) == 5 and len(format_tags) == 4,
                f"Nav screens: {len(nav_tabs)}, Formats: {format_tags}"
            )

            # -------------------------------------------------------------
            # TEST 5: Live Import Preview & Warning Code Rendering
            # -------------------------------------------------------------
            # Set conversation with dialogue turns containing commas and a warning
            test_convo = (
                "Header line before dialogue without speaker marker\n"
                "Ross: Hey Rachel, how are you feeling today?\n"
                "Rachel: I am doing wonderfully, thank you!\n"
                "Ross: That is fantastic, I am so pleased."
            )
            await eval_js(f"""
                document.getElementById('rawConversationInput').value = {json.dumps(test_convo)};
                handleTextInputChange();
            """)
            await asyncio.sleep(1.0)  # Wait for debounce and fetch

            preview_displayed = await eval_js("document.getElementById('importPreviewPanel').style.display !== 'none'")
            preview_format = await eval_js("document.getElementById('previewFormat').textContent")
            preview_turns = await eval_js("document.getElementById('previewEvalTurns').textContent")
            preview_warnings_html = await eval_js("document.getElementById('importWarningsList').innerHTML")

            record_result(
                "Home / Import",
                "Live Import Preview with Warnings",
                "Preview displayed, plain text format, 3 evaluable turns, warning code rendered (not undefined)",
                f"display={preview_displayed}, format='{preview_format}', turns={preview_turns}",
                preview_displayed and "plain text" in str(preview_format).lower() and str(preview_turns) == "3" and "UNPARSED_LINE" in str(preview_warnings_html),
                f"Rendered warning HTML: {preview_warnings_html[:120]}..."
            )

            # Capture Screen 1 Screenshot
            s1_path = await take_screenshot("task32_screen1_import.png")
            print(f"   [Screenshot Saved] Screen 1 -> {s1_path}")

            # -------------------------------------------------------------
            # TEST 6: Execute Analysis -> Transition to Screen 2 (Viewer)
            # -------------------------------------------------------------
            await eval_js("executeAnalysis();")
            await asyncio.sleep(3.0)  # Wait for inference & dynamics

            active_screen = await eval_js("document.querySelector('.screen-container.active').id")
            bubble_count = await eval_js("document.querySelectorAll('.bubble-wrapper').length")
            first_emo = await eval_js("document.querySelector('.bubble-wrapper .emo-badge').textContent.trim()")

            record_result(
                "Conversation Viewer",
                "Inference Execution & Bubble Stream",
                "Active screen is 'screen-viewer', 3 message bubbles rendered with emotion badges",
                f"active_screen='{active_screen}', bubbles={bubble_count}, first_badge='{first_emo}'",
                active_screen == "screen-viewer" and bubble_count == 3 and "Predicted" in first_emo,
                f"First turn predicted: {first_emo}"
            )

            # Test Softmax Distribution Drawer
            await eval_js("toggleProbDrawer('prob-drawer-0');")
            prob_drawer_open = await eval_js("document.getElementById('prob-drawer-0').classList.contains('open')")
            prob_bar_count = await eval_js("document.querySelectorAll('#prob-drawer-0 .prob-bar-row').length")
            record_result(
                "Conversation Viewer",
                "Softmax Probability Drawer",
                "Drawer toggles open and renders 7 emotion probability tracks",
                f"open={prob_drawer_open}, tracks={prob_bar_count}",
                prob_drawer_open and prob_bar_count == 7,
                f"7-class probability tracks present in drawer"
            )

            # Capture Screen 2 Screenshot
            s2_path = await take_screenshot("task32_screen2_viewer.png")
            print(f"   [Screenshot Saved] Screen 2 -> {s2_path}")

            # -------------------------------------------------------------
            # TEST 7: Screen 3: Emotion Dynamics
            # -------------------------------------------------------------
            await eval_js("navigateTo('dynamics');")
            await asyncio.sleep(1.0)

            active_screen_dyn = await eval_js("document.querySelector('.screen-container.active').id")
            kpi_transitions = await eval_js("document.getElementById('dynTotalTransitions').textContent")
            chart_rendered = await eval_js("AppState.chartInstance !== null")
            seq_pills = await eval_js("document.querySelectorAll('.seq-pill').length")

            record_result(
                "Emotion Dynamics",
                "KPIs, Sequence Bar, & Chart.js Timeline",
                "Screen active, transitions > 0, 3 sequence pills, Chart.js active",
                f"active='{active_screen_dyn}', transitions={kpi_transitions}, pills={seq_pills}, chart_active={chart_rendered}",
                active_screen_dyn == "screen-dynamics" and int(kpi_transitions) > 0 and seq_pills == 3 and chart_rendered,
                f"Transitions: {kpi_transitions}, Chart.js instance active"
            )

            # Capture Screen 3 Screenshot
            s3_path = await take_screenshot("task32_screen3_dynamics.png")
            print(f"   [Screenshot Saved] Screen 3 -> {s3_path}")

            # -------------------------------------------------------------
            # TEST 8: Screen 4: Multi-Model Comparison
            # -------------------------------------------------------------
            await eval_js("executeComparison();")
            await asyncio.sleep(3.0)  # Wait for 3-model comparison

            active_screen_comp = await eval_js("document.querySelector('.screen-container.active').id")
            consensus_badge = await eval_js("document.getElementById('consensusRateBadge').textContent")
            matrix_cells = await eval_js("document.querySelectorAll('#pairwiseMatrixBody td.agreement-cell').length")
            comp_rows = await eval_js("document.querySelectorAll('.comparison-row').length")
            pred_cards = await eval_js("document.querySelectorAll('.model-pred-card').length")

            record_result(
                "Model Comparison",
                "Multi-Model Consensus & Agreement Matrix",
                "Screen active, consensus badge visible, 9 matrix cells (3x3), 3 comparison rows, 9 prediction cards",
                f"active='{active_screen_comp}', consensus='{consensus_badge}', matrix_cells={matrix_cells}, rows={comp_rows}, cards={pred_cards}",
                active_screen_comp == "screen-compare" and "Consensus Rate" in consensus_badge and matrix_cells == 9 and comp_rows == 3 and pred_cards == 9,
                f"Matrix size: {matrix_cells} cells, rows: {comp_rows}, prediction cards: {pred_cards}"
            )

            # Capture Screen 4 Screenshot
            s4_path = await take_screenshot("task32_screen4_comparison.png")
            print(f"   [Screenshot Saved] Screen 4 -> {s4_path}")

            # -------------------------------------------------------------
            # TEST 9: Screen 5: Research / Model Inspection
            # -------------------------------------------------------------
            await eval_js("navigateTo('research');")
            await asyncio.sleep(1.0)

            active_screen_res = await eval_js("document.querySelector('.screen-container.active').id")
            cohort_rows = await eval_js("document.querySelectorAll('#cohortATableBody tr').length")
            per_class_rows = await eval_js("document.querySelectorAll('#perClassTableBody tr').length")
            cm_cells = await eval_js("document.querySelectorAll('#cmContainer td').length")
            ethical_notice = await eval_js("document.body.innerText.includes('Research note: Non-Clinical Use & Ethical Safeguards')")

            record_result(
                "Research Inspection",
                "Cohort A Table, Per-Class Metrics, & Confusion Matrix",
                "Screen active, >= 3 benchmark rows, 7 per-class rows, 49 matrix cells (7x7), ethical notice",
                f"active='{active_screen_res}', cohort_rows={cohort_rows}, per_class={per_class_rows}, cm_cells={cm_cells}, notice={ethical_notice}",
                active_screen_res == "screen-research" and cohort_rows >= 3 and per_class_rows == 7 and cm_cells == 49 and ethical_notice,
                f"Cohort A rows: {cohort_rows}, Matrix cells: {cm_cells}, Ethical notice verified"
            )

            # Capture Screen 5 Screenshot
            s5_path = await take_screenshot("task32_screen5_research.png")
            print(f"   [Screenshot Saved] Screen 5 -> {s5_path}")

    finally:
        chrome_proc.terminate()
        chrome_proc.wait()

    print("\n==================================================================")
    print("BROWSER AUTOMATION VERIFICATION SUMMARY")
    print("==================================================================")
    passed_count = sum(1 for r in RESULTS if r["status"] == "PASS")
    total_count = len(RESULTS)
    print(f"Total Tests: {total_count} | Passed: {passed_count} | Failed: {total_count - passed_count}")

    # Output JSON summary for report inclusion
    summary_path = Path("outputs/tables/task32_browser_verification_results.json")
    summary_path.write_text(json.dumps(RESULTS, indent=2))
    print(f"Results written to: {summary_path}")

    if passed_count < total_count:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run_browser_verification())
