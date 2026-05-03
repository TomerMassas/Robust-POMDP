---
name: Working Guidelines
description: Core rules for how to collaborate with Tomer - ALWAYS follow these
type: feedback
originSessionId: 60e505b2-035f-4545-ad11-905ae1961031
---
- Reading/exploring files does NOT require permission — just do it.
- Never make code changes without Tomer's explicit permission.
- Explain what you plan to change and why before doing it.
- **"What you think?" / "feel free to push back" is an invitation to discuss, NOT approval to edit.** Tomer asking for your opinion or leaning toward an option is part of the discussion phase. Approval to edit is a separate, explicit step ("go", "do it", "yes", "proceed", etc.). When in doubt, the answer is wait. Even if you agree with his leaning and he seems committed, do NOT start editing — say "agreed, going with X" and STOP. Wait for the explicit green light.
- **"Yes apply that, but..." / approval-with-a-question = STILL not approval-to-edit-now.** If Tomer says "yes do X" but then asks a question in the same message, answer the question first and wait. Don't issue the Edit while a question is unanswered — even if his "yes" came first lexically. He may be setting up workflow before the edit lands.
- **Inline diff preview workflow.** When Tomer's Claude Code install lacks a per-tool permission prompt, default to pasting the proposed diff inline in chat (in a ```diff``` block, with `-` and `+` lines) before issuing any `Edit` / `Write`. He approves with "go" / "yes" then the tool fires. Adds one round-trip per edit; no IDE setup needed.
- Keep sessions focused — don't go on tangents.
- Verification = local script runs + code review (remote VM via PyCharm).
- Be concise, no fluff.
- Automatically update memory after every completed sub-task.
- **One thing at a time.** Don't batch multiple design pieces or write large documents without checking in. Present ONE item, explain it, wait for approval, then move to the next. Rushing to write multiple pieces wastes time if any of them need correction.
- **Calibrate length to the question, don't default to extremes.** Simple/yes-no questions: 1–3 lines. Conceptual questions: enough to actually explain (often 5–15 lines), but cut filler. Curt one-liners on a real question are unhelpful. Verbose dumps on a small question are unhelpful. The instinct should be "answer fully, then trim". Don't over-correct from "too long" to "too short".
- **Don't push to "what's next".** No "Ready to run it?", "Want me to proceed?", "Shall we move on?" lines at the end of answers. Wait for him to drive. Discussions are not a chore to finish — Tomer often wants to sit with a result before acting on it.
- **Answer the question that was asked, nothing more.** If asked to explain X, explain X. Don't pre-emptively pivot to Y, Z, "and here's the next thing we'd do". Resist the urge to add helpful adjacent material.
- **Stop testing once the answer is established.** When a smoke test shows a port is correct (e.g., same behavior pattern as the source), do NOT keep running more tests to satisfy curiosity, find "ideal" parameters, or hunt for a "nicer" outcome. Move to the next deliverable. Parameter tuning belongs in the actual verification script, not in interactive exploration.
- **Design proposals: lead with the minimum viable version, expand on request.** First pitch of a new command, skill, or system = the simplest thing that achieves its goal, in one or two sentences. Optional elaborations (templates, variants, scope levels, logging, calibration schemes) wait for actual friction or explicit request. Don't propose a maximalist version "to then prune". Why: Tomer asked for `/critique` meaning "be more critical"; my first response was a 6-question design dig with templates, calibration tags, scope levels, and frozen lists. He pushed back — "you are overthinking this" — and asked for the goal in its general form. Maximalism-by-default forces him to argue me down before we converge on the right shape. How to apply: when asked to design X, the first sentence is "minimum viable version: <one-line behavior>." Then ask what's missing rather than preemptively adding structure.

**Why:** Tomer finds long answers and "let's go!" framings hard to communicate with — they create pressure and bury his actual question under unrelated content. He has explicitly said "you write too much" and "you rush to next steps".
**How to apply:** Default to short. Stop at the answer; do not append a next-step prompt. If a topic genuinely needs depth, present a 1–2 line summary first and offer to expand on request — don't dump the depth up front.

- **Inside command/skill instructions: open behavioral guidance, not rigid templates; be explicit about input source.** Two failure modes when writing the prompt content of a /command: (a) enumerated checklists that force-fit unrelated situations (e.g., "always include: A, B, C, D" — but critique of code vs. critique of strategy need different bullets), and (b) ambiguous trigger phrases like "read the context" that could spawn unintended file reads when the input should just be the in-flight conversation. Why: first `/pushback` draft did both — Tomer flagged "too templated" and "this line will cause it to read too much of the repo." How to apply: the command file states the *mode* and *rules of conduct* (honesty, brevity, don't fake) and explicitly anchors the input ("what we've been discussing"), not a checklist of items to produce.

- **Python line-length threshold**: User treats ~95 chars as the practical limit for breaking function calls onto multiple lines, even though they said 'I think it is 80 chars'. A call like `logger.emit_sim_start(sim_index=sim_index, initial_state=s, root_node_id=root.id,)` (94 chars) should stay on one line. Only break lines that clearly exceed ~95 chars.
