# Retry & Cooldown Logic — OpenRouter Better Proxy

This document explains the two provider-handling mechanisms in the OpenRouter
Better Proxy: **retry** and **backoff/cooldown**. They are often confused, and
the two overlap in a way that is easy to misread, so this is a precise, code-accurate
description with concrete end-to-end examples.

It is the reference that a future user-facing guide (README / INSTALLATION) will be
built from.

---

## 1. The two mechanisms are independent

| | **Retry** | **Backoff / Cooldown** |
|---|---|---|
| Where it lives | `routes.py` | `backoff.py` + `health.*` in config |
| Scope | **inside one single request**, on the **same provider** | **across multiple requests**, per provider |
| Config key | `retry.max_attempts`, `retry.delay_seconds` | `health.initial_cooldown_seconds`, `health.consecutive_threshold`, `health.escalation_seconds`, `health.max_cooldown_seconds` |
| Counter | `max_attempts` (in-request) | `consecutive_threshold` (across requests) |
| Decides | how many times the proxy tries the **same** provider within this request | for **how long** the provider is excluded on **future** requests |
| Triggers on | transient pre-stream error (429, 5xx, timeout, connect) | `mark_error` → cooldown window that escalates after `consecutive_threshold` errors |
| Effect on the other | retry does **not** touch the cooldown | cooldown does **not** perform retries |

### The bridge: `mark_error`

The only place the two connect is `mark_error`. With the retry feature:

- `mark_error` (which arms the cooldown window) is called **only after a
  provider has exhausted all its retries** for the current request.
- A provider that recovers inside the retry window is **never** put into cooldown.

### `consecutive_threshold` is NOT a retry counter

`consecutive_threshold` does not mean "how many retries". It is the threshold of
**cumulative errors** beyond which the cooldown duration escalates:

- 1st error → `initial_cooldown_seconds` (default 300 s)
- reaching `consecutive_threshold` (default 4) errors → longer cooldown
  (`escalation_seconds`, e.g. 3600 s, then 43200 s)
- capped at `max_cooldown_seconds` (default 43200 s)
- a success resets the error count

So `retry.max_attempts` (in-request, same provider) and `health.consecutive_threshold`
(across-request, escalates duration) are two fully independent numbers.

---

## 2. Configuration

```yaml
retry:
    max_attempts: 3        # total attempts per provider (3 = up to 2 retries)
    delay_seconds: 2.5     # fixed delay between retry attempts

health:
    initial_cooldown_seconds: 300
    consecutive_threshold: 4
    escalation_seconds: [3600, 43200]
    max_cooldown_seconds: 43200
```

- `max_attempts` is the **total** number of attempts for one provider in one request
  (`1` = no retry). Retries = `max_attempts - 1`.
- `delay_seconds` is a **fixed** wait between attempts.
- Both values are editable in the dashboard (Proxy Settings → "🔁 Retry").
- No `retry` section → behaves as `max_attempts: 1` (current single-attempt behaviour).

### Which errors are retried?

Only **transient pre-stream** errors:

- HTTP **429** and **5xx**
- connection errors, timeouts, httpx network errors

Definitive errors (**other 4xx** such as 400/404/422) are **not** retried — retrying a
bad request is pointless; the proxy fails over immediately. **Mid-stream** failures
(after the 200 + first chunk) are never retried either: the stream cannot be revoked
(DESIGN §4), so the proxy tears the stream down and reports the error.

---

## 3. Reference flows

Setup used in all examples: `retry.max_attempts: 3`, `retry.delay_seconds: 2.5`,
and 4 authorized providers in priority order `[P1, P2, P3, P4]`.

### Example A — P1 → 429, P2 → 429, P3 → 200 (all retries on P1/P2 exhausted)

**Request 1** (Hermes sends a chat turn):

```
select_candidates → [P1, P2, P3, P4]

├─ P1:  attempt 1 → 429 (transient)
│       wait 2.5s
│       attempt 2 → 429
│       wait 2.5s
│       attempt 3 → 429
│       retries exhausted → mark_error(P1) → cooldown 300s   (ONE error counted)
│
├─ P2:  same sequence → three 429 → mark_error(P2) → cooldown 300s
│
├─ P3:  attempt 1 → 200 ✅
│       record_success(P3) → session bound to P3
│       response returned to Hermes
│       (P4 never attempted)
└─ Hermes receives 200 ← one HTTP round-trip from the client's point of view
```

- Total upstream calls: **7** (3 + 3 + 1), with 4 waits of 2.5 s, all internal.
- Hermes sees **one** HTTP request/response; the failover is transparent.
- P1 and P2 each accumulate **1 error** → cooldown **starts after** this request.

**Request 2** (next turn of the same conversation — session is bound to P3):

```
select_candidates(session=…):
  • sticky = P3, healthy → P3 placed FIRST
  • P1 and P2 in cooldown (300 s not expired) → EXCLUDED
candidates → [P3, P4]
→ goes straight to P3 → 200 ✅
```

- P1/P2 are **not** retried while their cooldown is active.
- Even after the 300 s expire, within this session P1/P2 do **not** come back,
  because sticky routing keeps P3 first while it is healthy. Cooldown only matters
  outside the session, or if P3 should fail and the session re-selects.

### Example B — P1 answers on the 2nd retry

**Request 1**:

```
select_candidates → [P1, P2, P3, P4]

├─ P1:  attempt 1 → 429 (transient)
│       wait 2.5s
│       attempt 2 → 200 ✅
│       record_success(P1)  ← mark_error NOT called → NO cooldown
│       session bound to P1
│       response returned
│       (P2, P3, P4 never attempted)
└─ Hermes receives 200 on the first HTTP request
```

**Request 2** (same session):

```
sticky = P1, healthy → [P1, P2, P3, P4] with P1 first
→ P1 → 200 ✅ (no cooldown, nothing excluded)
```

---

## 4. Comparison summary

| Outcome for P1 | Cooldown for P1? | Providers visited in request 1 | Request 2 |
|---|---|---|---|
| 429 ×3 (Example A) | ✅ yes, 300 s (after the 3 attempts) | P1, P2, P3 | P1/P2 excluded → P3 (sticky) |
| 429, then 200 (Example B) | ❌ never | P1 (two attempts) | P1 (sticky) |

---

## 5. Key takeaways

1. **Retry** = retry the *same* provider immediately; **cooldown** = exclude the
   provider on *future* requests after it failed all retries of one request.
2. `consecutive_threshold` is about how long the cooldown lasts, **not** how many times
   a provider is retried.
3. A provider that recovers inside the retry window never enters cooldown.
4. The proxy only ever tries **authorized** providers (strict allowlist), retries
   only **same**-provider transients, and fails over only after retries are exhausted.
5. From Hermes'/the client's perspective the whole retry+fallback cascade is a single
   HTTP round-trip; the client only sees an error if **every** authorized candidate
   has exhausted its retries.

---

*Internal implementation reference: `routes.py` (`_forward_stream`, `_forward_non_stream`),
`backoff.py`, `config.py`, and the `retry` / `health` config sections.*
