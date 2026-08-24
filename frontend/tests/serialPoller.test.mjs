import assert from "node:assert/strict";
import test from "node:test";

import { startSerialPoller } from "../src/serialPoller.ts";

const flushMicrotasks = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

test("serial poller never overlaps a slow task and can be stopped", async () => {
  const scheduled = [];
  const cancelled = [];
  let resolveFirst;
  let calls = 0;
  const first = new Promise(resolve => { resolveFirst = resolve; });
  const task = () => {
    calls += 1;
    return calls === 1 ? first : Promise.resolve();
  };
  const schedule = (callback, delayMs) => {
    const handle = { callback, delayMs };
    scheduled.push(handle);
    return handle;
  };
  const cancel = handle => cancelled.push(handle);

  const stop = startSerialPoller(task, 3000, undefined, schedule, cancel);
  assert.equal(calls, 1);
  assert.equal(scheduled.length, 0, "no next poll is scheduled while the first request is pending");

  resolveFirst();
  await flushMicrotasks();
  assert.equal(scheduled.length, 1);
  assert.equal(scheduled[0].delayMs, 3000);

  scheduled.shift().callback();
  await flushMicrotasks();
  assert.equal(calls, 2);
  assert.equal(scheduled.length, 1);

  const pendingTimer = scheduled[0];
  stop();
  assert.deepEqual(cancelled, [pendingTimer]);
});

test("serial poller reports a failure and continues scheduling", async () => {
  const scheduled = [];
  const errors = [];
  const expected = new Error("temporary failure");
  const stop = startSerialPoller(
    () => Promise.reject(expected),
    100,
    error => errors.push(error),
    callback => { scheduled.push(callback); return callback; },
    () => undefined,
  );
  await flushMicrotasks();
  assert.deepEqual(errors, [expected]);
  assert.equal(scheduled.length, 1);
  stop();
});
