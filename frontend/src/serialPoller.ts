type Schedule = (callback: () => void, delayMs: number) => unknown;
type Cancel = (handle: unknown) => void;

export function startSerialPoller(
  task: () => Promise<unknown>,
  delayMs: number,
  onError?: (error: unknown) => void,
  schedule: Schedule = (callback, delay) => window.setTimeout(callback, delay),
  cancel: Cancel = handle => window.clearTimeout(handle as number),
) {
  let stopped = false;
  let timer: unknown;

  const run = async () => {
    try {
      await task();
    } catch (error) {
      onError?.(error);
    } finally {
      if (!stopped) timer = schedule(() => void run(), delayMs);
    }
  };

  void run();
  return () => {
    stopped = true;
    if (timer !== undefined) cancel(timer);
  };
}
