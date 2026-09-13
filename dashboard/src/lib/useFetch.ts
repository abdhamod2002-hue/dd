import { useEffect, useState } from "react";

/** Minimal data-fetching hook with loading + error states and optional polling interval. */
export function useFetch<T>(fetcher: () => Promise<T>, deps: unknown[] = [], pollIntervalMs: number = 0) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const refetch = () => setTick((t) => t + 1);

  useEffect(() => {
    let alive = true;
    
    const execute = (isInitial = false) => {
      if (isInitial) setLoading(true);
      fetcher()
        .then((d) => {
          if (alive) {
            setData(d);
            setError(null);
          }
        })
        .catch((e) => {
          if (alive) setError(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          if (alive && isInitial) setLoading(false);
        });
    };

    execute(true);

    let intervalId: any = null;
    if (pollIntervalMs > 0) {
      intervalId = setInterval(() => {
        execute(false);
      }, pollIntervalMs);
    }

    return () => {
      alive = false;
      if (intervalId) clearInterval(intervalId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick, pollIntervalMs]);

  return { data, loading, error, refetch };
}
