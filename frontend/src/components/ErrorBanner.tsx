/**
 * ErrorBanner — displays red (blocking) or yellow (degraded) banners
 * when the system has refresh failures. Fetches its own data via SWR.
 */
import useSWR from "swr";
import { api, SystemStatus } from "@/lib/api";

export default function ErrorBanner() {
  const { data: status } = useSWR<SystemStatus>("system-status", api.getSystemStatus, {
    refreshInterval: 5 * 60 * 1000,
  });

  const errors = status?.active_errors ?? [];
  if (errors.length === 0) return null;

  const hasRefreshFailure = errors.some((e) => e.type === "morning" || e.type === "afternoon");
  const isBlocking = hasRefreshFailure;
  const lastRefreshed = status?.last_morning_refresh?.ran_at;

  return (
    <div className={`banner ${isBlocking ? "blocking" : "degraded"}`}>
      <strong>{isBlocking ? "Data Refresh Failed" : "Degraded Data"}</strong>
      <ul>
        {errors.map((err, i) => (
          <li key={i}>
            <span className="err-type">[{err.type}]</span> {err.message}
          </li>
        ))}
      </ul>
      {isBlocking && lastRefreshed && (
        <p className="note">
          Last successful refresh: {new Date(lastRefreshed).toLocaleString()}
        </p>
      )}

      <style jsx>{`
        .banner {
          padding: 12px 16px;
          border-radius: 6px;
          margin: 12px 0;
          font-size: 14px;
        }
        .blocking { background: #c62828; color: #fff; }
        .degraded { background: #e65100; color: #fff; }
        ul { margin: 6px 0 0 16px; padding: 0; }
        li { margin-bottom: 4px; }
        .err-type { font-weight: 700; opacity: 0.8; }
        .note { margin: 8px 0 0; font-size: 12px; opacity: 0.85; }
      `}</style>
    </div>
  );
}
