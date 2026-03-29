/**
 * RefreshStatusBar — pinned top bar showing data freshness.
 * Fetches its own system status via SWR; no props required.
 */
import useSWR from "swr";
import { api, SystemStatus } from "@/lib/api";

function formatTime(iso?: string): string {
  if (!iso) return "Never";
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZoneName: "short" });
}

export default function RefreshStatusBar() {
  const { data: status } = useSWR<SystemStatus>("system-status", api.getSystemStatus, {
    refreshInterval: 5 * 60 * 1000,
  });

  const hasErrors = (status?.active_errors?.length ?? 0) > 0;
  const lastRefreshed = status?.last_morning_refresh?.ran_at;

  return (
    <div className="status-bar">
      <span className="status-item">
        <span className="label">Last updated:</span>{" "}
        <span className="value">{formatTime(lastRefreshed)}</span>
      </span>

      {status && (
        <span className="status-item">
          <span className="label">Models:</span>{" "}
          <span className="value">{status.active_model_versions.length} active</span>
        </span>
      )}

      {hasErrors && (
        <span className="status-item error">
          &#9888; {status!.active_errors.length} error{status!.active_errors.length > 1 ? "s" : ""}
        </span>
      )}

      <style jsx>{`
        .status-bar {
          display: flex;
          gap: 24px;
          align-items: center;
          background: #1a1a2e;
          color: #ccc;
          padding: 8px 16px;
          font-size: 13px;
          border-bottom: 1px solid #333;
          flex-wrap: wrap;
        }
        .label { color: #888; }
        .value { color: #fff; font-weight: 500; }
        .error { color: #f44336; font-weight: 600; }
      `}</style>
    </div>
  );
}
