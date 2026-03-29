/**
 * RefreshStatusBar — pinned top bar showing data freshness.
 * Shows last refresh time, lineup confirmation count, and error count.
 * Polls /api/system/status every 5 minutes.
 */
import useSWR from "swr";
import { api, SystemStatus } from "@/lib/api";

function formatTime(iso?: string): string {
  if (!iso) return "Never";
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZoneName: "short" });
}

interface Props {
  lineupsConfirmed: number;
  lineupsProjected: number;
  lastRefreshed?: string;
}

export default function RefreshStatusBar({ lineupsConfirmed, lineupsProjected, lastRefreshed }: Props) {
  const { data: status } = useSWR<SystemStatus>("system-status", api.getSystemStatus, {
    refreshInterval: 5 * 60 * 1000,
  });

  const hasErrors = (status?.active_errors?.length ?? 0) > 0;
  const totalLineups = lineupsConfirmed + lineupsProjected;

  return (
    <div className="status-bar">
      <span className="status-item">
        <span className="label">Last updated:</span>{" "}
        <span className="value">{formatTime(lastRefreshed)}</span>
      </span>

      <span className="status-item">
        <span className="label">Lineups:</span>{" "}
        <span className="confirmed">{lineupsConfirmed} confirmed</span>
        {lineupsProjected > 0 && (
          <span className="projected">, {lineupsProjected} projected</span>
        )}
      </span>

      {hasErrors && (
        <span className="status-item error">
          ⚠ {status!.active_errors.length} error{status!.active_errors.length > 1 ? "s" : ""}
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
        .confirmed { color: #4caf50; font-weight: 600; }
        .projected { color: #ff9800; }
        .error { color: #f44336; font-weight: 600; }
      `}</style>
    </div>
  );
}
