/**
 * RefreshStatusBar — pinned top bar showing refresh health and data freshness.
 * Shows morning/afternoon refresh status, model count, and any errors/QC warnings.
 */
import { useState } from "react";
import useSWR from "swr";
import { api, SystemStatus } from "@/lib/api";

function formatTime(iso?: string): string {
  if (!iso) return "Never";
  const utc = iso.endsWith("Z") ? iso : iso + "Z";
  return new Date(utc).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

function StatusDot({ status }: { status?: string }) {
  if (!status) return <span className="dot dot-grey" title="Not run yet" />;
  if (status === "completed") return <span className="dot dot-green" title="Completed OK" />;
  if (status === "partial") return <span className="dot dot-amber" title="Completed with warnings" />;
  return <span className="dot dot-red" title="Failed" />;
}

export default function RefreshStatusBar() {
  const { data: status } = useSWR<SystemStatus>("system-status", api.getSystemStatus, {
    refreshInterval: 60 * 1000, // poll every minute so errors appear quickly
  });

  const [expanded, setExpanded] = useState(false);

  const morning = status?.last_morning_refresh;
  const afternoon = status?.last_afternoon_refresh;
  const errors = status?.active_errors ?? [];
  const hasErrors = errors.length > 0;

  return (
    <div className="status-bar-wrap">
      <div className="status-bar">
        {/* Morning refresh */}
        <span className="status-item">
          <StatusDot status={morning?.status} />
          <span className="label">Morning:</span>{" "}
          <span className="value">{formatTime(morning?.ran_at)}</span>
          {morning?.games_updated != null && (
            <span className="sub"> · {morning.games_updated} games</span>
          )}
        </span>

        {/* Afternoon refresh */}
        <span className="status-item">
          <StatusDot status={afternoon?.status} />
          <span className="label">Afternoon:</span>{" "}
          <span className="value">{formatTime(afternoon?.ran_at)}</span>
          {afternoon?.lineups_confirmed != null && (
            <span className="sub"> · {afternoon.lineups_confirmed} lineups confirmed</span>
          )}
        </span>

        {/* Models */}
        {status && (
          <span className="status-item">
            <span className="dot dot-blue" />
            <span className="label">Models:</span>{" "}
            <span className="value">{status.active_model_versions.length} active</span>
          </span>
        )}

        {/* Error/warning toggle */}
        {hasErrors && (
          <button className="error-btn" onClick={() => setExpanded(v => !v)}>
            ⚠ {errors.length} issue{errors.length > 1 ? "s" : ""} {expanded ? "▲" : "▼"}
          </button>
        )}

        {!hasErrors && morning?.status === "completed" && (
          <span className="status-item ok">✓ All checks passed</span>
        )}
      </div>

      {/* Expanded error/QC details */}
      {expanded && hasErrors && (
        <div className="error-panel">
          <div className="error-panel-title">Issues from last refresh (includes QC warnings):</div>
          {errors.map((e, i) => (
            <div key={i} className="error-row">
              <span className="error-type">[{e.type}]</span>{" "}
              <span className="error-msg">{e.message}</span>
              {e.occurred_at && (
                <span className="error-time"> — {formatTime(e.occurred_at)}</span>
              )}
            </div>
          ))}
        </div>
      )}

      <style jsx>{`
        .status-bar-wrap { position: sticky; top: 0; z-index: 100; }
        .status-bar {
          display: flex;
          gap: 20px;
          align-items: center;
          background: #1a1a2e;
          color: #ccc;
          padding: 8px 16px;
          font-size: 12px;
          border-bottom: 1px solid #333;
          flex-wrap: wrap;
        }
        .status-item { display: flex; align-items: center; gap: 5px; }
        .label { color: #888; }
        .value { color: #fff; font-weight: 500; }
        .sub { color: #777; }
        .ok { color: #66bb6a; font-weight: 600; }
        .dot {
          width: 8px; height: 8px;
          border-radius: 50%;
          display: inline-block;
          flex-shrink: 0;
        }
        .dot-green  { background: #66bb6a; }
        .dot-amber  { background: #ffa726; }
        .dot-red    { background: #f44336; }
        .dot-grey   { background: #555; }
        .dot-blue   { background: #42a5f5; }
        .error-btn {
          background: none;
          border: 1px solid #f44336;
          color: #f44336;
          font-size: 12px;
          padding: 2px 8px;
          border-radius: 4px;
          cursor: pointer;
          margin-left: auto;
        }
        .error-btn:hover { background: rgba(244,67,54,0.1); }
        .error-panel {
          background: #1a0a0a;
          border-bottom: 1px solid #5c1a1a;
          padding: 10px 16px;
          font-size: 12px;
        }
        .error-panel-title { color: #f44336; font-weight: 600; margin-bottom: 6px; }
        .error-row { margin-bottom: 4px; line-height: 1.4; }
        .error-type { color: #ffa726; font-weight: 600; }
        .error-msg { color: #ccc; }
        .error-time { color: #666; }
      `}</style>
    </div>
  );
}
