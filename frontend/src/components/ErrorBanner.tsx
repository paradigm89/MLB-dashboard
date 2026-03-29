/**
 * ErrorBanner — displays red (blocking) or yellow (degraded) banners
 * when the system has refresh failures or data quality issues.
 */
interface ErrorBannerProps {
  errors: Array<{ type: string; message: string; occurred_at?: string }>;
  lastSuccessfulRefresh?: string;
}

export default function ErrorBanner({ errors, lastSuccessfulRefresh }: ErrorBannerProps) {
  if (!errors || errors.length === 0) return null;

  const hasRefreshFailure = errors.some((e) => e.type === "morning" || e.type === "afternoon");
  const isBlocking = hasRefreshFailure;

  return (
    <div className={`banner ${isBlocking ? "blocking" : "degraded"}`}>
      <strong>{isBlocking ? "⛔ Data Refresh Failed" : "⚠ Degraded Data"}</strong>
      <ul>
        {errors.map((err, i) => (
          <li key={i}>
            <span className="err-type">[{err.type}]</span> {err.message}
          </li>
        ))}
      </ul>
      {isBlocking && lastSuccessfulRefresh && (
        <p className="note">
          Last successful refresh: {new Date(lastSuccessfulRefresh).toLocaleString()}
        </p>
      )}

      <style jsx>{`
        .banner {
          padding: 12px 16px;
          border-radius: 6px;
          margin: 12px 16px;
          font-size: 14px;
        }
        .blocking { background: #ff1744; color: #fff; }
        .degraded { background: #ff6f00; color: #fff; }
        ul { margin: 6px 0 0 16px; padding: 0; }
        li { margin-bottom: 4px; }
        .err-type { font-weight: 700; opacity: 0.8; }
        .note { margin: 8px 0 0; font-size: 12px; opacity: 0.85; }
      `}</style>
    </div>
  );
}
