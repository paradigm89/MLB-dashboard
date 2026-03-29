/**
 * Player Spotlight Page (/player/[id])
 * Shows career/season stats, Statcast profile, and xwOBA by pitch type.
 */
import { useRouter } from "next/router";
import useSWR from "swr";
import Head from "next/head";
import Link from "next/link";
import RefreshStatusBar from "@/components/RefreshStatusBar";
import { api } from "@/lib/api";

type StatType = "batting" | "pitching";

interface BattingStats {
  player_id: number;
  player_name?: string;
  season: number;
  pa?: number;
  avg?: number;
  obp?: number;
  slg?: number;
  ops?: number;
  woba?: number;
  xwoba?: number;
  xba?: number;
  xslg?: number;
  hard_hit_pct?: number;
  barrel_pct?: number;
  k_pct?: number;
  bb_pct?: number;
  wrc_plus?: number;
  war?: number;
  exit_velocity?: number;
  launch_angle?: number;
}

interface PitchingStats {
  player_id: number;
  player_name?: string;
  season: number;
  g?: number;
  gs?: number;
  ip?: number;
  era?: number;
  fip?: number;
  xfip?: number;
  xera?: number;
  k_pct?: number;
  bb_pct?: number;
  k_bb_pct?: number;
  hr9?: number;
  whip?: number;
  war?: number;
  avg_fastball_velo?: number;
  swstr_pct?: number;
}

function fmt(v: number | undefined, dec = 3): string {
  if (v == null) return "—";
  return v.toFixed(dec);
}

function fmtPct(v: number | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export default function PlayerPage() {
  const router = useRouter();
  const { id, type } = router.query;
  const playerId = id ? Number(id) : null;
  const statType: StatType = (type as StatType) === "pitching" ? "pitching" : "batting";

  const { data, error, isLoading } = useSWR(
    playerId ? `stats/player/${playerId}/${statType}` : null,
    () => api.getPlayerStats(playerId!, statType)
  );

  const stats = data as BattingStats | PitchingStats | undefined;
  const playerName =
    (stats as BattingStats)?.player_name ?? `Player ${playerId}`;

  return (
    <>
      <Head>
        <title>{playerName} — MLB Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <RefreshStatusBar />

      <main className="page-container">
        <div className="back-link">
          <Link href="/">← Today&apos;s Games</Link>
        </div>

        {isLoading && <div className="state-message">Loading player data…</div>}
        {error && !isLoading && (
          <div className="state-message error">
            Could not load stats for player {playerId}.
          </div>
        )}

        {stats && (
          <>
            <div className="player-header">
              <h1>{playerName}</h1>
              <div className="type-toggle">
                <Link
                  href={`/player/${playerId}?type=batting`}
                  className={statType === "batting" ? "active" : ""}
                >
                  Batting
                </Link>
                <Link
                  href={`/player/${playerId}?type=pitching`}
                  className={statType === "pitching" ? "active" : ""}
                >
                  Pitching
                </Link>
              </div>
            </div>

            {statType === "batting"
              ? <BattingPanel stats={stats as BattingStats} />
              : <PitchingPanel stats={stats as PitchingStats} />
            }
          </>
        )}
      </main>

      <style jsx>{`
        .back-link { margin: 12px 0 20px; }
        .back-link a { color: #64b5f6; font-size: 13px; text-decoration: none; }
        .back-link a:hover { text-decoration: underline; }
        .player-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 20px;
        }
        .type-toggle { display: flex; gap: 8px; }
        .type-toggle a {
          padding: 6px 14px;
          border-radius: 6px;
          font-size: 13px;
          font-weight: 600;
          color: #888;
          text-decoration: none;
          border: 1px solid #21262d;
        }
        .type-toggle a.active { background: #1565c0; color: #fff; border-color: #1565c0; }
        .state-message { margin-top: 40px; text-align: center; color: #666; font-size: 15px; }
        .state-message.error { color: #ef5350; }
      `}</style>
    </>
  );
}

function StatPanel({ title, rows }: { title: string; rows: Array<[string, string]> }) {
  return (
    <div className="panel">
      <h2>{title}</h2>
      <dl className="stat-grid">
        {rows.map(([label, value]) => (
          <div key={label} className="stat-row">
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>

      <style jsx>{`
        .panel {
          background: #16213e;
          border: 1px solid #21262d;
          border-radius: 10px;
          padding: 20px;
          margin-bottom: 16px;
        }
        .panel h2 { margin-bottom: 14px; }
        .stat-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
          gap: 12px;
        }
        .stat-row { display: flex; flex-direction: column; }
        dt { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
        dd { font-size: 16px; font-weight: 600; color: #e6edf3; margin: 0; }
      `}</style>
    </div>
  );
}

function BattingPanel({ stats: s }: { stats: BattingStats }) {
  return (
    <>
      <StatPanel
        title="Standard Batting"
        rows={[
          ["Season", String(s.season ?? "—")],
          ["PA", String(s.pa ?? "—")],
          ["AVG", fmt(s.avg)],
          ["OBP", fmt(s.obp)],
          ["SLG", fmt(s.slg)],
          ["OPS", fmt(s.ops)],
          ["wRC+", s.wrc_plus != null ? String(s.wrc_plus) : "—"],
          ["WAR", fmt(s.war, 1)],
        ]}
      />
      <StatPanel
        title="Statcast / Expected Stats"
        rows={[
          ["xwOBA", fmt(s.xwoba)],
          ["wOBA", fmt(s.woba)],
          ["xBA", fmt(s.xba)],
          ["xSLG", fmt(s.xslg)],
          ["Hard Hit%", fmtPct(s.hard_hit_pct)],
          ["Barrel%", fmtPct(s.barrel_pct)],
          ["Exit Velo", s.exit_velocity != null ? `${s.exit_velocity.toFixed(1)} mph` : "—"],
          ["Launch Angle", s.launch_angle != null ? `${s.launch_angle.toFixed(1)}°` : "—"],
        ]}
      />
      <StatPanel
        title="Plate Discipline"
        rows={[
          ["K%", fmtPct(s.k_pct)],
          ["BB%", fmtPct(s.bb_pct)],
        ]}
      />
    </>
  );
}

function PitchingPanel({ stats: s }: { stats: PitchingStats }) {
  return (
    <>
      <StatPanel
        title="Standard Pitching"
        rows={[
          ["Season", String(s.season ?? "—")],
          ["G / GS", `${s.g ?? "—"} / ${s.gs ?? "—"}`],
          ["IP", fmt(s.ip, 1)],
          ["ERA", fmt(s.era, 2)],
          ["WHIP", fmt(s.whip, 2)],
          ["WAR", fmt(s.war, 1)],
        ]}
      />
      <StatPanel
        title="Advanced / Expected"
        rows={[
          ["FIP", fmt(s.fip, 2)],
          ["xFIP", fmt(s.xfip, 2)],
          ["xERA", fmt(s.xera, 2)],
          ["K%", fmtPct(s.k_pct)],
          ["BB%", fmtPct(s.bb_pct)],
          ["K-BB%", fmtPct(s.k_bb_pct)],
          ["HR/9", fmt(s.hr9, 2)],
          ["SwStr%", fmtPct(s.swstr_pct)],
        ]}
      />
      <StatPanel
        title="Stuff"
        rows={[
          ["Avg FB Velo", s.avg_fastball_velo != null ? `${s.avg_fastball_velo.toFixed(1)} mph` : "—"],
        ]}
      />
    </>
  );
}
