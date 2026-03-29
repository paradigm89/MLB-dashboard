/**
 * Game Detail Page (/game/[id])
 * Shows a full breakdown of ML predictions for one game:
 *   - Win probability bar
 *   - Run distribution chart (P10 / median / P90)
 *   - Pitcher matchup projections
 *   - Batter matchup tables for both lineups
 */
import { useRouter } from "next/router";
import useSWR from "swr";
import Head from "next/head";
import Link from "next/link";
import WinProbBar from "@/components/WinProbBar";
import RunDistributionChart from "@/components/RunDistributionChart";
import BatterMatchupTable from "@/components/BatterMatchupTable";
import RefreshStatusBar from "@/components/RefreshStatusBar";
import { api, GamePrediction } from "@/lib/api";

const FIVE_MINUTES = 5 * 60 * 1000;

function formatPct(v?: number) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function formatNum(v?: number, decimals = 2) {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

export default function GameDetailPage() {
  const router = useRouter();
  const { id } = router.query;
  const gamePk = id ? Number(id) : null;

  const { data: pred, error, isLoading } = useSWR<GamePrediction>(
    gamePk ? `predict/game/${gamePk}` : null,
    () => api.getGamePrediction(gamePk!),
    { refreshInterval: FIVE_MINUTES }
  );

  const title = pred
    ? `${pred.away_team.team_abbr} @ ${pred.home_team.team_abbr} — MLB Dashboard`
    : "Game Detail — MLB Dashboard";

  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <RefreshStatusBar />

      <main className="page-container">
        <div className="back-link">
          <Link href="/">← Today&apos;s Games</Link>
        </div>

        {isLoading && <div className="state-message">Loading game data…</div>}
        {error && !isLoading && (
          <div className="state-message error">
            Could not load prediction for game {gamePk}.
          </div>
        )}

        {pred && (
          <>
            {/* ── Header ── */}
            <div className="game-header">
              <h1 className="matchup-title">
                <span className="away">{pred.away_team.team_abbr}</span>
                <span className="at"> @ </span>
                <span className="home">{pred.home_team.team_abbr}</span>
              </h1>
              {pred.prediction_generated_at && (
                <span className="generated-at">
                  Prediction generated{" "}
                  {new Date(pred.prediction_generated_at).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              )}
            </div>

            {/* ── Lineup confirmation badges ── */}
            <div className="badges">
              <span className={`tag ${pred.away_lineup_confirmed ? "tag-green" : "tag-amber"}`}>
                {pred.away_team.team_abbr} lineup {pred.away_lineup_confirmed ? "confirmed" : "projected"}
              </span>
              <span className={`tag ${pred.home_lineup_confirmed ? "tag-green" : "tag-amber"}`} style={{ marginLeft: 8 }}>
                {pred.home_team.team_abbr} lineup {pred.home_lineup_confirmed ? "confirmed" : "projected"}
              </span>
              <span className="tag tag-blue" style={{ marginLeft: 8 }}>
                {pred.simulation_method}
              </span>
            </div>

            {/* ── Win probability ── */}
            <section className="panel">
              <h2>Win Probability</h2>
              <WinProbBar
                awayAbbr={pred.away_team.team_abbr}
                homeAbbr={pred.home_team.team_abbr}
                winProbHome={pred.win_prob_home}
                winProbAway={pred.win_prob_away}
                homeConfirmed={pred.home_lineup_confirmed}
                awayConfirmed={pred.away_lineup_confirmed}
                generatedAt={pred.prediction_generated_at}
              />
            </section>

            {/* ── Run distribution ── */}
            <section className="panel">
              <h2>Expected Run Totals</h2>
              <RunDistributionChart
                homeAbbr={pred.home_team.team_abbr}
                awayAbbr={pred.away_team.team_abbr}
                homeRuns={pred.runs_home}
                awayRuns={pred.runs_away}
              />
            </section>

            {/* ── Pitcher matchup ── */}
            <section className="panel">
              <h2>Starting Pitcher Matchup</h2>
              <div className="pitcher-grid">
                <PitcherCard
                  sp={pred.away_sp}
                  teamAbbr={pred.away_team.team_abbr}
                  side="Away"
                />
                <PitcherCard
                  sp={pred.home_sp}
                  teamAbbr={pred.home_team.team_abbr}
                  side="Home"
                />
              </div>
            </section>

            {/* ── Lineup edge ── */}
            {pred.lineup_edge != null && (
              <section className="panel">
                <h2>Lineup Edge</h2>
                <div className="edge-row">
                  {pred.lineup_edge > 0 ? (
                    <>
                      <strong>{pred.home_team.team_abbr}</strong> has the lineup advantage
                      &nbsp;(+{Math.abs(pred.lineup_edge).toFixed(3)} xwOBA)
                    </>
                  ) : pred.lineup_edge < 0 ? (
                    <>
                      <strong>{pred.away_team.team_abbr}</strong> has the lineup advantage
                      &nbsp;(+{Math.abs(pred.lineup_edge).toFixed(3)} xwOBA)
                    </>
                  ) : (
                    "Lineups are evenly matched"
                  )}
                </div>
                <div className="xwoba-row">
                  <span>{pred.away_team.team_abbr} lineup xwOBA: {formatNum(pred.away_lineup_xwoba, 3)}</span>
                  <span>{pred.home_team.team_abbr} lineup xwOBA: {formatNum(pred.home_lineup_xwoba, 3)}</span>
                </div>
              </section>
            )}

            {/* ── Batter matchup tables ── */}
            <section className="panel batter-section">
              <h2>Batter Matchups</h2>
              <div className="batter-tables">
                <BatterMatchupTable
                  batters={pred.away_batter_matchups}
                  teamAbbr={pred.away_team.team_abbr}
                  vsSpName={pred.home_sp?.pitcher_name ?? (pred.home_sp ? `#${pred.home_sp.pitcher_id}` : undefined)}
                />
                <BatterMatchupTable
                  batters={pred.home_batter_matchups}
                  teamAbbr={pred.home_team.team_abbr}
                  vsSpName={pred.away_sp?.pitcher_name ?? (pred.away_sp ? `#${pred.away_sp.pitcher_id}` : undefined)}
                />
              </div>
            </section>

            {/* ── Model metadata ── */}
            <section className="panel metadata">
              <h2>Prediction Details</h2>
              <dl className="meta-grid">
                <dt>Home win probability</dt>
                <dd>{formatPct(pred.win_prob_home)}</dd>
                <dt>Away win probability</dt>
                <dd>{formatPct(pred.win_prob_away)}</dd>
                <dt>Home median runs</dt>
                <dd>{formatNum(pred.runs_home?.median, 1)}</dd>
                <dt>Away median runs</dt>
                <dd>{formatNum(pred.runs_away?.median, 1)}</dd>
                <dt>Method</dt>
                <dd>{pred.simulation_method}</dd>
              </dl>
            </section>
          </>
        )}
      </main>

      <style jsx>{`
        .back-link { margin: 12px 0 20px; }
        .back-link a { color: #64b5f6; font-size: 13px; text-decoration: none; }
        .back-link a:hover { text-decoration: underline; }
        .game-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 12px; }
        .matchup-title { font-size: 28px; font-weight: 700; }
        .away { color: #ef5350; }
        .home { color: #64b5f6; }
        .at { color: #666; }
        .generated-at { font-size: 12px; color: #666; }
        .badges { display: flex; flex-wrap: wrap; gap: 0; margin-bottom: 20px; }
        .panel {
          background: #16213e;
          border: 1px solid #21262d;
          border-radius: 10px;
          padding: 20px;
          margin-bottom: 16px;
        }
        .panel h2 { margin-bottom: 14px; }
        .pitcher-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .edge-row { font-size: 15px; color: #90caf9; margin-bottom: 8px; }
        .xwoba-row { display: flex; gap: 24px; font-size: 13px; color: #aaa; }
        .batter-tables { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 8px; }
        .note { font-size: 12px; color: #555; margin-bottom: 12px; }
        .meta-grid { display: grid; grid-template-columns: 200px 1fr; gap: 6px 16px; font-size: 13px; }
        dt { color: #888; }
        dd { color: #e6edf3; margin: 0; }
        .metadata { margin-top: 8px; }
        .state-message { margin-top: 40px; text-align: center; color: #666; font-size: 15px; }
        .state-message.error { color: #ef5350; }
        @media (max-width: 640px) {
          .pitcher-grid { grid-template-columns: 1fr; }
          .batter-tables { grid-template-columns: 1fr; }
        }
      `}</style>
    </>
  );
}

function PitcherCard({
  sp,
  teamAbbr,
  side,
}: {
  sp?: { pitcher_id: number; pitcher_name?: string; throws?: string };
  teamAbbr: string;
  side: string;
}) {
  return (
    <div className="pitcher-card">
      <div className="pitcher-side">{side} — {teamAbbr}</div>
      {sp ? (
        <>
          <div className="pitcher-name">
            {sp.pitcher_name ?? `ID ${sp.pitcher_id}`}
          </div>
          {sp.throws && (
            <div className="pitcher-meta">Throws: {sp.throws.toUpperCase()}</div>
          )}
        </>
      ) : (
        <div className="pitcher-tbd">SP TBD</div>
      )}

      <style jsx>{`
        .pitcher-card {
          background: #0d1117;
          border: 1px solid #21262d;
          border-radius: 8px;
          padding: 14px;
        }
        .pitcher-side { font-size: 11px; color: #666; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
        .pitcher-name { font-size: 16px; font-weight: 600; color: #fff; }
        .pitcher-meta { font-size: 12px; color: #888; margin-top: 4px; }
        .pitcher-tbd { font-size: 14px; color: #555; }
      `}</style>
    </div>
  );
}
