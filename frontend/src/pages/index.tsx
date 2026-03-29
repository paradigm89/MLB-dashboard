/**
 * Today's Games Dashboard — polls /api/schedule/today every 5 minutes.
 * Shows all games with MatchupCards, a pinned RefreshStatusBar, and an ErrorBanner.
 */
import useSWR from "swr";
import Head from "next/head";
import MatchupCard from "@/components/MatchupCard";
import RefreshStatusBar from "@/components/RefreshStatusBar";
import ErrorBanner from "@/components/ErrorBanner";
import { api, TodayScheduleResponse } from "@/lib/api";

const FIVE_MINUTES = 5 * 60 * 1000;

export default function HomePage() {
  const { data, error, isLoading } = useSWR<TodayScheduleResponse>(
    "schedule/today",
    () => api.getTodaySchedule(),
    { refreshInterval: FIVE_MINUTES, revalidateOnFocus: true }
  );

  const games = data?.games ?? [];
  const hasGames = games.length > 0;

  return (
    <>
      <Head>
        <title>MLB Dashboard — Today&apos;s Games</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <RefreshStatusBar />

      <main className="page-container">
        <ErrorBanner />

        <div className="section-header">
          <h1>
            Today&apos;s Games
            {data && (
              <span className="game-count">
                {" "}— {data.total_games} game{data.total_games !== 1 ? "s" : ""}
              </span>
            )}
          </h1>
          {data && (
            <span className="lineup-summary">
              <span className="tag tag-green">{data.lineups_confirmed} confirmed</span>
              {data.lineups_projected > 0 && (
                <span className="tag tag-amber" style={{ marginLeft: 6 }}>
                  {data.lineups_projected} projected
                </span>
              )}
            </span>
          )}
        </div>

        {isLoading && (
          <div className="state-message">Loading today&apos;s games…</div>
        )}

        {error && !isLoading && (
          <div className="state-message error">
            Could not load games. Is the backend running?
          </div>
        )}

        {!isLoading && !error && !hasGames && (
          <div className="state-message">No games scheduled today.</div>
        )}

        {hasGames && (
          <div className="games-grid">
            {games.map((game) => (
              <MatchupCard key={game.game_pk} game={game} />
            ))}
          </div>
        )}
      </main>

      <style jsx>{`
        .game-count { font-weight: 400; color: #888; font-size: 16px; }
        .lineup-summary { display: flex; align-items: center; }
        .state-message {
          margin-top: 40px;
          text-align: center;
          color: #666;
          font-size: 15px;
        }
        .state-message.error { color: #ef5350; }
      `}</style>
    </>
  );
}
