# ionic.games

Studio homepage for [ionic.games](https://ionic.games). Vite + TypeGPU, deployed to Cloudflare Pages (`ionic-games`).

## Click interest (D1)

Every `.game-card` click beacons `POST /api/click` (Pages Function) and inserts a row into the `ionic-games-interest` D1 database. There is no public letter board yet — query counts in the Cloudflare dashboard:

**Workers & Pages → D1 → `ionic-games-interest` → Console**

```sql
SELECT title, kind, COUNT(*) AS n
FROM clicks
GROUP BY title, kind
ORDER BY n DESC;
```

- Account: `4b1bc7e2a1fe0656005764a93810f674`
- Database name: `ionic-games-interest`
- Database id: `17d893bb-b576-4ecb-9d3c-1a94f4a0f1ee`

Schema (already applied remotely):

```bash
npx wrangler d1 execute ionic-games-interest --remote --file migrations/0001_create_clicks.sql
```

## Deploy

Needs a Cloudflare login (or `CLOUDFLARE_API_TOKEN`) on account `4b1bc7e2a1fe0656005764a93810f674`:

```bash
export CLOUDFLARE_ACCOUNT_ID=4b1bc7e2a1fe0656005764a93810f674
npm run deploy
```

That runs `npm run build` then `wrangler pages deploy dist --project-name=ionic-games`. The D1 binding `DB` comes from `wrangler.toml`.
