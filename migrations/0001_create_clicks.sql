CREATE TABLE IF NOT EXISTS clicks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  href TEXT,
  kind TEXT NOT NULL DEFAULT 'live', -- live | coming_soon | wishlist
  referrer TEXT,
  user_agent TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_clicks_title ON clicks(title);
CREATE INDEX IF NOT EXISTS idx_clicks_created ON clicks(created_at);
