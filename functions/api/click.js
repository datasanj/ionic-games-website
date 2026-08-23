const TITLE_RE = /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/;
const KINDS = new Set(['live', 'coming_soon', 'wishlist']);

function corsHeaders(request) {
  const origin = request.headers.get('Origin');
  const self = new URL(request.url).origin;
  if (origin && origin !== self) return null;
  return {
    'Access-Control-Allow-Origin': origin || self,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    Vary: 'Origin',
  };
}

function json(body, status, headers) {
  return Response.json(body, { status, headers });
}

export function onRequestOptions({ request }) {
  const headers = corsHeaders(request);
  if (!headers) return new Response(null, { status: 403 });
  return new Response(null, { status: 204, headers });
}

export async function onRequestPost({ request, env }) {
  const headers = corsHeaders(request);
  if (!headers) return json({ ok: false }, 403, undefined);

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ ok: false }, 400, headers);
  }

  const title = typeof body?.title === 'string' ? body.title.trim() : '';
  if (!TITLE_RE.test(title)) {
    return json({ ok: false }, 400, headers);
  }

  const kind = typeof body?.kind === 'string' && body.kind ? body.kind : 'live';
  if (!KINDS.has(kind)) {
    return json({ ok: false }, 400, headers);
  }

  let href = null;
  if (typeof body?.href === 'string' && body.href) {
    href = body.href.slice(0, 512);
  }

  const referrer = request.headers.get('Referer')?.slice(0, 512) ?? null;
  const userAgent = request.headers.get('User-Agent')?.slice(0, 512) ?? null;

  await env.DB.prepare(
    `INSERT INTO clicks (title, href, kind, referrer, user_agent)
     VALUES (?, ?, ?, ?, ?)`,
  )
    .bind(title, href, kind, referrer, userAgent)
    .run();

  return json({ ok: true }, 200, headers);
}
